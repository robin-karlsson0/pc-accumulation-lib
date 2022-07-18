import numpy as np
from PIL import Image
import os.path as osp
from pyquaternion import Quaternion
from abc import ABC

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix, view_points


def print_dict(d: dict):
    for k, v in d.items():
        print(f'{k}: {v}')


class NuScenesSensor(ABC):
    """
    Abstract class for NuScenes sensors (camera, lidar)
    """
    def __init__(self, nusc, record):
        """
        Args:
            nusc (NuScenes):
            record (dict): sensor record
        """
        self.token = record['token']
        self.channel = record['channel']
        cs_record = nusc.get('calibrated_sensor', record['calibrated_sensor_token'])
        self.ego_from_self = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']))  # 4x4
        ego_record = nusc.get('ego_pose', record['ego_pose_token'])  # ego pose @ timestamp of sensor
        self.glob_from_ego = transform_matrix(ego_record['translation'], Quaternion(ego_record['rotation']))  # 4x4
        self.glob_from_self = self.glob_from_ego @ self.ego_from_self
        # --------
        # modality-specific attributes
        # --------
        self.img = None
        self.img_hw = None
        self.cam_K = None
        self.pc = None


class NuScenesCamera(NuScenesSensor):
    def __init__(self, nusc, record):
        """
        Args:
            nusc (NuScenes):
            record (dict): sensor record
        """
        super().__init__(nusc, record)
        self.img_wh = np.array([record['width'], record['height']], dtype=float)
        self.img = Image.open(osp.join(nusc.dataroot, record['filename']))
        cs_record = nusc.get('calibrated_sensor', record['calibrated_sensor_token'])
        self.cam_K = np.array(cs_record['camera_intrinsic'])

    def project_pts3d(self, pc, depth_thres=1e-3):
        """
        Project a set of 3D points onto camera's image plane
        Args:
            pc (np.ndarray): (N, 3) - X, Y, Z; must be in the CAMERA frame
            depth_thres (float): to determine invalide points

        Returns:
            uv (np.ndarray): (N, 2) - pixel coordinates (u == x-axis, v == y-axis, v for vertical)
            mask_in_img (np.ndarray): (N,) - bool, True for pts that have projection inside of camera image
        """
        # id points have zero or negative depth
        mask_valid = pc[:, 2] > depth_thres

        # projection
        out = np.zeros((pc.shape[0], 2), dtype=float) - 10
        uv = view_points(pc[mask_valid].T, self.cam_K, normalize=True)  # (3, N)
        out[mask_valid] = uv[:2, :].T

        # id points inside of image
        mask_in_img = (out > 1) & (out < self.img_wh - 1)
        mask_in_img = np.all(mask_in_img, axis=1) & mask_valid
        return out, mask_in_img

    def get_pts_feat(self, pts_uv, cam_feat=None, method='bilinear'):
        """
        Get camera feat for 3d points using their projectd coordinate
        Args:
            pts_uv (np.ndarray): (N, 2) - float
            cam_feat (np.ndarray): feat map where pts get feature from, If not provided, takes camera image as value
            method (str): support bilinear & nearest neighbor

        Returns:
            pc_feat (np.ndarray): (N, C), C is the number of channels of feat map
        """
        assert method in ('bilinear', 'nearest'), f"{method} is not supported"
        mask_inside = (pts_uv > 1) & (pts_uv < self.img_wh - 1)
        assert np.all(mask_inside), f"pts_uv must be all inside image"

        if cam_feat is None:
            cam_feat = np.array(self.img)  # (H, W, 3)
            cam_feat = cam_feat.transpose((2, 0, 1))  # (3, H, W)

        if method == 'bilinear':
            u, v = pts_uv[:, 0], pts_uv[:, 1]
            u_floor, u_ceil = np.floor(u), np.ceil(u)
            v_floor, v_ceil = np.floor(v), np.ceil(v)
            total = (u_ceil - u_floor) * (v_ceil - v_floor)
            w_ff = (u_ceil - u) * (v_ceil - v) / total  # area (uv, uv_cc)
            w_cc = (u - u_floor) * (v - v_floor) / total  # area (uv, uv_ff)
            w_fc = (u - u_floor) * (v_ceil - v) / total  # area (uv, uv_fc)
            w_cf = 1. - (w_ff + w_cc + w_fc)
            u_floor, v_floor = u_floor.astype(int), v_floor.astype(int)
            u_ceil, v_ceil = u_ceil.astype(int), v_ceil.astype(int)
            pts_feat = w_ff * cam_feat[:, v_floor, u_floor] + w_cc * cam_feat[:, v_ceil, u_ceil] + \
                       w_cf * cam_feat[:, v_ceil, u_floor] + w_fc * cam_feat[:, v_floor, u_ceil]  # (C, N)
            return pts_feat.transpose((1, 0))  # (N, C)

        elif method == 'nearest':
            uv_ = np.round(pts_uv).astype(int)
            return cam_feat[:, uv_[:, 1], uv_[:, 0]].transpose((1, 0))  # (N, C)
