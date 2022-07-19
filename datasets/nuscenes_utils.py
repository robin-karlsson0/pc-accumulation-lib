import numpy as np
from PIL import Image
import os.path as osp
from pyquaternion import Quaternion
from abc import ABC

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from nuscenes.utils.data_classes import LidarPointCloud


def print_dict(d: dict):
    for k, v in d.items():
        print(f'{k}: {v}')


def homo_transform(tf, points):
    """
    Apply homogeneous transformation to a set of points
    Args:
        tf (np.ndarray): 4x4
        points (np.ndarray): (N, 3) - X, Y, Z

    Returns:
        transformed_points (np.ndarray): (N, 3)
    """
    assert tf.shape == (4, 4), f"{tf.shape} is not (4, 4)"
    assert points.shape == (points.shape[0], 3), f"{points.shape} is not (N, 3)"
    _pts = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    _pts = tf @ _pts.T
    return _pts[:3, :].T


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


class NuScenesLidar(NuScenesSensor):
    def __init__(self, nusc, lidar_record):
        """
        Args:
            nusc (NuScenes):
            lidar_record (dict):
        """
        super().__init__(nusc, lidar_record)

    @staticmethod
    def get_pointcloud(nusc, sample_record, num_sweeps=None):
        """

        Args:
            nusc (NuScenes):
            sample_record (dict): record of the entire keyframe
            num_sweeps (int): < 10
        Returns:
            pc (np.ndarray): (N, 3 (+1)) - X, Y, Z in LiDAR frame (and time lag w.r.t timestamp of keyframe)
        """
        if num_sweeps is not None:
            assert sample_record is not None
            assert num_sweeps <= 10
            pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_record, 'LIDAR_TOP', 'LIDAR_TOP',
                                                             nsweeps=num_sweeps)
            out = np.vstack([pc.points[:3, :], times])  # (4, N)
            return out.T  # (N, 4)
        else:
            lidar_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, lidar_record['filename']))  # pc is in LIDAR frame
            return pc.points[:3, :].T  # (N, 3) - X, Y, Z in LIDAR_TOP frame


def pts_feat_from_img(pts_uv, img, method='bilinear'):
    """
    Get camera feat for 3d points using their projectd coordinate
    Args:
        pts_uv (np.ndarray): (N, 2) - float
        img (np.ndarray): (H, W, C) - feat map where pts get feature from
        method (str): support bilinear & nearest neighbor

    Returns:
        pc_feat (np.ndarray): (N, C), C is the number of channels of feat map
    """
    assert isinstance(img, np.ndarray), f"{type(img)} is not supported"
    assert method in ('bilinear', 'nearest'), f"{method} is not supported"
    img_wh = np.array([img.shape[1], img.shape[0]], dtype=float)
    mask_inside = (pts_uv > 1) & (pts_uv < img_wh - 1)
    assert np.all(mask_inside), f"pts_uv must be all inside image"

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
        pts_feat = w_ff * img[v_floor, u_floor] + w_cc * img[v_ceil, u_ceil] + \
                   w_cf * img[v_ceil, u_floor] + w_fc * img[v_floor, u_ceil]  # (N, C)
        return pts_feat  # (N, C)

    elif method == 'nearest':
        uv_ = np.round(pts_uv).astype(int)
        return img[uv_[:, 1], uv_[:, 0]]  # (N, C)
