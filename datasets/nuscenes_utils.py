import os.path as osp
from abc import ABC

import numpy as np
import numpy.linalg as LA
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from PIL import Image
from pyquaternion import Quaternion

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


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
    assert points.shape == (points.shape[0],
                            3), f"{points.shape} is not (N, 3)"
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
        cs_record = nusc.get('calibrated_sensor',
                             record['calibrated_sensor_token'])
        self.ego_from_self = transform_matrix(
            cs_record['translation'], Quaternion(cs_record['rotation']))  # 4x4
        ego_record = nusc.get(
            'ego_pose',
            record['ego_pose_token'])  # ego pose @ timestamp of sensor
        self.glob_from_ego = transform_matrix(
            ego_record['translation'],
            Quaternion(ego_record['rotation']))  # 4x4
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
        self.img_wh = np.array([record['width'], record['height']],
                               dtype=float)
        self.img = Image.open(osp.join(nusc.dataroot, record['filename']))
        cs_record = nusc.get('calibrated_sensor',
                             record['calibrated_sensor_token'])
        self.cam_K = np.array(cs_record['camera_intrinsic'])

    def project_pts3d(self, pc, depth_thres=1e-3):
        """
        Project a set of 3D points onto camera's image plane
        Args:
            pc (np.ndarray): (N, 3) - X, Y, Z; must be in the CAMERA frame
            depth_thres (float): to determine invalide points
        Returns:
            uv (np.ndarray): (N, 2) - pixel coordinates
                (u == x-axis, v == y-axis, v for vertical)
            mask_in_img (np.ndarray): (N,) - bool, True for pts that have
                projection inside of camera image
        """
        # id points have zero or negative depth
        mask_valid = pc[:, 2] > depth_thres

        # projection
        out = np.zeros((pc.shape[0], 2), dtype=float) - 10
        uv = view_points(pc[mask_valid].T, self.cam_K,
                         normalize=True)  # (3, N)
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
            pc (np.ndarray): (N, 4 (+1))
                [X, Y, Z in LiDAR frame, intensity, time lag w.r.t timestamp
                of keyframe]
        """
        if num_sweeps is not None:
            assert sample_record is not None
            assert num_sweeps <= 10
            pc, times = LidarPointCloud.from_file_multisweep(
                nusc,
                sample_record,
                'LIDAR_TOP',
                'LIDAR_TOP',
                nsweeps=num_sweeps)
            out = np.vstack([pc.points[:4, :], times])  # (5, N)
            return out.T  # (N, 5)
        else:
            lidar_record = nusc.get('sample_data',
                                    sample_record['data']['LIDAR_TOP'])
            pc = LidarPointCloud.from_file(
                osp.join(nusc.dataroot,
                         lidar_record['filename']))  # pc is in LIDAR frame
            return pc.points[:4, :].T  # (N, 4) - X, Y, Z in LIDAR_TOP frame


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
    assert np.all(mask_inside), "pts_uv must be all inside image"

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
        pts_feat = w_ff * img[v_floor, u_floor] + w_cc * img[v_ceil, u_ceil] \
            + w_cf * img[v_ceil, u_floor] + w_fc * img[v_floor, u_ceil]
        return pts_feat  # (N, C)

    elif method == 'nearest':
        uv_ = np.round(pts_uv).astype(int)
        return img[uv_[:, 1], uv_[:, 0]]  # (N, C)


def tf(translation, rotation) -> np.ndarray:
    """
    Build transformation matrix
    Return:
        tf_mat: (4, 4) homogeneous transformation matrix
    """
    if not isinstance(rotation, Quaternion):
        assert isinstance(rotation, list) or isinstance(
            rotation, np.ndarray), f"{type(rotation)} is not supported"
        rotation = Quaternion(rotation)
    tf_mat = np.eye(4)
    tf_mat[:3, :3] = rotation.rotation_matrix
    tf_mat[:3, 3] = translation
    return tf_mat


def apply_tf(tf: np.ndarray, points: np.ndarray, in_place=False):
    assert points.shape[
        1] >= 3, f"expect points.shape[1] >= 3, get {points.shape[1]}"
    assert tf.shape == (4, 4), f"expect tf.shape == 4, get {tf.shape}"
    xyz1 = np.pad(points[:, :3],
                  pad_width=[(0, 0), (0, 1)],
                  constant_values=1.0)  # (N, 4)
    if in_place:
        points[:, :3] = (xyz1 @ tf.T)[:, :3]
    else:
        return (xyz1 @ tf.T)[:, :3]


def get_sweeps_token(nusc: NuScenes, curr_sd_token: str, n_sweeps: int,
                     return_time_lag: bool, return_sweep_idx: bool) -> list:
    ref_sd_rec = nusc.get('sample_data', curr_sd_token)
    ref_time = ref_sd_rec['timestamp'] * 1e-6
    sd_tokens_times = []
    for s_idx in range(n_sweeps):
        curr_sd = nusc.get('sample_data', curr_sd_token)
        if not return_sweep_idx:
            sd_tokens_times.append(
                (curr_sd_token, ref_time - curr_sd['timestamp'] * 1e-6))
        else:
            sd_tokens_times.append(
                (curr_sd_token, ref_time - curr_sd['timestamp'] * 1e-6,
                 n_sweeps - 1 - s_idx))
        # s_idx: the higher, the closer to the current
        # move to previous
        if curr_sd['prev'] != '':
            curr_sd_token = curr_sd['prev']

    # organize from PAST to PRESENCE
    sd_tokens_times.reverse()

    if return_time_lag:
        return sd_tokens_times
    else:
        return [token for token, _ in sd_tokens_times]


def get_nuscenes_sensor_pose_in_ego_vehicle(nusc: NuScenes,
                                            curr_sd_token: str) -> np.ndarray:
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_cs_rec = nusc.get('calibrated_sensor',
                           curr_rec['calibrated_sensor_token'])
    ego_from_curr = tf(curr_cs_rec['translation'], curr_cs_rec['rotation'])
    return ego_from_curr


def get_nuscenes_sensor_pose_in_global(nusc: NuScenes,
                                       curr_sd_token: str) -> np.ndarray:
    ego_from_curr = get_nuscenes_sensor_pose_in_ego_vehicle(
        nusc, curr_sd_token)
    curr_rec = nusc.get('sample_data', curr_sd_token)
    curr_ego_rec = nusc.get('ego_pose', curr_rec['ego_pose_token'])
    glob_from_ego = tf(curr_ego_rec['translation'], curr_ego_rec['rotation'])
    glob_from_curr = glob_from_ego @ ego_from_curr
    return glob_from_curr


def get_sample_data_point_cloud(nusc: NuScenes, sample_data_token: str,
                                time_lag: float, sweep_idx: int) -> np.ndarray:
    """
    Returns:
        pc: (N, 6) - (x, y, z, intensity, [time, sweep_idx])
    """
    pcfile = nusc.get_sample_data_path(sample_data_token)
    pc = np.fromfile(pcfile, dtype=np.float32, count=-1).reshape(
        [-1, 5])[:, :4]  # (N, 4) - (x, y, z, intensity)
    pc = np.pad(pc,
                pad_width=[(0, 0), (0, 2)],
                mode='constant',
                constant_values=0)  # (N, 6)
    pc[:, -2] = time_lag
    pc[:, -1] = sweep_idx
    return pc


def remove_ego_vehicle_points(points: np.ndarray, center_radius) -> np.ndarray:
    dist_xy = LA.norm(points[:, :2], axis=1)
    return points[dist_xy > center_radius]


def find_points_in_box(points: np.ndarray, target_from_box: np.ndarray,
                       dxdydz: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Args:
        points: (N, 3 + C) - x, y, z in target frame
        target_from_box: (4, 4) - transformation from box frame to target frame
        dxdydz: box's size
        tolerance:
    """
    box_points = apply_tf(LA.inv(target_from_box), points[:, :3])  # (N, 3)
    mask_inside = np.all(np.abs(box_points / dxdydz) < (0.5 + tolerance),
                         axis=1)  # (N,)
    return mask_inside


def inst_centric_get_sweeps(nusc: NuScenes, sample_token: str, n_sweeps: int,
                            center_radius: float, in_box_tolerance: float,
                            return_instances_last_box: bool,
                            point_cloud_range: list, detection_classes: tuple,
                            map_point_feat2idx: dict) -> dict:
    """
    Each object is denoted by a unique 'instance idx'
    Each point is associated with an 'instance idx'

    Returns:
        {
            'points' (np.ndarray): (N, 8)
                [0] --> [2]: (x, y, z)
                [3]        : intensity
                [4]        : time-lag
                [5]        : sweep_idx
                [6]        : instance_idx
                [7]        : class_idx
            'instances_tf' (np.ndarray): (N_inst, N_sweep, 4, 4)
            'instances_last_box' (np.ndarray): (N_inst, 9)
                [0] --> [2]: bbox center (x, y, z)
                [3] --> [5]: bbox size (dx, dy, dz)
                [6]        : bbox yaw angle [rad?]
                [7] --> [8]: bbox velocities vx, vy ?
            'instances_name' (np.ndarray): (N_inst,) - index of class of each
                                                       instance
                [0]: 'car'
                [1]: 'truck'
                [2]: 'construction_vehicle'
                [3]: 'bus'
                [4]: 'trailer'
                [5]: 'motorcycle'
                [6]: 'bicycle'
                [7]: 'pedestrian'
        }
    """
    sample_rec = nusc.get('sample', sample_token)
    target_sd_token = sample_rec['data']['LIDAR_TOP']
    sd_tokens_times = get_sweeps_token(nusc,
                                       target_sd_token,
                                       n_sweeps,
                                       return_time_lag=True,
                                       return_sweep_idx=True)

    target_from_glob = LA.inv(
        get_nuscenes_sensor_pose_in_global(nusc, target_sd_token))

    inst_token_2_index = dict()  # use this to access "instances"
    inst_idx = 0
    instances = list()  # for each instance, store list of poses
    instances_sweep_indices = list(
    )  # for each instance, store list of sweep index
    instances_size = list(
    )  # for each instance, store its sizes (dx, dy ,dz) == l, w, h
    instances_name = list()  # for each instance, store its detection name
    inst_tk_2_sample_tk = dict(
    )  # to store the sample_tk where an inst last appears
    inst_latest_anno_tk = list()
    instances_token = list()
    instances_center = list()
    all_points = []

    for sd_token, time_lag, s_idx in sd_tokens_times:
        glob_from_cur = get_nuscenes_sensor_pose_in_global(nusc, sd_token)
        cur_points = get_sample_data_point_cloud(
            nusc, sd_token, time_lag, s_idx)  # (N, 6), in "cur" frame
        cur_points = remove_ego_vehicle_points(cur_points, center_radius)

        # map to target
        cur_points[:, :3] = apply_tf(
            target_from_glob @ glob_from_cur,
            cur_points[:, :3])  # (N, 6) in target frame

        # pad points with instances index & class index
        cur_points = np.pad(cur_points,
                            pad_width=[(0, 0), (0, 2)],
                            constant_values=-1)

        boxes = nusc.get_boxes(sd_token)

        for b_idx, box in enumerate(boxes):
            box_det_name = map_name_from_general_to_detection[box.name]
            if box_det_name not in detection_classes:
                continue

            anno_rec = nusc.get('sample_annotation', box.token)

            if anno_rec['num_lidar_pts'] < 1:
                continue

            # map box to target
            glob_from_box = tf(box.center, box.orientation)
            target_from_box = target_from_glob @ glob_from_box

            # find points inside this box
            mask_in = find_points_in_box(
                cur_points, target_from_box,
                np.array([box.wlh[1], box.wlh[0], box.wlh[2]]),
                in_box_tolerance)
            if not np.any(mask_in):
                # empty box -> move on to next box
                continue

            # store box's pose according to the instance which it belongs to
            inst_token = anno_rec['instance_token']
            if inst_token not in inst_token_2_index:
                # new instance
                inst_token_2_index[inst_token] = inst_idx
                inst_idx += 1
                instances.append([target_from_box])
                instances_sweep_indices.append([s_idx])
                # store size
                instances_size.append([box.wlh[1], box.wlh[0], box.wlh[2]])
                # store name
                instances_name.append(detection_classes.index(box_det_name))
                # store anno token to be used later for calculating box's vel
                inst_latest_anno_tk.append(anno_rec['token'])
            else:
                cur_instance_idx = inst_token_2_index[inst_token]
                instances[cur_instance_idx].append(target_from_box)
                instances_sweep_indices[cur_instance_idx].append(s_idx)
                # update anno token to be used later for calculating box's vel
                inst_latest_anno_tk[cur_instance_idx] = anno_rec['token']

            inst_tk_2_sample_tk[inst_token] = anno_rec['sample_token']

            # set points' instance index
            cur_points[mask_in,
                       map_point_feat2idx['inst_idx']] = inst_token_2_index[
                           inst_token]
            # set points' class index
            cur_points[
                mask_in,
                map_point_feat2idx['cls_idx']] = detection_classes.index(
                    box_det_name)

            # Store instance token
            instances_token.append(inst_token)
            instances_center.append(box.center)

        all_points.append(cur_points)

    all_points = np.concatenate(all_points, axis=0)

    # merge instances & instances_sweep_indices
    instances_tf = np.zeros(
        (len(instances), n_sweeps, 4,
         4))  # rigid tf that map fg points to their correct position
    for inst_idx in range(len(instances)):
        inst_poses = instances[inst_idx]  # list
        inst_sweep_ids = instances_sweep_indices[inst_idx]  # list
        for sw_i, pose in zip(inst_sweep_ids, inst_poses):
            instances_tf[inst_idx, sw_i] = inst_poses[-1] @ LA.inv(pose)

    out = {
        'points': all_points,
        'instances_token': instances_token,
        'instances_center': instances_center,
    }

    if return_instances_last_box:
        assert point_cloud_range is not None
        if not isinstance(point_cloud_range, np.ndarray):
            point_cloud_range = np.array(point_cloud_range)
        instances_last_box = np.zeros(
            (len(instances),
             9))  # 9 := c_x, c_y, c_z, d_x, d_y, d_z, yaw, vx, vy

        for _idx, (_size, _poses) in enumerate(zip(instances_size, instances)):
            # find the pose that has center inside point cloud range & is
            # closest to the target time step
            # if couldn't find any, take the 1st pose (i.e. the furthest into
            # the past)
            chosen_pose_idx = 0
            for pose_idx in range(-1, -len(_poses) - 1, -1):
                if np.all(
                        np.logical_and(
                            _poses[pose_idx][:3, -1] >= point_cloud_range[:3],
                            _poses[pose_idx][:3, -1] <
                            point_cloud_range[3:] - 1e-2)):
                    chosen_pose_idx = pose_idx
                    break
            yaw = np.arctan2(_poses[chosen_pose_idx][1, 0],
                             _poses[chosen_pose_idx][0, 0])
            instances_last_box[_idx, :3] = _poses[chosen_pose_idx][:3, -1]
            instances_last_box[_idx, 3:6] = np.array(_size)
            instances_last_box[_idx, 6] = yaw

            # instance velocity
            velo = nusc.box_velocity(inst_latest_anno_tk[_idx]).reshape(
                1, 3)  # - [vx, vy, vz] in global frame
            velo = apply_tf(target_from_glob,
                            velo).reshape(3)[:2]  # [vx, vy] in target frame
            instances_last_box[_idx, 7:9] = velo

        out['instances_last_box'] = instances_last_box
        out['instances_name'] = np.array(instances_name)

    return out


def load_data_to_tensor(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            raise ValueError("images are not supported")
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int()
        else:
            batch_dict[key] = torch.from_numpy(val).float()
