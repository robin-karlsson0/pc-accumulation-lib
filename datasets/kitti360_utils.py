import os

import numpy as np


def read_pc_bin_file(path):
    """
    Ref: https://stackoverflow.com/questions/60506331/conversion-of-binary-lidar-data-bin-to-point-cloud-data-pcd-format
    """
    pc = np.fromfile(path, dtype=np.float32)
    pc = pc.reshape((-1, 4))
    return pc


def filter_semseg_pc(pc, filters):
    for filter in filters:
        mask = pc[:, -1] != filter
        pc = pc[mask]

    return pc


def extract_seseg_pc(pc, filter):
    mask = pc[:, -1] == filter
    pc = pc[mask]

    return pc


def get_transf_matrices(kitti360_path: str):
    '''
    Returns homogeneous transformation matrices.
        H_cam_velo: Camera coords --> Velodyne coords
        H_velo_cam: Velodyne coords --> Camera coords
    '''
    calib_file = os.path.join(kitti360_path, 'calibration',
                              'calib_cam_to_velo.txt')
    # (3, 4) matrix
    H_cam_velo = np.genfromtxt(calib_file, delimiter=" ")
    H_cam_velo = H_cam_velo.reshape((3, 4))
    # --> (4, 4) matrix
    H_cam_velo = np.concatenate(
        (H_cam_velo, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    # Invert homogeneous transformation matrix
    H_velo_cam = np.linalg.inv(H_cam_velo)

    return H_cam_velo, H_velo_cam


def get_camera_intrinsics(kitti360_path: str):
    '''
    '''
    calib_file = os.path.join(kitti360_path, 'calibration', 'perspective.txt')
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split(':')
        if line[0] == 'P_rect_00':
            # Process P matrix
            num_strs = line[1].split(' ')
            num_strs = [num_str.replace('\n', '') for num_str in num_strs]
            num_strs.remove('')
            P_cam_frame = np.array(num_strs, dtype=float)
            P_cam_frame = P_cam_frame.reshape((3, 4))

            return P_cam_frame

    raise Exception('Did not find \'P_rect_00\' entry in calibration file.')


def idx2str(idx: int):
    '''
    Convert frame idx integer to string with leading zeros
    '''
    return f"{idx:010d}"
