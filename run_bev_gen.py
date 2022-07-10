import os

import numpy as np

from datasets.kitti360_utils import get_camera_intrinsics, get_transf_matrices
from obs_dataloaders.kitti360_obs_dataloader import Kitti360Dataloader
from sem_pc_accum import SemanticPointCloudAccumulator


def calc_forward_dist(poses: np.array) -> np.array:
    '''
    Calculates the Euclidean distance from first pose to all other poses.

    Args:
        poses: Spatial positions as matrix w. dim (N, 3).

    Returns:
        dists: Euclidean distances as vector w. dim (N).

    '''
    dists = poses - poses[0]
    dists = np.sum(dists * dists, 1)
    dists = np.sqrt(dists)
    return dists


def dist(pose_0: np.array, pose_1: np.array):
    '''
        Returns the Euclidean distance between two poses.
            dist = sqrt( dx**2 + dy**2 )

        Args:
            pose_0: 1D vector [x, y]
            pose_1:
        '''
    dist = np.sqrt(np.sum((pose_1 - pose_0)**2))
    return dist


if __name__ == '__main__':

    # Path to dataset root directory
    kitti360_path = '/home/robin/datasets/KITTI-360'
    # Path to ONNX semantic segmentation model
    semseg_onnx_path = 'semseg_rn50_160k_cm.onnx'
    # Semantic exclusion filters
    # 0 : Road
    # 1 : Sidewalk
    # 2 : Building
    # 3 : Wall
    # 4 : Fence
    # 5 : Pole
    # 6 : Traffic Light
    # 7 : Traffic Sign
    # 8 : Vegetation
    # 9 : Terrain
    # 10 : Sky
    # 11 : Person
    # 12 : Rider
    # 13 : Car
    # 14 : Truck
    # 15 : Bus
    # 16 : Train
    # 17 : Motorcycle
    # 18 : Bicycle
    filters = [10, 11, 12, 16, 18]

    accum_horizon_dist = 80

    ######################
    #  Calibration info
    ######################
    h_cam_velo, h_velo_cam = get_transf_matrices(kitti360_path)
    p_cam_frame = get_camera_intrinsics(kitti360_path)
    p_velo_frame = np.matmul(p_cam_frame, h_velo_cam)
    c_x = p_cam_frame[0, 2]
    c_y = p_cam_frame[1, 2]
    f_x = p_cam_frame[0, 0]
    f_y = p_cam_frame[1, 1]

    calib_params = {}
    calib_params['h_velo_cam'] = h_velo_cam
    calib_params['p_cam_frame'] = p_cam_frame
    calib_params['p_velo_frame'] = p_velo_frame
    calib_params['c_x'] = c_x
    calib_params['c_y'] = c_y
    calib_params['f_x'] = f_x
    calib_params['f_y'] = f_y

    ####################
    #  ICP parameters
    ####################
    icp_threshold = 1e3

    # Initialize accumulator
    sem_pc_accum = SemanticPointCloudAccumulator(accum_horizon_dist,
                                                 calib_params, icp_threshold,
                                                 semseg_onnx_path, filters)

    #################
    #  Sample data
    #################
    batch_size = 50  # 10
    sequences = [
        '2013_05_28_drive_0000_sync',
        # '2013_05_28_drive_0002_sync',
        # '2013_05_28_drive_0003_sync',
        # '2013_05_28_drive_0004_sync',
        # '2013_05_28_drive_0005_sync',
        # '2013_05_28_drive_0006_sync',
        # '2013_05_28_drive_0007_sync',
        # '2013_05_28_drive_0009_sync',
        # '2013_05_28_drive_0010_sync',
    ]
    start_idxs = [200]  # [130, 4613, 40, 90, 50, 120, 0, 90, 0]
    end_idxs = [250
                ]  # [11400, 1899, 770, 11530, 6660, 9698, 2960, 13945, 3540]

    dataloader = Kitti360Dataloader(kitti360_path, batch_size, sequences,
                                    start_idxs, end_idxs)

    ####################
    #  BEV parameters
    ####################
    bevs_per_sample = 10
    bev_horizon_dist = 5  # 40.
    voxel_size = 0.1

    aug_params = {
        'max_translation_radius': 10,
        'zoom_threshold': 0.25,
    }
    bev_params = {
        'view_size': 80,
        'pixel_size': 512,
    }

    savedir = 'bevs'
    subdir_size = 1000
    viz_to_disk = True  # For debugging purposes

    ###################
    #  Generate BEVs
    ###################
    bev_idx = 0
    subdir_idx = 0
    bev_count = 0

    for observations in dataloader:

        sem_pc_accum.integrate(observations)

        pose_past = sem_pc_accum.get_pose(0)  # (3)
        pose_future = sem_pc_accum.get_pose(-1)  # (3)

        poses = sem_pc_accum.get_pose()  # (N, 3)

        # Find first idx with sufficient distance to past horizon
        d_past2idx = calc_forward_dist(poses)
        d_ph_margin = d_past2idx - bev_horizon_dist
        d_ph_margin[d_ph_margin < 0] = np.max(d_ph_margin)
        present_idx = np.argmin(d_ph_margin)

        pose_present = sem_pc_accum.get_pose(present_idx)

        # Check sufficient distances to past and future horizon
        d_past2present = dist(pose_past, pose_present)
        if d_past2present < bev_horizon_dist:
            continue
        d_present2future = dist(pose_present, pose_future)
        if d_present2future < bev_horizon_dist:
            continue

        bevs = sem_pc_accum.generate_bev(present_idx,
                                         bevs_per_sample,
                                         bev_params,
                                         aug_params=aug_params,
                                         gen_future=True)

        for bev in bevs:

            # Store BEV samples
            if bev_idx > 1000:
                bev_idx = 0
                subdir_idx += 1
            filename = f'bev_{bev_idx}.pkl'
            output_path = f'./{savedir}/subdir{subdir_idx:03d}/'

            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            sem_pc_accum.write_compressed_pickle(bev, filename, output_path)

            bev_idx += 1
            bev_count += 1
