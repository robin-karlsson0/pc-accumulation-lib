import os

import numpy as np

from datasets.kitti360_utils import get_camera_intrinsics, get_transf_matrices
from obs_dataloaders.kitti360_obs_dataloader import Kitti360Dataloader
from sem_pc_accum import SemanticPointCloudAccumulator

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
    sem_idxs = {'road': 0, 'car': 13, 'truck': 14, 'bus': 15, 'motorcycle': 17}

    accum_horizon_dist = 200  # From front to back

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

    ####################
    #  BEV parameters
    ####################
    bevs_per_sample = 1
    bev_horizon_dist = 60
    bev_dist_between_samples = 5.
    voxel_size = 0.1

    bev_params = {
        'type': 'sem',  # Options: ['sem', 'rgb']
        'view_size': 80,
        'pixel_size': 512,
        'max_trans_radius': 0.,
        'zoom_thresh': 0.,
    }

    savedir = 'bevs_online'
    subdir_size = 1000
    viz_to_disk = True  # For debugging purposes

    # Initialize accumulator
    sem_pc_accum = SemanticPointCloudAccumulator(
        accum_horizon_dist,
        calib_params,
        icp_threshold,
        semseg_onnx_path,
        filters,
        sem_idxs,
        bev_params,
    )

    #################
    #  Sample data
    #################
    batch_size = 10
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
    start_idxs = [130]
    # [130, 4613, 40, 90, 50, 120, 0, 90, 0]
    end_idxs = [11400]
    # [11400, 1899, 770, 11530, 6660, 9698, 2960, 13945, 3540]

    dataloader = Kitti360Dataloader(kitti360_path, batch_size, sequences,
                                    start_idxs, end_idxs)

    ###################
    #  Generate BEVs
    ###################
    bev_idx = 0
    subdir_idx = 0

    for observations in dataloader:

        sem_pc_accum.integrate(observations)

        bev = sem_pc_accum.generate_bev()
        bev = bev[0]

        # Store BEV samples
        if bev_idx > 1000:
            bev_idx = 0
            subdir_idx += 1
        filename = f'bev_{bev_idx}.pkl'
        output_path = f'./{savedir}/subdir{subdir_idx:03d}/'

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        # Visualize BEV samples
        if viz_to_disk:
            viz_file = os.path.join(output_path, f'viz_{bev_idx}.png')
            sem_pc_accum.viz_bev(bev, viz_file)

        bev_idx += 1
