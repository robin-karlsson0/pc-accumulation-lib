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
    bev_params = {'type': None}

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
    kitti360_path = '/home/robin/datasets/KITTI-360'
    batch_size = 50
    sequences = ['2013_05_28_drive_0000_sync']
    start_idxs = [200]
    end_idxs = [250]

    dataloader = Kitti360Dataloader(kitti360_path, batch_size, sequences,
                                    start_idxs, end_idxs)

    ############################
    #  Integrate observations
    ############################
    for observations in dataloader:

        sem_pc_accum.integrate(observations)

        sem_pc_accum.viz_sem_vec_space()
