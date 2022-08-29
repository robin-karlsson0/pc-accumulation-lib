import argparse

import numpy as np

from datasets.kitti360_utils import get_camera_intrinsics, get_transf_matrices
from nuscenes_sem_pc_accum import NuScenesSemanticPointCloudAccumulator
from obs_dataloaders.nuscenes_obs_dataloader import NuScenesDataloader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('nuscenes_path',
                        type=str,
                        help='Absolute path to dataset root (nuscenes/).')
    parser.add_argument(
        'semseg_onnx_path',
        type=str,
        help='Relative path to a semantic segmentation ONNX model.')
    parser.add_argument('--nuscenes_version', type=str, default='v1.0-mini')
    parser.add_argument('--accumulation_horizon',
                        type=int,
                        default=200,
                        help='Number of point clouds to accumulate.')
    parser.add_argument('--icp_threshold', type=float, default=1e3)

    args = parser.parse_args()

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

    ####################
    #  BEV parameters
    ####################
    bev_params = {'type': None}

    # Initialize accumulator
    sem_pc_accum = NuScenesSemanticPointCloudAccumulator(
        args.accumulation_horizon,
        args.icp_threshold,
        args.semseg_onnx_path,
        filters,
        sem_idxs,
        bev_params,
    )

    #################
    #  Sample data
    #################
    batch_size = 10
    num_sweeps = 1
    scene_ids = [0]
    dataloader = NuScenesDataloader(
        args.nuscenes_path,
        scene_ids,
        batch_size,
        num_sweeps,
        args.nuscenes_version,
    )

    ############################
    #  Integrate observations
    ############################
    for observations in dataloader:

        sem_pc_accum.integrate(observations)

        sem_pc_accum.viz_sem_vec_space()
