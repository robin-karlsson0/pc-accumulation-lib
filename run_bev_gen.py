import argparse
import os

import numpy as np

from datasets.kitti360_utils import get_camera_intrinsics, get_transf_matrices
from obs_dataloaders.kitti360_obs_dataloader import Kitti360Dataloader
from sem_pc_accum import SemanticPointCloudAccumulator


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

    parser = argparse.ArgumentParser()
    parser.add_argument('kitti360_path',
                        type=str,
                        help='Absolute path to dataset root (KITTI-360/).')
    parser.add_argument(
        'semseg_onnx_path',
        type=str,
        help='Relative path to a semantic segmentation ONNX model.')
    # Accumulator parameters
    parser.add_argument('--accumulation_horizon',
                        type=int,
                        default=200,
                        help='Number of point clouds to accumulate.')
    parser.add_argument('--accum_batch_size',
                        type=int,
                        default=2,
                        help='Set > 1 to avoid zero length computations')
    parser.add_argument('--accum_horizon_dist',
                        type=float,
                        default=300,
                        help='From front to back')
    parser.add_argument('--use_gt_sem', action="store_true")
    # BEV parameters
    parser.add_argument('--bev_output_dir', type=str, default='bevs')
    parser.add_argument('--bevs_per_sample', type=int, default=1)
    parser.add_argument('--bev_horizon_dist', type=int, default=120)
    parser.add_argument('--bev_dist_between_samples',
                        type=int,
                        default=1,
                        help='[m]')
    parser.add_argument('--bev_type',
                        type=str,
                        default='sem',
                        help='sem or rgb')
    parser.add_argument('--bev_view_size',
                        type=int,
                        default=80,
                        help='BEV representation size in [m]')
    parser.add_argument('--bev_pixel_size',
                        type=int,
                        default=512,
                        help='BEV representation size in [px]')
    parser.add_argument('--bev_max_trans_radius', type=float, default=0)
    parser.add_argument('--bev_zoom_thresh', type=float, default=0)
    parser.add_argument('--bev_do_warp', action="store_true")
    # ICP parameters
    parser.add_argument('--icp_threshold', type=float, default=1e3)

    args = parser.parse_args()

    # Path to dataset root directory
    kitti360_path = args.kitti360_path
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
    # 255: Ignore
    filters = [10, 11, 12, 16, 18, 255]
    sem_idxs = {'road': 0, 'car': 13, 'truck': 14, 'bus': 15, 'motorcycle': 17}

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
    #  BEV parameters
    ####################
    bevs_per_sample = args.bevs_per_sample
    bev_horizon_dist = args.bev_horizon_dist
    bev_dist_between_samples = args.bev_dist_between_samples

    bev_params = {
        'type': args.bev_type,  # Options: ['sem', 'rgb']
        'view_size': args.bev_view_size,
        'pixel_size': args.bev_pixel_size,
        'max_trans_radius': args.bev_max_trans_radius,  # 10,
        'zoom_thresh': args.bev_zoom_thresh,  # 0.10,
        'do_warp': args.bev_do_warp,
    }

    savedir = args.bev_output_dir
    subdir_size = 1000
    viz_to_disk = True  # For debugging purposes

    # Initialize accumulator
    sem_pc_accum = SemanticPointCloudAccumulator(
        args.accum_horizon_dist,
        calib_params,
        args.icp_threshold,
        args.semseg_onnx_path,
        filters,
        sem_idxs,
        args.use_gt_sem,
        bev_params,
    )

    #################
    #  Sample data
    #################
    batch_size = 1  # args.accum_batch_size
    sequences = [
        '2013_05_28_drive_0000_sync',
        '2013_05_28_drive_0002_sync',
        '2013_05_28_drive_0003_sync',
        '2013_05_28_drive_0004_sync',
        '2013_05_28_drive_0005_sync',
        '2013_05_28_drive_0006_sync',
        '2013_05_28_drive_0007_sync',
        '2013_05_28_drive_0009_sync',
        '2013_05_28_drive_0010_sync',
    ]
    start_idxs = [130, 4613, 40, 90, 50, 120, 0, 90, 0]
    end_idxs = [11400, 18997, 770, 11530, 6660, 9698, 2960, 13945, 3540]

    dataloader = Kitti360Dataloader(kitti360_path, batch_size, sequences,
                                    start_idxs, end_idxs)

    ###################
    #  Generate BEVs
    ###################
    bev_idx = 0
    subdir_idx = 0
    bev_count = 0

    previous_idx = 0
    for sample_idx, observations in enumerate(dataloader):

        # Number of observations removed from memory (used for pose diff.)
        num_obs_removed = sem_pc_accum.integrate(observations)

        # print(f'    num_obs_removed {num_obs_removed}')

        # Update last sampled 'abs pose' relative to 'ego pose' by
        # incrementally applying each pose change associated with each
        # observation
        #
        # NOTE Every step integrates #batch_size observations
        #      ==> Pose change correspond to #batch_size poses
        #
        # Observations    1 2 3
        #                 - - -
        #                | | | |
        # Indices        1 2 3 4
        #
        # The first iteration lacks first starting index
        # if len(sem_pc_accum.poses) > (batch_size + 1):
        #     last_idx = batch_size
        # else:
        #     last_idx = len(sem_pc_accum.poses) - 1
        # for idx in range(1, last_idx + 1):
        #     pose_f = np.array(sem_pc_accum.poses[-idx])
        #     pose_b = np.array(sem_pc_accum.poses[-idx - 1])
        #     delta_pose = pose_f - pose_b
        #     pose_0 -= delta_pose
        previous_idx -= num_obs_removed

        if len(sem_pc_accum.poses) < 2:
            continue

        incr_path_dists = sem_pc_accum.get_incremental_path_dists()

        # Condition (1): Sufficient distance to backward horizon
        if incr_path_dists[-1] < bev_horizon_dist:
            continue

        # Find 'present' idx position
        dists = (incr_path_dists - bev_horizon_dist)
        present_idx = (dists > 0).argmax()

        # Condition (2): Sufficient distance from present to future horizon
        fut_dist = incr_path_dists[-1] - incr_path_dists[present_idx]
        if fut_dist < bev_horizon_dist:
            continue

        # Condition (3): Sufficient distance from previous sample
        pose_0 = sem_pc_accum.get_pose(previous_idx)
        pose_1 = sem_pc_accum.get_pose(present_idx)
        dist_pose_1_2 = dist(pose_0, pose_1)

        if dist_pose_1_2 < bev_dist_between_samples:
            continue
        # pose_0 = pose_1
        previous_idx = present_idx

        print(
            f'{sample_idx*batch_size} | {bev_count} |',
            f' back {incr_path_dists[present_idx]:.1f} | front {fut_dist:.1f}')

        bevs = sem_pc_accum.generate_bev(present_idx,
                                         bevs_per_sample,
                                         gen_future=True)

        rgbs = sem_pc_accum.get_rgb(present_idx)
        semsegs = sem_pc_accum.get_semseg(present_idx)

        for bev in bevs:

            # Store BEV samples
            if bev_idx >= 1000:
                bev_idx = 0
                subdir_idx += 1
            filename = f'bev_{bev_idx}.pkl'
            output_path = f'./{savedir}/subdir{subdir_idx:03d}/'

            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            sem_pc_accum.write_compressed_pickle(bev, filename, output_path)

            # Visualize BEV samples
            if viz_to_disk:
                viz_file = os.path.join(output_path, f'viz_{bev_idx}.png')
                sem_pc_accum.viz_bev(bev, viz_file, rgbs, semsegs)

            bev_idx += 1
            bev_count += 1
