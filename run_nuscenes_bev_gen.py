import argparse
import os

import numpy as np
from nuscenes.nuscenes import NuScenes

from nuscenes_sem_pc_accum import NuScenesSemanticPointCloudAccumulator
from obs_dataloaders.nuscenes_obs_dataloader import NuScenesDataloader


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
    parser.add_argument('nuscenes_path',
                        type=str,
                        help='Absolute path to dataset root (nuscenes/).')
    parser.add_argument(
        'semseg_onnx_path',
        type=str,
        help='Relative path to a semantic segmentation ONNX model.')
    parser.add_argument('--nuscenes_version', type=str, default='v1.0-mini')
    # Accumulator parameters
    parser.add_argument('--accumulation_horizon',
                        type=int,
                        default=200,
                        help='Number of point clouds to accumulate.')
    parser.add_argument('--accum_batch_size', type=int, default=1)
    parser.add_argument('--num_sweeps',
                        type=int,
                        default=1,
                        help='Should be 1 to avoid noise not segmented out')
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
    # NuScenes invalid scene attributes
    parser.add_argument('--skip_attr',
                        type=str,
                        nargs='*',
                        default=[],
                        help='\'night\' etc.')

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

    ###################
    #  Generate BEVs
    ###################
    bev_idx = 0
    subdir_idx = 0
    bev_count = 0

    # For accessing scene attributes
    skip_attributes = args.skip_attr
    nusc = NuScenes(dataroot=args.nuscenes_path, version=args.nuscenes_version)

    for scene_id in range(850):

        print(f'Processing scene id {scene_id}')

        # Create a list of attribute strings to check validity of scene
        scene_invalid = False
        scene = nusc.scene[scene_id]
        scene['description'] = scene['description'].lower()
        scene_attributes = scene['description'].replace(', ', ',').split(',')

        # Skip scene if any attributes are invalid by specified attribute being
        # a substring in a scene description attribute
        for skip_attr in skip_attributes:
            for scene_attr in scene_attributes:
                if skip_attr in scene_attr:
                    scene_invalid = True
                    break

        if scene_invalid:
            print(f'Skip scene id {scene_id} ({skip_attr})')
            continue

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
        num_sweeps = 1
        scene_ids = [scene_id]
        dataloader = NuScenesDataloader(
            args.nuscenes_path,
            scene_ids,
            args.accum_batch_size,
            num_sweeps,
            args.nuscenes_version,
        )

        # Integrate entire sequence
        for sample_idx, observations in enumerate(dataloader):
            sem_pc_accum.integrate(observations)

        num_poses = len(sem_pc_accum.poses)
        incr_path_dists = sem_pc_accum.get_incremental_path_dists()

        for present_idx in range(1, num_poses):

            bevs = sem_pc_accum.generate_bev(present_idx,
                                             bevs_per_sample,
                                             gen_future=True)

            rgbs = sem_pc_accum.get_rgb(present_idx)[0]
            semsegs = sem_pc_accum.get_semseg(present_idx)[0]

            for bev in bevs:

                # Store BEV samples
                if bev_idx > 1000:
                    bev_idx = 0
                    subdir_idx += 1
                filename = f'bev_{bev_idx}.pkl'
                output_path = f'./{savedir}/subdir{subdir_idx:03d}/'

                if not os.path.isdir(output_path):
                    os.makedirs(output_path)

                sem_pc_accum.write_compressed_pickle(bev, filename,
                                                     output_path)

                # Visualize BEV samples
                if viz_to_disk:
                    viz_file = os.path.join(output_path, f'viz_{bev_idx}.png')
                    sem_pc_accum.viz_bev(bev, viz_file, rgbs, semsegs)

                bev_idx += 1
                bev_count += 1
