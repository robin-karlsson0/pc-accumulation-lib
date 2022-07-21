from obs_dataloaders.nuscenes_obs_dataloader import NuScenesDataloader
from nuscenes_sem_pc_accum import NuScenesSemanticPointCloudAccumulator
from datasets.nuscenes_utils import pts_feat_from_img
import numpy as np
from sandbox.misc import show_pointcloud
import os


if __name__ == '__main__':
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
    savedir = 'output/bev_online'
    subdir_size = 1000
    viz_to_disk = True  # For debugging purposes

    sem_pc_accum = NuScenesSemanticPointCloudAccumulator(
        horizon_dist=accum_horizon_dist,
        semseg_filters=filters,
        sem_idxs=sem_idxs,
        bev_params=bev_params,
    )

    # ###################
    # Dataloader param
    # ###################
    nusc_root = '/home/user/dataset/nuscenes/'
    nusc_version = 'v1.0-mini'
    batch_size = 1
    num_sweeps = 5
    scene_ids = [0]

    dataloader = NuScenesDataloader(nusc_root, scene_ids, batch_size, num_sweeps, nusc_version)
    counter = 0

    for obss in dataloader:
        sem_pc_accum.integrate(obss)  # heavy-lifting happens here
        bev = sem_pc_accum.generate_bev()
        bev = bev[0]

        # save bev img
        output_dir = os.path.join(savedir, 'scene', obss[-1]['meta']['scene_token'])
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        viz_file = os.path.join(output_dir, f"cnt_{counter}_bev_online_batch_size1_{obss[-1]['meta']['sample_token']}.png")
        sem_pc_accum.viz_bev(bev, viz_file)

        counter += 1
        if counter > 10:
            sem_pc_accum.viz_sem_vec_space()
            break
