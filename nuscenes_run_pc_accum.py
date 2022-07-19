from obs_dataloaders.nuscenes_obs_dataloader import NuScenesDataloader
from nuscenes_sem_pc_accum import NuScenesSemanticPointCloudAccumulator
from datasets.nuscenes_utils import pts_feat_from_img
import numpy as np
from sandbox.misc import show_pointcloud


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
    calib_params = {}
    calib_params['h_velo_cam'] = None
    calib_params['p_cam_frame'] = None
    calib_params['p_velo_frame'] = None
    calib_params['c_x'] = None
    calib_params['c_y'] = None
    calib_params['f_x'] = None
    calib_params['f_y'] = None
    icp_threshold = 1e3
    bev_params = {'type': None}

    sem_pc_accum = NuScenesSemanticPointCloudAccumulator(
        accum_horizon_dist,
        calib_params,
        icp_threshold,
        semseg_onnx_path,
        filters,
        sem_idxs,
        bev_params,
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
        counter += 1
        if counter > 10:
            sem_pc_accum.viz_sem_vec_space()
            break
