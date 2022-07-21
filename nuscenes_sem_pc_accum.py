import numpy as np
import os.path as osp
import torch
from mmseg.apis import inference_segmentor, init_segmentor
from sem_pc_accum import SemanticPointCloudAccumulator
from datasets.nuscenes_utils import pts_feat_from_img, homo_transform
from utils.onnx_utils import SemSegONNX
from bev_generator.sem_bev import SemBEVGenerator
from bev_generator.rgb_bev import RGBBEVGenerator


class NuScenesSemanticPointCloudAccumulator(SemanticPointCloudAccumulator):
    def __init__(self,
                 horizon_dist,
                 calib_params=None,
                 icp_threshold=None,
                 semseg_onnx_path=None,
                 semseg_filters=None,
                 sem_idxs=None,
                 bev_params=None):
        """
        Args:
            horizon_dist (float): maximum distance that ego vehicle traveled within an accumulated pointcloud. If
                ego vehicle travels more than this, past pointclouds will be discarded
            calib_params (dict): not used in NuScenes
            icp_threshold (float): not used if using ground truth ego pose
            semseg_onnx_path (str): path to onnx file defining semseg model
            semseg_filters (list[int]): classes that are removed
            sem_idxs (dict): mapping semseg class to str
            bev_params (dict):
        """
        # Semseg model
        if semseg_onnx_path is not None:
            self.semseg_model = SemSegONNX(semseg_onnx_path)
            self.semseg_by_onnx = True
        else:
            mmseg_root = '/home/user/Desktop/python_ws/mmsegmentation'
            config_file = osp.join(mmseg_root, 'configs', 'deeplabv3', 'deeplabv3_r18-d8_512x1024_80k_cityscapes.py')
            ckpt_file = osp.join(mmseg_root, 'checkpoints',
                                 'deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth')
            self.semseg_model = init_segmentor(config_file, ckpt_file, device='cuda:0')
            self.semseg_by_onnx = False
        self.semseg_filters = semseg_filters
        self.sem_idxs = sem_idxs

        # Calibration param
        if calib_params is not None:
            raise ValueError("calib_params is not supported for NuScenes")
        self.icp_threshold = icp_threshold

        # Init pose and transformation matrix
        if icp_threshold is not None:
            self.T_prev_origin = np.eye(4)
            self.icp_trans_init = np.eye(4)
            self.pcd_prev = None  # pointcloud in the last observation
        else:
            self.world_from_glob = None  # 4x4 tf that maps pts in GLOBAL frame to WORLD frame == the 1st EGO frame

        self.horizon_dist = horizon_dist

        self.sem_pcs = []  # (N)
        self.poses = []  # (N)
        self.seg_dists = []  # (N-1)
        # BEV generator
        # Default initialization is 'None'
        self.sem_bev_generator = None
        if bev_params['type'] == 'sem':
            self.sem_bev_generator = SemBEVGenerator(
                self.sem_idxs,
                bev_params['view_size'],
                bev_params['pixel_size'],
                bev_params['max_trans_radius'],
                bev_params['zoom_thresh'],
            )
        elif bev_params['type'] == 'rgb':
            self.sem_bev_generator = RGBBEVGenerator(
                bev_params['view_size'],
                bev_params['pixel_size'],
                0,
                bev_params['max_trans_radius'],
                bev_params['zoom_thresh'],
            )

    def integrate(self, observations: list):
        """
        Integrates a sequence of K observations into the common vector space (i.e., a world frame which can be
        different to the global frame of NuScenes, for example 1st ego vehicle frame)

        Points in vector space are defined by [X, Y, Z] coordinates.

        Args:
            observations: List of K dict. One dict has the following keys
                images (list[PIL]):
                pc (np.ndarray): (N, 3+2[+1]) - X, Y, Z in EGO VEHICLE, pixel_u, pixel_v, [time-lag w.r.t keyframe]
                pc_cam_idx (np.ndarray): (N,) - index of camera where each point projected onto
                ego_at_lidar_ts (np.ndarray): (4, 4) ego vehicle pose w.r.t global frame @ timestamp of lidar
                cam_channels (list[str]):
        """
        for obs_idx in range(len(observations)):
            obs = observations[obs_idx]
            if self.world_from_glob is None:
                self.world_from_glob = np.linalg.inv(obs['ego_at_lidar_ts'])

            sem_pc, pose = self.nusc_obs2sem_vec_space(obs)
            self.sem_pcs.append(sem_pc)
            self.poses.append(pose)

            # Compute path segment distance
            if len(self.poses) > 1:
                seg_dist = self.dist(np.array(self.poses[-1]), np.array(self.poses[-2]))
                self.seg_dists.append(seg_dist)

                path_length = np.sum(self.seg_dists)

                if path_length > self.horizon_dist:
                    # Incremental path distance starting from zero
                    incr_path_dists = self.get_incremental_path_dists()
                    # Elements beyond horizon distance become negative
                    overshoot = path_length - self.horizon_dist
                    incr_path_dists -= overshoot
                    # Find first non-negative element index ==> Within horizon
                    idx = (incr_path_dists > 0.).argmax()
                    # Remove elements before 'idx' as they are outside horizon
                    self.sem_pcs = self.sem_pcs[idx:]
                    self.poses = self.poses[idx:]
                    self.seg_dists = self.seg_dists[idx:]

                print(f'    #pc {len(self.sem_pcs)} |',
                      f'path length {path_length:.2f}')

    def nusc_obs2sem_vec_space(self, obs) -> tuple:
        """
        Converts a new observation to a semantic point cloud in the common
        vector space.

        The function maintains the most recent pointcloud and transformation
        for the next observation update.

        Args:
            obs (dict):
                images (list[PIL]):
                pc (np.ndarray): (N, 3+2[+1]) - X, Y, Z in EGO VEHICLE, pixel_u, pixel_v, [time-lag w.r.t keyframe]
                pc_cam_idx (np.ndarray): (N,) - index of camera where each point projected onto
                ego_at_lidar_ts (np.ndarray): (4, 4) ego vehicle pose w.r.t global frame @ timestamp of lidar
                cam_channels (list[str]):

        Returns:
            pc_velo_rgbsem (np.array): Semantic point cloud as row vector
                                       matrix w. dim (N, 8)
                                       [x, y, z, intensity, r, g, b, sem_idx]
            pose (list): List with (x, y, z) coordinates as floats.
        """
        # #################################
        # Get EGO pose w.r.t WORLD frame
        # #################################
        glob_from_ego = obs['ego_at_lidar_ts']
        world_from_ego = self.world_from_glob @ glob_from_ego  # 4x4
        pose = world_from_ego[:3, -1].tolist()

        # ##############################################
        # Decorate pointcloud with semantic from images
        # ##############################################
        pc = obs['pc']  # (N, 3+2[+1]) X, Y, Z in EGO VEHICLE, pixel_u, pixel_v, [time-lag w.r.t keyframe]
        pc_cam_idx = obs['pc_cam_idx']
        pc_rgb_sem = -np.ones((pc.shape[0], 4), dtype=float)  # r, g, b, semseg
        for cam_idx, img in enumerate(obs['images']):
            if self.semseg_by_onnx:
                semseg = self.semseg_model.pred(img)[0, 0]
                img = np.array(img)
            else:
                img = np.array(img)
                semseg = inference_segmentor(self.semseg_model, img)[0]
            mask_in_img = pc_cam_idx == cam_idx
            pc_rgb_sem[mask_in_img] = pts_feat_from_img(
                pc[mask_in_img, 3: 5],
                np.concatenate([img, np.expand_dims(semseg, -1)], axis=2), 'nearest'
            )

        # ##################################
        # Filter pointcloud based on semseg
        # ##################################
        mask_invalid_pts = np.any(pc_rgb_sem < 0, axis=1)  # pts that are not on any images
        for invalid_cls in self.semseg_filters:
            mask_invalid_pts = mask_invalid_pts | (pc_rgb_sem[:, -1] == invalid_cls)

        mask_valid = np.logical_not(mask_invalid_pts)
        pc, pc_rgb_sem = pc[mask_valid], pc_rgb_sem[mask_valid]

        # #####################################
        # Transform pointcloud to WORLD frame
        # #####################################
        pc_xyz = homo_transform(world_from_ego, pc[:, :3])

        pc_xyz_rgb_sem = np.concatenate([pc_xyz, np.zeros((pc_xyz.shape[0], 1), dtype=float), pc_rgb_sem], axis=1)  # (N, 8)
        return pc_xyz_rgb_sem, pose




