import numpy as np
import open3d as o3d

from datasets.nuscenes_utils import homo_transform, pts_feat_from_img
from sem_pc_accum import SemanticPointCloudAccumulator


class NuScenesOracleSemanticPointCloudAccumulator(SemanticPointCloudAccumulator
                                                  ):

    def __init__(self,
                 semseg_onnx_path=None,
                 semseg_filters=None,
                 sem_idxs=None,
                 use_gt_sem=None,
                 bev_params=None):
        """
        Semantic point cloud accumulator compatible with NuScenes GT ego pose
        annotations (i.e. perfect acccumulation without ICP).

        Coordinate systems
            global: Map frame.
            world: Origo at first ego frame.
            ego: Origo at ego vehicle.

        Args:
            horizon_dist (float): maximum distance that ego vehicle traveled
                within an accumulated pointcloud. If ego vehicle travels more
                than this, past pointclouds will be discarded.
            icp_threshold (float): not used if using ground truth ego pose
            semseg_onnx_path (str): path to onnx file defining semseg model
            semseg_filters (list[int]): classes that are removed
            sem_idxs (dict): mapping semseg class to str
            bev_params (dict):
        """
        super().__init__(None, None, semseg_onnx_path, semseg_filters,
                         sem_idxs, use_gt_sem, bev_params)

        if use_gt_sem:
            raise NotImplementedError()

        # PC matrix column indices
        self.xyz_idx = 0
        self.dyn_idx = 8

        # 4x4 transformation matrix mapping pnts 'global' --> 'world' frame
        # Specified at first observation integration
        self.T_global_world = None

    def integrate(self, observations: list):
        """
        Integrates a sequence of K observations into the common vector space
        (i.e., a world frame which can be different to the global frame of
        NuScenes, for example 1st ego vehicle frame)

        Points in vector space are defined by [X, Y, Z] coordinates.

        Args:
            observations: List of K dict. One dict has the following keys
                images (list[PIL]):
                pc (np.ndarray): (N, 3+2[+1]) - X, Y, Z in EGO VEHICLE, pixel_u, pixel_v, [time-lag w.r.t keyframe]
                pc_cam_idx (np.ndarray): (N,) - index of camera where each point projected onto
                ego_at_lidar_ts (np.ndarray): (4, 4) ego vehicle pose w.r.t global frame @ timestamp of lidar
                cam_channels (list[str]):
        """

        obs = observations[0]
        rgbs = obs['images']
        pc = obs['pc']
        pc_cam_idx = obs['pc_cam_idx']
        T_ego_global = obs['ego_at_lidar_ts']

        if self.T_global_world is None:
            self.T_global_world = np.linalg.inv(T_ego_global)

        sem_pc, pose, semsegs = self.obs2sem_vec_space(rgbs, pc, pc_cam_idx,
                                                       T_ego_global)

        self.sem_pcs.append(sem_pc)
        self.poses.append(pose)
        self.rgbs.append(rgbs)
        self.semsegs.append(semsegs)

        # Compute path segment distance
        if len(self.poses) > 1:
            seg_dist = self.dist(np.array(self.poses[-1]),
                                 np.array(self.poses[-2]))
            self.seg_dists.append(seg_dist)

            path_length = np.sum(self.seg_dists)

            print(f'    #pc {len(self.sem_pcs)} |',
                  f'path length {path_length:.2f}')

    def obs2sem_vec_space(self, rgbs: list, pc: np.array, pc_cam_idx: np.array,
                          T_ego_global: np.array) -> tuple:
        """
        Converts a new observation to a semantic point cloud in the common
        vector space using oracle ego pose (i.e. ground truth).

        Args:
            rgbs: List of RGB images (PIL)
            pc: Point cloud as row vector matrix w. dim (N, 7)
                [x, y, z, int, pixel_u, pixel_v, time-lag w.r.t keyframe]
            pc_cam_idx: Index of camera where each point projected onto
            T_ego_global: 4x4 transformation matrix mapping pnts 'ego' -->
                          'global' frame

        Returns:
            pc_velo_rgbsem: Semantic point cloud as row vector matrix w. dim
                            (N, 9).
                            [x, y, z, intensity, r, g, b, sem_idx, dyn]
            pose: List with (x, y, z) coordinates as floats.
            semsegs: List with np.array semantic segmentation outputs.
        """
        #######################################
        #  Ego pose (x,y,z) in 'world' frame
        #######################################
        T_ego_world = self.T_global_world @ T_ego_global  # 4x4
        pose = T_ego_world[:3, -1].tolist()

        ###################################################
        #  Decorate pointcloud with semantic from images
        ###################################################
        # All points initialized to -1 as "invalid until masked"
        pc_rgb_sem = -np.ones((pc.shape[0], 4), dtype=float)  # r, g, b, semseg

        semsegs = []
        for cam_idx, rgb in enumerate(rgbs):
            semseg = self.semseg_model.pred(rgb)[0, 0]
            rgb = np.array(rgb)

            mask_in_rgb = (pc_cam_idx == cam_idx)
            pc_rgb_sem[mask_in_rgb] = pts_feat_from_img(
                pc[mask_in_rgb, 4:6],
                np.concatenate([rgb, np.expand_dims(semseg, -1)], axis=2),
                'nearest')

            semsegs.append(semseg)

        #######################################
        #  Filter pointcloud based on semseg
        #######################################
        mask_invalid_pts = np.any(pc_rgb_sem < 0,
                                  axis=1)  # pts that are not on any images
        for invalid_cls in self.semseg_filters:
            mask_invalid_pts = mask_invalid_pts | (pc_rgb_sem[:, -1]
                                                   == invalid_cls)

        mask_valid = np.logical_not(mask_invalid_pts)
        pc, pc_rgb_sem = pc[mask_valid], pc_rgb_sem[mask_valid]

        #########################################
        #  Transform pointcloud to WORLD frame
        #########################################
        pc_xyz = homo_transform(T_ego_world, pc[:, :3])

        # Normalized point cloud intensity
        pc_intensity = pc[:, 3:4] / 255.
        pc_dyn = -np.zeros((pc.shape[0], 1), dtype=float)  # dyn
        pc_velo_rgbsem = np.concatenate(
            [pc_xyz, pc_intensity, pc_rgb_sem, pc_dyn], axis=1)  # (N, 9)

        return pc_velo_rgbsem, pose, semsegs
