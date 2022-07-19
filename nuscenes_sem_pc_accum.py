import numpy as np

from sem_pc_accum import SemanticPointCloudAccumulator
from datasets.nuscenes_utils import pts_feat_from_img, homo_transform


class NuScenesSemanticPointCloudAccumulator(SemanticPointCloudAccumulator):
    def __init__(self, horizon_dist: float, calib_params: dict,
                 icp_threshold: float, semseg_onnx_path: str,
                 semseg_filters: list, sem_idxs: dict, bev_params: dict):
        super().__init__(horizon_dist, calib_params, icp_threshold, semseg_onnx_path, semseg_filters, sem_idxs,
                         bev_params)
        self.world_from_glob = None  # 4x4 tf that maps pts in GLOBAL frame to WORLD frame == the 1st EGO frame

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
        pc_rgb_sem = -np.ones((pc.shape[0], 3), dtype=float)  # r, g, b, semseg
        for cam_idx, img in enumerate(obs['images']):
            # semseg = self.semseg_model.pred(img)[0, 0]
            mask_in_img = pc_cam_idx == cam_idx
            pc_rgb_sem[mask_in_img] = pts_feat_from_img(
                pc[mask_in_img, 3: 5], np.array(img), 'nearest'
            )

        # ##################################
        # Filter pointcloud based on semseg
        # ##################################
        mask_invalid_pts = np.any(pc_rgb_sem < 0, axis=1)  # pts that are not on any images
        # for invalid_cls in self.semseg_filters:
        #     mask_invalid_pts = mask_invalid_pts | (pc_rgb_sem[:, -1] == invalid_cls)
        mask_valid = np.logical_not(mask_invalid_pts)
        pc, pc_rgb_sem = pc[mask_valid], pc_rgb_sem[mask_valid]

        # #####################################
        # Transform pointcloud to WORLD frame
        # #####################################
        pc_xyz = homo_transform(world_from_ego, pc[:, :3])

        pc_xyz_rgb_sem = np.concatenate([pc_xyz, np.zeros((pc_xyz.shape[0], 1), dtype=float), pc_rgb_sem], axis=1)  # (N, 8)
        return pc_xyz_rgb_sem, pose




