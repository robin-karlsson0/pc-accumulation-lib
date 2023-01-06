import numpy as np
import open3d as o3d

from datasets.nuscenes_utils import pts_feat_from_img
from sem_pc_accum import SemanticPointCloudAccumulator


class NuScenesSemanticPointCloudAccumulator(SemanticPointCloudAccumulator):

    def __init__(self,
                 horizon_dist,
                 icp_threshold,
                 semseg_onnx_path=None,
                 semseg_filters=None,
                 sem_idxs=None,
                 use_gt_sem=None,
                 bev_params=None):
        """
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
        super().__init__(horizon_dist, icp_threshold, semseg_onnx_path,
                         semseg_filters, sem_idxs, use_gt_sem, bev_params)

        if use_gt_sem:
            raise NotImplementedError()

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
        obs = observations[0]
        rgbs = obs['images']
        pc = obs['pc']
        pc_cam_idx = obs['pc_cam_idx']

        sem_pc, pose, semsegs, T_new_prev = self.obs2sem_vec_space(
            rgbs, pc, pc_cam_idx)

        if len(self.poses) > 0:

            # Transform previous poses to new ego coordinate system
            new_poses = []
            for pose_ in self.poses:
                # Homogeneous spatial coordinates
                new_pose = np.matmul(T_new_prev, np.array([pose_ + [1]]).T)
                new_pose = new_pose[:, 0][:-1]  # (4,1) --> (3)
                new_pose = list(new_pose)
                new_poses.append(new_pose)
            self.poses = new_poses

            # Transform previous observations to new ego coordinate system
            new_sem_pcs = []
            for sem_pc_ in self.sem_pcs:
                # Skip transforming empty point clouds
                if sem_pc_.shape[0] == 0:
                    new_sem_pcs.append(sem_pc_)
                    continue
                # Homogeneous spatial coordinates
                N = sem_pc_.shape[0]
                sem_pc_homo = np.concatenate((sem_pc_[:, :3], np.ones((N, 1))),
                                             axis=1)
                sem_pc_homo = np.matmul(T_new_prev, sem_pc_homo.T).T
                # Replace spatial coordinates
                sem_pc_[:, :3] = sem_pc_homo[:, :3]
                new_sem_pcs.append(sem_pc_)
            self.sem_pcs = new_sem_pcs

        # TODO Skip integrating when self-localization fails (discontinous path)

        self.sem_pcs.append(sem_pc)
        self.poses.append(pose)
        self.rgbs.append(rgbs)
        self.semsegs.append(semsegs)

        # Compute path segment distance
        idx = 0  # Default value for no removed observations
        if len(self.poses) > 1:
            seg_dist = self.dist(np.array(self.poses[-1]),
                                 np.array(self.poses[-2]))
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
                self.rgbs = self.rgbs[idx:]
                self.semsegs = self.semsegs[idx:]

            print(f'    #pc {len(self.sem_pcs)} |',
                  f'path length {path_length:.2f}')

        # Number of observations removed
        return idx

    def obs2sem_vec_space(self,
                          rgbs: list,
                          pc: np.array,
                          pc_cam_idx: np.array,
                          pose_z_origin: float = 1.) -> tuple:
        """
        Converts a new observation to a semantic point cloud in the common
        vector space.
        The function maintains the most recent pointcloud and transformation
        for the next observation update.

        Args:
            rgbs: List of RGB images (PIL)
            pc: Point cloud as row vector matrix w. dim (N, 6)
                [x, y, z, pixel_u, pixel_v, time-lag w.r.t keyframe]
            pc_cam_idx: Index of camera where each point projected onto
            pose_z_orgin (int): Move origin above ground level.

        Returns:
            pc_velo_rgbsem (np.array): Semantic point cloud as row vector
                                       matrix w. dim (N, 8)
                                       [x, y, z, intensity, r, g, b, sem_idx]
            pose (list): List with (x, y, z) coordinates as floats.
        """
        # Convert point cloud to Open3D format using (x,y,z) data
        pcd_new = self.pc2pcd(pc[:, :3])
        if self.pcd_prev is None:
            self.pcd_prev = pcd_new

        # Compute pose transformation T 'origin' --> 'current pc'
        # Transform 'ego' --> 'abs' ref. frame
        target = self.pcd_prev
        source = pcd_new
        reg_p2l = o3d.pipelines.registration.registration_icp(
            target, source, self.icp_threshold, self.icp_trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T_new_prev = reg_p2l.transformation
        T_new_origin = np.matmul(self.T_prev_origin, T_new_prev)

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
                pc[mask_in_rgb, 3:5],
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

        pc_xyz = pc[:, :3]
        pc_intensity = np.zeros((pc.shape[0], 1), dtype=float)
        pc_velo_rgbsem = np.concatenate([pc_xyz, pc_intensity, pc_rgb_sem],
                                        axis=1)  # (N, 8)

        # Pose of new observations always ego-centered
        pose = [0., 0., 0.]

        # Move stored pose origin up from the ground
        pose[2] += pose_z_origin

        self.T_prev_origin = T_new_origin
        self.pcd_prev = pcd_new

        return pc_velo_rgbsem, pose, semsegs, T_new_prev

    def get_rgb(self, idx: int = None) -> list:
        '''
        Returns one or all rgb images (PIL.Image) depending on 'idx'.
        '''
        if idx is None:
            return self.rgbs
        else:
            return self.rgbs[idx]

    def get_semseg(self, idx: int = None) -> list:
        '''
        Returns one or all semseg outputs (np.array) depending on 'idx'.
        '''
        if idx is None:
            return self.semsegs
        else:
            return self.semsegs[idx]
