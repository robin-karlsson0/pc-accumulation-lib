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

        self.xyz_idx = 0
        self.dyn_idx = 8

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

        # Transform previous poses and sem_pcs to new ego coordinate system
        if len(self.poses) > 0:
            self.update_poses(T_new_prev)
            self.update_sem_pcs(T_new_prev)

        self.sem_pcs.append(sem_pc)
        self.poses.append(pose)
        self.rgbs.append(rgbs)
        self.semsegs.append(semsegs)

        # Remove observations beyond "memory horizon"
        idx = 0  # Default value for no removed observations
        if len(self.poses) > 1:
            idx, path_length = self.remove_observations()

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
            pc: Point cloud as row vector matrix w. dim (N, 7)
                [x, y, z, int, pixel_u, pixel_v, time-lag w.r.t keyframe]
            pc_cam_idx: Index of camera where each point projected onto
            pose_z_orgin (int): Move origin above ground level.

        Returns:
            pc_velo_rgbsem: Semantic point cloud as row vector matrix w. dim
                            (N, 9).
                            [x, y, z, intensity, r, g, b, sem_idx, dyn]
            pose: List with (x, y, z) coordinates as floats.
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

        pc_xyz = pc[:, :3]
        # Normalized point cloud intensity
        pc_intensity = pc[:, 3:4] / 255.
        pc_dyn = -np.zeros((pc.shape[0], 1), dtype=float)  # dyn
        pc_velo_rgbsem = np.concatenate(
            [pc_xyz, pc_intensity, pc_rgb_sem, pc_dyn], axis=1)  # (N, 9)

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
