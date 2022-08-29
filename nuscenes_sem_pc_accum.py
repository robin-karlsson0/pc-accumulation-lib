import os.path as osp

import numpy as np
import open3d as o3d
import torch

from bev_generator.rgb_bev import RGBBEVGenerator
from bev_generator.sem_bev import SemBEVGenerator
from datasets.nuscenes_utils import homo_transform, pts_feat_from_img
from sem_pc_accum import SemanticPointCloudAccumulator
from utils.onnx_utils import SemSegONNX


class NuScenesSemanticPointCloudAccumulator(SemanticPointCloudAccumulator):

    def __init__(self,
                 horizon_dist,
                 icp_threshold,
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
        self.semseg_model = SemSegONNX(semseg_onnx_path)
        self.semseg_by_onnx = True

        self.semseg_filters = semseg_filters
        self.sem_idxs = sem_idxs

        self.icp_threshold = icp_threshold

        # Init pose and transformation matrix
        self.T_prev_origin = np.eye(4)
        self.icp_trans_init = np.eye(4)
        self.pcd_prev = None  # pointcloud in the last observation

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

            rgbs = obs['images']
            pc = obs['pc']
            pc_cam_idx = obs['pc_cam_idx']

            sem_pc, pose = self.nusc_obs2sem_vec_space(rgbs, pc, pc_cam_idx)
            self.sem_pcs.append(sem_pc)
            self.poses.append(pose)

            # Compute path segment distance
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

                print(f'    #pc {len(self.sem_pcs)} |',
                      f'path length {path_length:.2f}')

    def nusc_obs2sem_vec_space(self,
                               rgbs,
                               pc,
                               pc_cam_idx,
                               pose_z_origin=1.) -> tuple:
        """
        Converts a new observation to a semantic point cloud in the common
        vector space.
        The function maintains the most recent pointcloud and transformation
        for the next observation update.

        Args:
            rgbs (list[PIL]):
            pc (np.ndarray): (N, 3+2[+1]) - X, Y, Z in EGO VEHICLE, pixel_u, pixel_v, [time-lag w.r.t keyframe]
            pc_cam_idx (np.ndarray): (N,) - index of camera where each point projected onto
            pose_z_orgin (int): Move origin above ground level.

        Returns:
            pc_xyz_rgb_sem (np.array): Semantic point cloud as row vector
                                       matrix w. dim (N, 8)
                                       [x, y, z, intensity, r, g, b, sem_idx]
            pose (list): List with (x, y, z) coordinates as floats.
        """
        # Convert point cloud to Open3D format using (x,y,z,i) data
        pcd_new = self.pc2pcd(pc[:, :4])
        if self.pcd_prev is None:
            self.pcd_prev = pcd_new

        # Compute pose transformation T 'origin' --> 'current pc'
        # Transform 'ego' --> 'abs' ref. frame
        target = self.pcd_prev
        source = pcd_new
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, self.icp_threshold, self.icp_trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T_new_prev = reg_p2l.transformation
        T_new_origin = np.matmul(self.T_prev_origin, T_new_prev)

        ###################################################
        #  Decorate pointcloud with semantic from images
        ###################################################
        # All points initialized to -1 as "invalid until masked"
        pc_rgb_sem = -np.ones((pc.shape[0], 4), dtype=float)  # r, g, b, semseg

        for cam_idx, rgb in enumerate(rgbs):
            semseg = self.semseg_model.pred(rgb)[0, 0]
            rgb = np.array(rgb)

            mask_in_rgb = (pc_cam_idx == cam_idx)
            pc_rgb_sem[mask_in_rgb] = pts_feat_from_img(
                pc[mask_in_rgb, 3:5],
                np.concatenate([rgb, np.expand_dims(semseg, -1)], axis=2),
                'nearest')

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
        pc_xyz_rgb_sem = np.concatenate([pc_xyz, pc_intensity, pc_rgb_sem],
                                        axis=1)  # (N, 8)

        # Transform point cloud 'ego --> abs' homogeneous coordinates
        N = pc_xyz_rgb_sem.shape[0]
        pc_xyz_homo = np.concatenate((pc_xyz_rgb_sem[:, :3], np.ones((N, 1))),
                                     axis=1)
        pc_xyz_homo = np.matmul(T_new_origin, pc_xyz_homo.T).T
        # Replace spatial coordinates
        pc_xyz_rgb_sem[:, :3] = pc_xyz_homo[:, :3]

        # Compute pose in 'absolute' coordinates
        # Pose = Project origin in ego ref. frame --> abs
        pose = np.array([[0., 0., 0., 1.]]).T
        pose = np.matmul(T_new_origin, pose)
        pose = pose.T[0][:-1]  # Remove homogeneous coordinate
        pose = pose.tolist()

        # Move stored pose origin up from the ground
        pose[2] += pose_z_origin

        self.T_prev_origin = T_new_origin
        self.pcd_prev = pcd_new

        return pc_xyz_rgb_sem, pose
