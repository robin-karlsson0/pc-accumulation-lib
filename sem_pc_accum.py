import os

import numpy as np
import open3d as o3d
import PIL.Image as Image

from datasets.kitti360_utils import (get_camera_intrinsics,
                                     get_transf_matrices, idx2str,
                                     read_pc_bin_file)
from utils.onnx_utils import SemSegONNX
from utils.transformations import gen_semantic_pc


class SemanticPointCloudAccumulator:
    '''

    Based on the Open3D point cloud library
    Ref: http://www.open3d.org/

    How to use:

    '''

    def __init__(self, horizon_dist: float, calib_params: dict,
                 icp_threshold: float, semseg_onnx_path: str,
                 semseg_filters: list):
        '''
        Args:
            calib_params: h_velo_cam: np.array,
                          p_cam_frame: np.array,
                          p_velo_frame: np.array,
                          c_x, c_y, f_x, f_y: int
                              calib_params['c_x'] --> c_x
                              calib_params['c_y'] --> c_y
                              calib_params['f_x'] --> f_x
                              calib_params['f_y'] --> f_y
            semseg_filters: List of semantic class idxs to filter out from
                            point cloud.
        '''
        # Semantic segmentation model
        self.semseg_model = SemSegONNX(semseg_onnx_path)
        self.semseg_filters = semseg_filters

        # Calibration parameters
        self.H_velo_cam = calib_params['h_velo_cam']
        self.P_cam_frame = calib_params['p_cam_frame']
        self.P_velo_frame = calib_params['p_velo_frame']

        self.icp_threshold = icp_threshold

        self.icp_trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0],
                                          [0, 0, 1, 0], [0, 0, 0, 1]])

        # Initial pose and transformation matrix
        self.T_prev_origin = np.eye(4)

        # Point cloud of last observations
        self.pcd_prev = None

        self.horizon_dist = horizon_dist

        self.sem_pcs = []
        self.poses = []

    def integrate(self, observations: list):
        '''
        Integrates a sequence of K observations into the common vector space.

        Points in vector space are defined by absolute coordinates.

        rgb (Image): RGB images.

        pc (np.array): Point cloud as row-vector matrix w. dim (N, 4) having
                       values x, y, z, intensity.

        sem_pc (np.array): Semantic point cloud as row vector matrix w. dim
                           (N, 8) [x, y, z, intensity, r, g, b, sem_idx]

        Args:
            observations: List of K tuples (rgb, pc)
        '''
        for obs_idx in range(len(observations)):
            rgb, pc = observations[obs_idx]
            sem_pc, pose = self.obs2sem_vec_space(rgb, pc)
            self.sem_pcs.append(sem_pc)
            self.poses.append(pose)

            # Remove obsolete point clouds above distance horizon threshold
            pose_xy_current = pose[:2]
            while self.dist(np.array(self.poses[0][:2]),
                            np.array(pose_xy_current)) > self.horizon_dist:
                self.sem_pcs.pop(0)
                self.poses.pop(0)

    def obs2sem_vec_space(self, rgb, pc):
        '''
        '''
        # Convert point cloud to Open3D format
        pcd_new = self.pc2pcd(pc)
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

        # Semantic point cloud
        semseg = self.semseg_model.pred(rgb)[0, 0]
        pc_velo_rgb = gen_semantic_pc(pc, np.array(rgb), self.P_velo_frame)
        pc_velo_sem = gen_semantic_pc(pc, np.expand_dims(semseg, -1),
                                      self.P_velo_frame)

        pc_velo_rgbsem = np.concatenate((pc_velo_rgb, pc_velo_sem[:, -1:]),
                                        axis=1)

        # Transform point cloud 'ego --> abs' homogeneous coordinates
        N = pc_velo_rgbsem.shape[0]
        pc_velo_homo = np.concatenate((pc_velo_rgbsem[:, :3], np.ones((N, 1))),
                                      axis=1)
        pc_velo_homo = np.matmul(T_new_origin, pc_velo_homo.T).T
        # Replace spatial coordinates
        pc_velo_rgbsem[:, :3] = pc_velo_homo[:, :3]

        # Filter out unwanted points according to semantics
        pc_velo_rgbsem = self.filter_semseg_pc(pc_velo_rgbsem, )

        # Compute pose in 'absolute' coordinates
        # Pose = Project origin in ego ref. frame --> abs
        pose = np.array([[0., 0., 0., 1.]]).T
        pose = np.matmul(T_new_origin, pose)
        pose = pose.T[0][:-1]  # Remove homogeneous coordinate
        pose = pose.tolist()

        self.T_prev_origin = T_new_origin
        self.pcd_prev = pcd_new

        return pc_velo_rgbsem, pose

    def generate(self, type: str):
        pass

    @staticmethod
    def pc2pcd(pc):
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(pc[:, :3])
        pcd_new.estimate_normals()
        return pcd_new

    def filter_semseg_pc(self, pc):
        for filter in self.semseg_filters:
            mask = pc[:, -1] != filter
            pc = pc[mask]
        return pc

    @staticmethod
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

    def viz_sem_vec_space(self):
        '''
        Visualize stored semantic vector space.
        '''
        sem_vec_space = np.concatenate(self.sem_pcs, axis=0)
        self.viz_sem_pc(sem_vec_space, self.poses)

    @staticmethod
    def viz_sem_pc(sem_pc: np.array, poses: list = []):
        '''
        Args:
            sem_pc: Semantic point cloud as row vector matrix w. dim (N, 8)
                    [x, y, z, intensity, r, g, b, sem_idx]
            poses: List of lists with (x, y, z) coordinates.
        '''
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sem_pc[:, :3])
        rgb = sem_pc[:, 4:7]
        rgb /= 255
        # rgb = np.tile(sem_pc[:, 4:5], (1, 3))
        # rgb /= np.max(rgb)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        if len(poses) == 0:
            origin = [0, 0, 0]
        else:
            origin = poses[0]
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=origin)
        # Ego path
        lines = [[idx, idx + 1] for idx in range(len(poses) - 1)]
        colors = [[1, 0, 0] for _ in range(len(poses) - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(poses),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([mesh_frame, line_set, pcd])


if __name__ == '__main__':

    # Path to dataset root directory
    kitti360_path = '/home/robin/datasets/KITTI-360'
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

    horizon_dist = 80

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
    #  ICP parameters
    ####################
    icp_threshold = 1e3

    # Initialize accumulator

    sem_pc_accum = SemanticPointCloudAccumulator(horizon_dist, calib_params,
                                                 icp_threshold,
                                                 semseg_onnx_path, filters)

    #################
    #  Sample data
    #################

    pc_dir = os.path.join(kitti360_path, 'data_3d_raw',
                          '2013_05_28_drive_0000_sync', 'velodyne_points',
                          'data')
    img_dir = os.path.join(kitti360_path, 'data_2d_raw',
                           '2013_05_28_drive_0000_sync', 'image_00',
                           'data_rect')
    observations = []
    for frame_idx in range(200, 250):
        idx_str = idx2str(frame_idx)

        pc_path = os.path.join(pc_dir, idx_str + '.bin')
        pc = read_pc_bin_file(pc_path)

        rgb_path = os.path.join(img_dir, idx_str + '.png')
        rgb = Image.open(rgb_path)

        observation = (rgb, pc)
        observations.append(observation)

    ############################
    #  Integrate observations
    ############################

    sem_pc_accum.integrate(observations)

    sem_pc_accum.viz_sem_vec_space()

    print('Knut')
