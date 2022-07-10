import gzip
import os
import pickle
from multiprocessing import Pool

import numpy as np
import open3d as o3d
import PIL.Image as Image

from utils.bev_generation import gen_aug_view
from utils.onnx_utils import SemSegONNX
from utils.transformations import gen_semantic_pc


class SemanticPointCloudAccumulator:
    '''

    Based on the Open3D point cloud library
    Ref: http://www.open3d.org/

    Usage pattern 1: Integrate observations into a semantic point cloud in a
                     common vector space

        sem_pc_accum = SemanticPointCloudAccumulator(...)
        sem_pc_accum.integrate( [(rgb, pc), ... ] )
        sem_pc_accum.viz_sem_vec_space()

    Usage pattern 2: Generate BEV representations

        sem_pc_accum = SemanticPointCloudAccumulator(...)
        sem_pc_accum.integrate( [(rgb, pc), ... ] )
        TODO

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

    def obs2sem_vec_space(self, rgb: Image, pc: np.array) -> tuple:
        '''
        Converts a new observation to a semantic point cloud in the common
        vector space.

        The function maintains the most recent pointcloud and transformation
        for the next observation update.

        Args:
            rgb: RGB image.
            pc: Semantic point cloud as row vector matrix w. dim (N, 8)
                [x, y, z, intensity, r, g, b, sem_idx]

        Returns:
            pc_velo_rgbsem (np.array): Semantic point cloud as row vector
                                       matrix w. dim (N, 8)
                                       [x, y, z, intensity, r, g, b, sem_idx]
            pose (list): List with (x, y, z) coordinates as floats.
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

    def get_pose(self, idx: int = None) -> np.array:
        '''
        Returns the pose matrix w. dim (N, 3) or pose vector if given an index.
        '''
        if idx is None:
            return np.array(self.poses)
        else:
            return np.array(self.poses[idx])

    def generate_bev(self,
                     present_idx: int,
                     bev_num: int,
                     bev_params: dict,
                     aug_params: dict = None,
                     gen_future: bool = False):
        '''
        Generates a single BEV representation.

        Args:

        Returns:
            bevs: List of dictionaries containg probabilistic semantic gridmaps
                  and trajectory information.

        '''
        # Build up input dictonary
        bev_gen_inputs = {}

        # 'Present' pose is origo
        bev_frame_coords = np.array(self.poses[present_idx])

        # TODO confirm axis
        pc_present = np.concatenate(self.sem_pcs[:present_idx])
        poses_present = np.concatenate([self.poses[:present_idx]])

        # Transform 'absolute' --> 'bev' coordinates
        pc_present[:, :3] = pc_present[:, :3] - bev_frame_coords
        poses_present = poses_present - bev_frame_coords

        bev_gen_inputs.update({
            'pc_present': pc_present,
            'poses_present': poses_present,
        })

        if gen_future:
            pc_future = np.concatenate(self.sem_pcs[present_idx:])
            poses_future = np.concatenate([self.poses[:present_idx:]])

            # Transform 'absolute' --> 'bev' coordinates
            pc_future[:, :3] = pc_future[:, :3] - bev_frame_coords
            poses_future = poses_future - bev_frame_coords

            bev_gen_inputs.update({
                'pc_future': pc_future,
                'poses_future': poses_future,
            })

        bev_gen_inputs.update(bev_params)
        bev_gen_inputs.update(aug_params)

        # Generate BEVs in parallel
        bev_gen_inputs = [bev_gen_inputs] * bev_num
        pool = Pool(processes=bev_num)
        bevs = pool.map(gen_aug_view, bev_gen_inputs)

        return bevs

    @staticmethod
    def write_compressed_pickle(obj, filename, write_dir):
        '''Converts an object into byte representation and writes a compressed file.
        Args:
            obj: Generic Python object.
            filename: Name of file without file ending.
            write_dir (str): Output path.
        '''
        path = os.path.join(write_dir, f"{filename}.gz")
        pkl_obj = pickle.dumps(obj)
        try:
            with gzip.open(path, "wb") as f:
                f.write(pkl_obj)
        except IOError as error:
            print(error)

    @staticmethod
    def read_compressed_pickle(path):
        '''Reads a compressed binary file and return the object.
        Args:
            path (str): Path to the file (incl. filename)
        '''
        try:
            with gzip.open(path, "rb") as f:
                pkl_obj = f.read()
                obj = pickle.loads(pkl_obj)
                return obj
        except IOError as error:
            print(error)

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
