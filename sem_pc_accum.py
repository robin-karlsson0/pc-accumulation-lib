import gzip
import os
import pickle
from multiprocessing import Pool

import numpy as np
import open3d as o3d
import PIL.Image as Image

from bev_generator.rgb_bev import RGBBEVGenerator
from bev_generator.sem_bev import SemBEVGenerator
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
                 semseg_filters: list, sem_idxs: dict, bev_params: dict):
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
        self.sem_idxs = sem_idxs

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

    @staticmethod
    def comp_incr_path_dist(seg_dists: list):
        '''
        Computes a sequence of incremental path distances from a sequence of
        path segment distances using matrix multiplication.

        Args:
            seg_dists: List of path segment distances [d1, d2, d3, ... ]

        Returns:

        '''
        lower_tri_mat = np.tri(len(seg_dists))
        seg_dists = np.array(seg_dists)

        incr_path_dist = np.matmul(lower_tri_mat, seg_dists)

        return incr_path_dist

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

    def get_segment_dists(self) -> list:
        '''
        Returns list of path segment distances.
        '''
        return self.seg_dists

    def get_incremental_path_dists(self) -> np.array:
        '''
        Returns vector of incremental path distances.
        '''
        seg_dists_np = np.array(self.seg_dists)
        incr_path_dists = self.comp_incr_path_dist(seg_dists_np)
        return incr_path_dists

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
                     gen_future: bool = False):
        '''
        Generates a single BEV representation.

        Args:

        Returns:
            bevs: List of dictionaries containg probabilistic semantic gridmaps
                  and trajectory information.

        '''
        # Build up input dictonary
        pcs = {}
        poses = {}

        # 'Present' pose is origo
        bev_frame_coords = np.array(self.poses[present_idx])

        pc_present = np.concatenate(self.sem_pcs[:present_idx])
        poses_present = np.concatenate([self.poses[:present_idx]])

        # Transform 'absolute' --> 'bev' coordinates
        pc_present[:, :3] = pc_present[:, :3] - bev_frame_coords
        poses_present = poses_present - bev_frame_coords

        pcs.update({'pc_present': pc_present})
        poses.update({'poses_present': poses_present})

        if gen_future:
            pc_future = np.concatenate(self.sem_pcs[present_idx:])
            poses_future = np.concatenate([self.poses[present_idx:]])

            # Transform 'absolute' --> 'bev' coordinates
            pc_future[:, :3] = pc_future[:, :3] - bev_frame_coords
            poses_future = poses_future - bev_frame_coords

            pcs.update({'pc_future': pc_future})
            poses.update({'poses_future': poses_future})

        # Generate BEVs in parallel
        # Package inputs as a tuple for multiprocessing
        bev_gen_inputs = [(pcs, poses)] * bev_num
        pool = Pool(processes=bev_num)
        bevs = pool.map(self.sem_bev_generator.generate_multiproc,
                        bev_gen_inputs)

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

    def viz_bev(self, bev, file_path):
        '''
        Visualizes a BEV using the BEV generator's visualization function.
        '''
        self.sem_bev_generator.viz_bev(bev, file_path)
