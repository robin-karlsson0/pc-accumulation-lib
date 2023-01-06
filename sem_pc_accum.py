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

    Explanation how an 'observation' is integrated
        NOTE: The structure of an 'observation' is platform dependent

    1. Run the integration function
           sem_pc_accum.integrate(observation)

    2. Unpack 'observation' into a point cloud and RGB image(s)
           rgbs = obs['images']
           pc = obs['pc']

    3. Transform 'observation' to semantic point cloud (sem_pc) in vector space
           sem_pc, pose, semseg, T = obs2sem_vec_space(rgbs, pc)

    4. Update relative pose of all stored poses and sem_pc:s
           new_pose   := T_new_prev old_pose
           new_sem_pc := T_new_prev old_sem_pc

    5. Store new pose and sem_pc
           ==> Latest ego pose (0, 0, 0)) and observations

    6. Remove all old (pose, sem_pc) beyond "memory horizon"
           if path_dist > thresh:
               remove (pose, sem_pc)

    7. Return the number of removed (pose, sem_pc) to allow book keeping
       from calling code (e.g. computing distance between BEV generations)

    '''

    def __init__(self, horizon_dist: float, icp_threshold: float,
                 semseg_onnx_path: str, semseg_filters: list, sem_idxs: dict,
                 use_gt_sem: bool, bev_params: dict):
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
            use_gt_sem: Uses GT semantic classes for each point if true.
        '''
        # Semantic segmentation model
        self.semseg_model = None
        if use_gt_sem is False:
            self.semseg_model = SemSegONNX(semseg_onnx_path)
        self.semseg_filters = semseg_filters
        self.sem_idxs = sem_idxs
        self.use_gt_sem = use_gt_sem

        self.icp_threshold = icp_threshold

        self.icp_trans_init = np.eye(4)

        # Initial pose and transformation matrix
        self.T_prev_origin = np.eye(4)

        # Point cloud of last observations
        self.pcd_prev = None

        self.horizon_dist = horizon_dist

        self.sem_pcs = []  # (N)
        self.poses = []  # (N)
        self.seg_dists = []  # (N-1)
        self.rgbs = []
        self.semsegs = []

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
                bev_params['do_warp'],
                bev_params['int_scaler'],
                bev_params['int_sep_scaler'],
                bev_params['int_mid_threshold'],
            )
        elif bev_params['type'] == 'rgb':
            self.sem_bev_generator = RGBBEVGenerator(
                bev_params['view_size'],
                bev_params['pixel_size'],
                0,
                bev_params['max_trans_radius'],
                bev_params['zoom_thresh'],
                bev_params['do_warp'],
                bev_params['int_scaler'],
                bev_params['int_sep_scaler'],
                bev_params['int_mid_threshold'],
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

        sem_pcs (list)] [ sem_pc_1, sem_pc_2, ... ]
        pooses (list): [ [x,y,z]_0, [x,y,z]_1, ... ]

        Args:
            observations: List of K tuples (rgb, pc)
        '''
        raise NotImplementedError()

    def update_poses(self, T_new_prev):
        # Transform previous poses to new ego coordinate system
        new_poses = []
        for pose_ in self.poses:
            # Homogeneous spatial coordinates
            new_pose = np.matmul(T_new_prev, np.array([pose_ + [1]]).T)
            new_pose = new_pose[:, 0][:-1]  # (4,1) --> (3)
            new_pose = list(new_pose)
            new_poses.append(new_pose)
        self.poses = new_poses

    def update_sem_pcs(self, T_new_prev):
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

    def remove_observations(self):
        idx = 0  # Default value for no removed observations
        # Compute path segment distance
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

        return idx, path_length

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

    def obs2sem_vec_space(self,
                          rgb: Image,
                          pc: np.array,
                          sem_gt: np.array = None) -> tuple:
        '''
        Abstract class
        '''
        raise NotImplementedError()

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

    def get_rgb(self, idx: int = None) -> list:
        '''
        Returns one or all rgb images (PIL.Image) depending on 'idx'.
        '''
        if idx is None:
            return self.rgbs
        else:
            return [self.rgbs[idx]]

    def get_semseg(self, idx: int = None) -> list:
        '''
        Returns one or all semseg outputs (np.array) depending on 'idx'.
        '''
        if idx is None:
            return self.semsegs
        else:
            return [self.semsegs[idx]]

    def generate_bev(self,
                     present_idx: int = None,
                     bev_num: int = 1,
                     gen_future: bool = False):
        '''
        Generates a single BEV representation.
        Args:
            present_idx: Concatenate all point clouds up to the index.
                         NOTE: The default value concatenates all point clouds.
        Returns:
            bevs: List of dictionaries containg probabilistic semantic gridmaps
                  and trajectory information.
        '''
        # Build up input dictonary
        pcs = {}
        poses = {}

        # 'Present' pose is origo
        if present_idx is None:
            bev_frame_coords = np.array(self.poses[-1])
        else:
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

            pc_full = np.concatenate(self.sem_pcs)
            poses_full = np.concatenate([self.poses])

            # Transform 'absolute' --> 'bev' coordinates
            pc_future[:, :3] = pc_future[:, :3] - bev_frame_coords
            poses_future = poses_future - bev_frame_coords

            pc_full[:, :3] = pc_full[:, :3] - bev_frame_coords
            poses_full = poses_full - bev_frame_coords
        else:
            pc_future = None
            poses_future = None
            pc_full = None
            poses_full = None
        pcs.update({'pc_future': pc_future})
        poses.update({'poses_future': poses_future})
        pcs.update({'pc_full': pc_full})
        poses.update({'poses_full': poses_full})

        if bev_num == 1:
            bev_gen_inputs = (pcs, poses)
            bevs = self.sem_bev_generator.generate_multiproc(bev_gen_inputs)
            # Mimic multiprocessing list output
            bevs = [bevs]
        else:
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

    def gen_semantic_pc(self, pc_velo, semantic_map, P_velo_frame):
        """
        Returns a subset of points with semantic content from the semantic map.

        Args:
            P_velo_frame: np.array (3, 4) Velodyne coords --> Image frame coords.
            semantic_map: np.array (h, w, k) w. K layers.

        Returns:
            pc_velo_sem: np.array (M, 4+K) [x, y, z, i, sem_1, ... , sem_K]
        """
        img_h, img_w, _ = semantic_map.shape

        pc_velo_img = self.velo2img(pc_velo, P_velo_frame, img_h, img_w)

        u = pc_velo_img[:, -2].astype(int)
        v = pc_velo_img[:, -1].astype(int)

        sem = semantic_map[v, u, :]

        pc_velo_sem = np.concatenate([pc_velo_img[:, :4], sem], axis=1)

        return pc_velo_sem

    @staticmethod
    def velo2frame(pc_velo, P_velo_frame):
        """
        Transforms point cloud from 'velodyne' to 'image frame' coordinates.

        Args:
            pc_velo: np.array (N, 3)
            P_velo_frame: np.array (3, 4)
        """
        # Covnert point cloud to homogeneous coordinates
        pc_num = pc_velo.shape[0]
        pc_homo_velo = np.concatenate((pc_velo, np.ones((pc_num, 1))), axis=1)
        pc_homo_velo = pc_homo_velo.T

        # Transform point cloud 'velodyne' --> 'frame'
        pc_homo_frame = np.matmul(P_velo_frame, pc_homo_velo)
        pc_homo_frame = pc_homo_frame.T

        return pc_homo_frame

    def velo2img(self, pc_velo, P_velo_frame, img_h, img_w, max_depth=np.inf):
        """
        Compures image coordinates for points and returns the point cloud
        contained in the image.

        Args:
            pc_velo: np.array (N, 4) [x, y, z, i]
            P_velo_frame: np.array (3, 4)
            img_h: int
            img_w: int
            max_depth: float

        Returns:
            pc_velo_frame: np.array (M, 6) [x, y, z, i, img_i, img_j]
        """
        pc_frame = self.velo2frame(pc_velo[:, :3], P_velo_frame)

        depth = pc_frame[:, 2]
        depth[depth == 0] = -1e-6
        u = np.round(pc_frame[:, 0] / np.abs(depth)).astype(int)
        v = np.round(pc_frame[:, 1] / np.abs(depth)).astype(int)

        # Generate mask for points within image
        mask = np.logical_and(
            np.logical_and(np.logical_and(u >= 0, u < img_w), v >= 0),
            v < img_h)
        mask = np.logical_and(np.logical_and(mask, depth > 0),
                              depth < max_depth)
        # Convert to column vectors
        u = u[:, np.newaxis]
        v = v[:, np.newaxis]

        pc_velo_img = np.concatenate([pc_velo, u, v], axis=1)
        pc_velo_img = pc_velo_img[mask]

        return pc_velo_img

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

        sem = sem_pc[:, 7]
        yellow = np.array([[253, 231, 36]])
        blue = np.array([[68, 2, 85]])
        N = sem.shape[0]
        rgb = np.zeros((N, 3))
        for idx in range(N):
            if sem[idx] == 0:
                rgb[idx] = yellow
            else:
                rgb[idx] = blue

        # rgb = sem_pc[:, 4:7]
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

    def viz_bev(self, bev, file_path, rgbs: list = [], semsegs: list = []):
        '''
        Visualizes a BEV using the BEV generator's visualization function.
        '''
        self.sem_bev_generator.viz_bev(bev, file_path, rgbs, semsegs)
