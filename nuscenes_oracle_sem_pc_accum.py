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

        Object instance class idxs
            0: 'car'
            1: 'truck'
            2: 'construction_vehicle'
            3: 'bus'
            4: 'trailer'
            5: 'motorcycle'
            6: 'bicycle'
            7: 'pedestrian'

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

        self.ts = 0

        # PC matrix column indices
        self.xyz_idx = 0
        self.int_idx = 3
        self.rgb_idx = 4
        self.sem_idx = 7
        self.inst_idx = 8
        self.dyn_idx = 9

        # 4x4 transformation matrix mapping pnts 'global' --> 'world' frame
        # Specified at first observation integration
        self.T_global_world = None

        # To lift ego pose from ground
        self.ego_pose_z = 1.

        # Fake object detection and tracking system
        # Store observed pose of instances
        #     {token: [(pose, ts)_0, (pose, ts)_1, ...]}
        self.instances = {}
        # Add tokens of detected dynamic instances
        self.dyn_instances = []
        self.dyn_obj_trans_thresh = 1.0  # [m]

        # For each ts, add dictionary {token: idx} for remembering what object
        # instance each pnt belongs to
        #     --> [{token:idx}_ts0, {token:idx}_ts1, ... ]
        self.token2idx = []

        self.track_inst_clss = [0, 1, 2, 3, 5]  # NOTE: Skips 'trailer'

    def integrate(self, observations: list):
        """
        Integrates a sequence of K observations into the common vector space
        (i.e., a world frame which can be different to the global frame of
        NuScenes, for example 1st ego vehicle frame).

        Points in vector space are defined by (x,y,z) coordinates.

        Args:
            observations: List of K dict. One 'obs' dict has the following keys
                images (list[PIL]):
                pc (np.ndarray): (N, 7)
                    [0] --> [2]: (x, y, z) in ego vehicle frame
                    [3]        : Intensity
                    [4] --> [5]: Pixel (u, v) coordinates
                    [6]        : Object instance idx
                pc_cam_idx (np.ndarray): (N,) index of camera where each
                                         point projected onto
                ego_at_lidar_ts (np.ndarray): (4, 4) ego vehicle pose w.r.t
                                              global frame @ timestamp of lidar
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
                                                       T_ego_global,
                                                       self.ego_pose_z)

        self.sem_pcs.append(sem_pc)
        self.poses.append(pose)
        self.rgbs.append(rgbs)
        self.semsegs.append(semsegs)

        ###############################################
        #  Fake object detection and tracking system
        ###############################################
        # Detected objects
        inst_tokens = obs['inst_tokens']
        inst_clss = obs['inst_cls']
        inst_centers = obs['inst_center']

        # Add new token --> pnt inst_idx correspondance dict
        self.token2idx.append({'ts': self.ts})

        # Tracking
        # 'idx' relates 'token' <--> 'cls' <--> 'center'
        for idx, token in enumerate(inst_tokens):
            # 1) Object is a tracked object
            cls = inst_clss[idx]
            if cls not in self.track_inst_clss:
                continue
            # 2) Track object
            # pose = self.get_tf_pose(inst_tfs[idx])
            pose = inst_centers[idx]  # global
            pose = np.expand_dims(pose, 0)  # (3) --> (1,3)
            pose = homo_transform(self.T_global_world, pose)[0]
            if token not in self.instances.keys():
                # Instantiate if first detection
                self.instances[token] = []
            self.instances[token].append((pose, self.ts))
            # 3) Store token --> pnt inst_idx correspondances
            self.token2idx[-1][token] = idx

            # Detect dynamic objects
            # 1) Object is known dynamic object
            if token in self.dyn_instances:
                # Set new observations to dynamic
                inst_idx = self.token2idx[-1][token]
                sem_pc = self.sem_pcs[-1]
                mask = sem_pc[:, self.inst_idx] == inst_idx
                sem_pc[mask, self.dyn_idx] = 1
                self.sem_pcs[-1] = sem_pc
                continue
            # 2) Object has more than one observation
            poses, tss = self.get_obj_inst_poses_ts(self.instances[token])
            if len(poses) < 2:
                continue
            # 3) Compute (x,y) pose change between first and last observation and
            #    check if dynamic
            delta_pose = self.cal_pose_change(poses[0][:2], poses[-1][:2])

            if delta_pose > self.dyn_obj_trans_thresh:
                self.dyn_instances.append(token)

                # Modify prior pnt observations to dynamic
                for pc_ts, sem_pc in enumerate(self.sem_pcs):
                    # Skip if object not observed
                    if token not in self.token2idx[pc_ts].keys():
                        continue
                    inst_idx = self.token2idx[pc_ts][token]
                    mask = sem_pc[:, self.inst_idx] == inst_idx
                    sem_pc[mask, self.dyn_idx] = 1
                    self.sem_pcs[pc_ts] = sem_pc

        # Compute path segment distance
        if len(self.poses) > 1:
            seg_dist = self.dist(np.array(self.poses[-1]),
                                 np.array(self.poses[-2]))
            self.seg_dists.append(seg_dist)
            path_length = np.sum(self.seg_dists)
        else:
            path_length = 0
        print(f'    ts {self.ts} | #pc {len(self.sem_pcs)} |',
              f'path length {path_length:.2f}')

        if self.ts == 38:
            # if (self.ts + 1) % 5 == 0:
            sem_vec_space = np.concatenate(self.sem_pcs, axis=0)

            concat_seq_poses = []
            for _, pose_obss in self.instances.items():

                poses, tss = zip(*pose_obss)

                seq_poses = self.parse_coherent_pose_seqs(poses, tss)
                for seq_pose in seq_poses:

                    # Skip single observations
                    if len(seq_pose) < 2:
                        continue

                    concat_seq_poses.append(seq_pose)

            concat_seq_poses.append(self.poses)

            self.viz_sem_pc(sem_vec_space, (0, 0, 0), 'dyn', concat_seq_poses)

        # Update integration time step
        self.ts += 1

    def parse_coherent_pose_seqs(self, poses, tss):
        seq_tss = self.parse_seq_into_coherent_seqs(tss)

        seq_poses = []
        for seq_ts in seq_tss:

            # Create sequence of consecutive poses
            seq_poses.append([])
            for ts in seq_ts:
                pose = poses[ts]
                seq_poses[-1].append(pose.tolist())

        return seq_poses

    @staticmethod
    def parse_seq_into_coherent_seqs(ts: list):
        '''
        Args:
            ts: List of partially sequential integers [ts_0, ts_1, ... ]
        Returns:
            seq_tss: List of lists of sequential integers
                     [[0, 1], [2, 3, 4], ... ]
        '''
        # Initialize list of lists
        seq_tss = []
        seq_tss.append([])
        t_prev = ts[0] - 1
        seq_idx = 0
        for t in ts:
            t_diff = t - t_prev
            # New consecutive sequence ==> Start new list
            if t_diff != 1:
                seq_tss.append([])
            # Continue consecutive sequence ==> Add to newest list
            seq_tss[-1].append(seq_idx)
            t_prev = t
            seq_idx += 1

        return seq_tss

    def obs2sem_vec_space(self,
                          rgbs: list,
                          pc: np.array,
                          pc_cam_idx: np.array,
                          T_ego_global: np.array,
                          ego_pose_z: float = 0) -> tuple:
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
            pc_velo_rgbsem (np.ndarray): (N, 10) semantic point cloud as row
                                         vector matrix w. dim
                    [0] --> [2]: (x, y, z) in ego vehicle frame
                    [3]        : Intensity
                    [4] --> [6]: RGB
                    [7]        : Semantic idx
                    [8]        : Instance idx
                    [9]        : Dynamic (prob) (0: static, 1: dynamic)
            pose: List with (x, y, z) coordinates as floats.
            semsegs: List with np.array semantic segmentation outputs.
        """
        #######################################
        #  Ego pose (x,y,z) in 'world' frame
        #######################################
        T_ego_world = self.T_global_world @ T_ego_global  # 4x4
        pose = T_ego_world[:3, -1].tolist()
        # Lift ego pose from road surface
        pose[2] += ego_pose_z

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

        # Object instance idx
        pc_inst_idx = pc[:, 6:7]

        # Dynamic observation (probability)
        pc_dyn = np.zeros((pc.shape[0], 1), dtype=float)  # dyn

        pc_velo_rgbsem = np.concatenate(
            [pc_xyz, pc_intensity, pc_rgb_sem, pc_inst_idx, pc_dyn],
            axis=1)  # (N, 9)

        return pc_velo_rgbsem, pose, semsegs

    @staticmethod
    def viz_sem_pc(sem_pc: np.array, origin: tuple, type: str,
                   poses_sets: list()):
        '''
        Ref: http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html

        Args:
            sem_pc: Semantic point cloud as row vector matrix w. dim (N, 10)
            origin: Tuple of (x,y,z) coordinates for crating axes.
            type: String specifying type of point cloud decoration.
            pose_sets: List of lists of lists with (x,y,z) coordinates.
                   [ [pose_0, ... ]_agent2, [pose_0, ... ]_agent1, ... ]
        ''' # noqa
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sem_pc[:, :3])

        if type == 'sem':
            raise NotImplementedError()
            # sem = sem_pc[:, 7]
            # yellow = np.array([[253, 231, 36]])
            # blue = np.array([[68, 2, 85]])
            # N = sem.shape[0]
            # rgb = np.zeros((N, 3))
            # for idx in range(N):
            #     if sem[idx] == 0:
            #         rgb[idx] = yellow
            #     else:
            #         rgb[idx] = blue

            # For visualizing general semantics?
            # rgb = np.tile(sem_pc[:, 4:5], (1, 3))
            # rgb /= np.max(rgb)

        elif type == 'dyn':
            yellow = np.array([[253, 231, 36]])
            blue = np.array([[68, 2, 85]])
            dyn = sem_pc[:, 9]
            N = dyn.shape[0]
            rgb = np.zeros((N, 3))
            for idx in range(N):
                if dyn[idx] == 0:
                    rgb[idx] = blue
                else:
                    rgb[idx] = yellow
            rgb /= 255

        elif type == 'rgb':
            rgb = sem_pc[:, 4:7] / 255

        pcd.colors = o3d.utility.Vector3dVector(rgb)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=origin)

        concat_lines_set = []
        concat_poses_set = []

        new_idx = 0
        for poses in poses_sets:
            lines = [[new_idx + idx, new_idx + idx + 1]
                     for idx in range(len(poses) - 1)]
            concat_lines_set += lines
            concat_poses_set += poses

            new_idx = lines[-1][-1] + 1

        # Add spheres to endpoints to indicate directionality
        mesh_spheres = []
        for poses in poses_sets:
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
            mesh_sphere.compute_vertex_normals()
            mesh_sphere.paint_uniform_color([1.0, 0., 0.])
            # (3,1) translation vector
            mesh_sphere.translate(np.expand_dims(np.array(poses[-1]), 0).T)
            mesh_spheres.append(mesh_sphere)

        # Ego path
        # lines = [[idx, idx + 1] for idx in range(len(poses) - 1)]
        colors = [[1, 0, 0] for _ in range(len(concat_lines_set))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(concat_poses_set),
            lines=o3d.utility.Vector2iVector(concat_lines_set),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries(
            [mesh_frame, line_set, pcd, *mesh_spheres])

    @staticmethod
    def get_tf_pose(inst_tf: np.array) -> np.array:
        '''
        Extracts (x,y,z) pose from a NuScenes object instance tf pose matrix.
        Args:
            inst_tf: (4, 4) pose matrix
        Returns:
            (3,) pose vector (x,y,z)
        '''
        return inst_tf[:3, -1]

    @staticmethod
    def get_obj_inst_poses_ts(inst_obs: list) -> tuple:
        '''
        Args:
            inst_obs: [(pose, ts)_0, (pose, ts)_1, ... ]
                pose: (torch.tensor) (x,y,z) vector
                ts (int): Observation integration step
        Returns:
            poses: [(x,y,z)_0, (x,y,z)_1, ... ]
            tss: [ts_0, ts_1, ...]
        '''
        poses, tss = zip(*inst_obs)
        return poses, tss

    @staticmethod
    def cal_pose_change(pose_0: np.array, pose_1: np.array) -> float:
        '''
        Args:
            pose_0: (x,y,z) vector
            pose_1: (x,y,z) vector
        '''
        return np.linalg.norm(pose_1 - pose_0)
