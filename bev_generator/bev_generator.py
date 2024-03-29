import math
import os
import random
import time
from abc import ABC, abstractmethod

import numpy as np


class BEVGenerator(ABC):
    '''
    '''

    def __init__(self,
                 view_size: int,
                 pixel_size: int,
                 max_trans_radius: float = 0.,
                 zoom_thresh: float = 0.,
                 do_warp: bool = False,
                 int_scaler: float = 1.,
                 int_sep_scaler: float = 1.,
                 int_mid_threshold: float = 0.5,
                 height_filter=None):
        '''
        '''
        # View frame size in [m]
        self.view_size = view_size
        # View frame size in [px]
        self.pixel_size = pixel_size

        # Random augmentation parameters
        self.max_trans_radius = max_trans_radius
        self.zoom_thresh = zoom_thresh
        self.do_warp = do_warp
        if self.max_trans_radius > 0. or self.zoom_thresh > 0.:
            self.do_aug = True
        else:
            self.do_aug = False

        self.int_scaler = int_scaler
        self.int_sep_scaler = int_sep_scaler
        self.int_mid_threshold = int_mid_threshold

        # Column index of semantic information
        # E.g. [x,y,z,i,r,g,b,sem,dyn]
        self.sem_idx = 7

        # Remove point above ego-vehicle height
        # NOTE: Ego pose == static World frame origin in NuScenes
        self.height_filter = height_filter
        if self.height_filter is not None:
            print("NOTE: Removes points above ego-vehicle height!")

    @abstractmethod
    def generate_bev(self, pc_present: np.array, pc_future: np.array,
                     trajs_present: np.array, trajs_future: np.array):
        '''
        Implement this function to generate actual BEV representations from the
        preprocessed semantic point cloud and trajs.
        '''
        pass

    def generate(self,
                 pcs: dict,
                 trajs: dict,
                 rot_ang: float = 0.,
                 trans_dx: float = 0.,
                 trans_dy: float = 0.,
                 zoom_scalar: float = 1.,
                 do_warping: bool = False):
        '''
        '''
        # Extract semantic point cloud and pose information
        pc_present, pc_future, pc_full = self.extract_pc_dict(pcs)
        ego_trajs = self.extract_ego_traj_dict(trajs)
        ego_traj_present, ego_traj_future, ego_traj_full = ego_trajs
        other_trajs = self.extract_other_traj_dicts(trajs)
        other_trajs_present, other_trajs_future, other_trajs_full = other_trajs

        if "gt_lanes" in trajs.keys():
            gt_lane_trajs = self.extract_gt_lane_dicts(trajs)
        else:
            gt_lane_trajs = None

        aug_view_size = zoom_scalar * self.view_size

        if do_warping is False:
            rot_ang = 0.5 * np.pi
            if len(ego_traj_present) > 1:
                dx = ego_traj_present[-1][0] - ego_traj_present[-2][0]
                dy = ego_traj_present[-1][1] - ego_traj_present[-2][1]
                rot_ang += np.arctan2(dy, dx)
            rot_ang = np.pi - rot_ang

        trajs_present = [ego_traj_present] + other_trajs_present

        pc_present, trajs_present = self.preprocess_pc_and_trajs(
            pc_present, trajs_present, rot_ang, trans_dx, trans_dy,
            aug_view_size)

        if "gt_lanes" in trajs.keys():
            dummy_pc = np.zeros((1, pc_present.shape[1]))
            _, gt_lane_trajs = self.preprocess_pc_and_trajs(
                dummy_pc, gt_lane_trajs, rot_ang, trans_dx, trans_dy,
                aug_view_size)
            # Remove empty GT lanes outside BEV
            gt_lane_trajs = [
                lane for lane in gt_lane_trajs if lane.shape[0] > 0
            ]

        if pc_future is not None:
            trajs_future = [ego_traj_future] + other_trajs_future
            pc_future, trajs_future = self.preprocess_pc_and_trajs(
                pc_future, trajs_future, rot_ang, trans_dx, trans_dy,
                aug_view_size)

            trajs_full = [ego_traj_full] + other_trajs_full
            pc_full, trajs_full = self.preprocess_pc_and_trajs(
                pc_full, trajs_full, rot_ang, trans_dx, trans_dy,
                aug_view_size)

        bev = self.generate_bev(pc_present, pc_future, pc_full, trajs_present,
                                trajs_future, trajs_full, gt_lane_trajs)

        return bev

    def preprocess_pc_and_trajs(self, pc, trajs, rot_ang, trans_dx, trans_dy,
                                aug_view_size):
        '''
        Applies transformations and converts to gridmap coordinates.

        Args:
            pc:
            trajs: List of (N, 3) np.arrays with poses.
        '''
        # Apply transformations (rot, trans, zoom)
        pc = self.geometric_transform(pc, rot_ang, trans_dx, trans_dy,
                                      aug_view_size)

        transf_trajs = []
        for traj in trajs:
            traj = self.geometric_transform(traj,
                                            rot_ang,
                                            trans_dx,
                                            trans_dy,
                                            aug_view_size,
                                            is_traj=True)
            transf_trajs.append(traj)
        trajs = transf_trajs

        # Remove points above ego-vehicle height (for bridges, tunnels etc.)
        if self.height_filter is not None:
            mask = pc[:, 2] < self.height_filter
            pc = pc[mask]

        # Metric to pixel coordinates
        pc = self.pos2grid(pc, aug_view_size)
        trajs = [self.pos2grid(traj, aug_view_size) for traj in trajs]

        return pc, trajs

    def generate_rand_aug(self,
                          pcs: dict,
                          trajs: dict,
                          do_warping: bool = True):
        '''
        '''
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        rot_ang = 2 * np.pi * np.random.random()
        trans_r = self.max_trans_radius * np.random.random()
        trans_ang = 2 * np.pi * np.random.random()
        trans_dx = trans_r * np.cos(trans_ang)
        trans_dy = trans_r * np.sin(trans_ang)
        zoom_scalar = np.random.normal(0, 0.1)
        if zoom_scalar < -self.zoom_thresh:
            zoom_scalar = -self.zoom_thresh
        elif zoom_scalar > self.zoom_thresh:
            zoom_scalar = self.zoom_thresh
        zoom_scalar = 1 + zoom_scalar

        bev = self.generate(pcs, trajs, rot_ang, trans_dx, trans_dy,
                            zoom_scalar, do_warping)

        return bev

    def generate_multiproc(self, bev_gen_inputs):
        '''
        '''
        pcs, trajs = bev_gen_inputs

        if self.do_aug:
            bev = self.generate_rand_aug(pcs, trajs)
        else:
            bev = self.generate(pcs, trajs)

        return bev

    def generate_rand_aug_multiproc(self, bev_gen_inputs):
        '''
        '''
        pcs, trajs = bev_gen_inputs

        bev = self.generate_rand_aug(pcs, trajs, do_warping=True)

        return bev

    def geometric_transform(self,
                            pc_mat: np.array,
                            rot_ang: float,
                            trans_dx: float,
                            trans_dy: float,
                            aug_view_size: float,
                            is_traj: bool = False) -> np.array:
        '''
        Function to transform both point cloud and pose matrices.

        Args:
            pc_mat: Matrix of N row vectors (x,y,z, ... ) having dim (N, 3 + M)
            rot_ang:
            trans_dx:
            trans_dy:
            is_traj: Crop points and include intersection points.
        '''
        # Rotation
        rot_mat = self.rotation_matrix_3d(rot_ang)
        xyz = pc_mat[:, :3]
        xyz = np.matmul(rot_mat, xyz.T).T
        pc_mat[:, :3] = xyz
        # Translation
        pc_mat[:, 0] += trans_dx
        pc_mat[:, 1] += trans_dy
        # Zoom
        if is_traj:
            pc_mat = self.crop_trajectory(pc_mat, aug_view_size)
        else:
            pc_mat = self.crop_view(pc_mat, aug_view_size)
        return pc_mat

    @staticmethod
    def crop_view(pc_mat: np.array, aug_view_size: float):
        '''
        Removes points outside the view frame.

        Args:
            pc: np.array (N, M)
                    [x, y, ...]
        '''
        mask = np.logical_and(pc_mat[:, 0] > -0.5 * aug_view_size,
                              pc_mat[:, 0] < 0.5 * aug_view_size)
        pc_mat = pc_mat[mask]
        mask = np.logical_and(pc_mat[:, 1] > -0.5 * aug_view_size,
                              pc_mat[:, 1] < 0.5 * aug_view_size)
        pc_mat = pc_mat[mask]

        return pc_mat

    def crop_trajectory(self,
                        traj: np.array,
                        aug_view_size: float,
                        thresh: float = 1e-4):
        '''
        Removes points outside the view frame with edge interpolation.

        Args:
            traj: (N, 3) [x, y, z]
            thesh: Numerical accuracy of intesection points.
        '''
        bx0 = -0.5 * aug_view_size
        by0 = -0.5 * aug_view_size
        bx1 = 0.5 * aug_view_size
        by1 = 0.5 * aug_view_size
        bbox = [bx0, by0, bx1, by1]
        new_traj = []
        for idx in range(traj.shape[0] - 1):
            pnt_0_x, pnt_0_y = list(traj[idx][:2])
            pnt_1_x, pnt_1_y = list(traj[idx + 1][:2])

            pnt_0_z = traj[idx][2]

            pnt_0_in = self.point_in_box(pnt_0_x, pnt_0_y, bx0, by0, bx1, by1)
            pnt_1_in = self.point_in_box(pnt_1_x, pnt_1_y, bx0, by0, bx1, by1)

            # Case 1: Edge outside
            if not pnt_0_in and not pnt_1_in:
                continue
            # Case 2: Edge inside
            elif pnt_0_in and pnt_1_in:
                new_traj.append([pnt_0_x, pnt_0_y, pnt_0_z])
            # Case 3: Edge intersection (first in, second out)
            elif pnt_0_in and not pnt_1_in:
                new_traj.append([pnt_0_x, pnt_0_y, pnt_0_z])
                # Intersection point
                inter_x, inter_y, _ = self.cal_intersec_pnt(
                    pnt_0_x, pnt_0_y, pnt_1_x, pnt_1_y, bbox)
                new_traj.append([inter_x, inter_y, pnt_0_z])

            # Case 4: Edge intersection (first out, second int)
            elif not pnt_0_in and pnt_1_in:
                # Intersection point
                inter_x, inter_y, _ = self.cal_intersec_pnt(
                    pnt_0_x, pnt_0_y, pnt_1_x, pnt_1_y, bbox, thresh)

                new_traj.append([inter_x, inter_y, pnt_0_z])
            else:
                raise ValueError('Undefined trajectory points:\n',
                                 f'    pnt_0: ({pnt_0_x, pnt_0_y})\n',
                                 f'    pnt_1: ({pnt_1_x, pnt_1_y})')

        # Handle entirely removed trajectories
        if len(new_traj) == 0:
            new_traj = np.zeros((0, 3))
        else:
            new_traj = np.array(new_traj)

        return new_traj

    @staticmethod
    def point_in_box(pnt_x, pnt_y, box_x0, box_y0, box_x1, box_y1):
        return (box_x0 < pnt_x and pnt_x < box_x1) and (box_y0 < pnt_y
                                                        and pnt_y < box_y1)

    def cal_intersec_pnt(self, x0, y0, x1, y1, bbox, thresh=1e-4):
        '''
        Finds the bounding box intersection point using a midpoint iterative
        refinement appraoch.

        NOTE: Presumes the line actually intersects the bbox!

        Args:
            x0:
            y0:
            x1:
            y1:
            bbox: [bx0, by0, bx1, by1]
        '''
        bx0, by0, bx1, by1 = bbox
        diff = np.inf
        iters = 0
        while diff > thresh:
            x_mid = 0.5 * (x0 + x1)
            y_mid = 0.5 * (y0 + y1)

            # If pnt0 is inside ==> pnt1 must be outside (and vice-versa)
            pnt0_in = self.point_in_box(x0, y0, bx0, by0, bx1, by1)
            mid_in = self.point_in_box(x_mid, y_mid, bx0, by0, bx1, by1)

            # Middle pnt inside ==> Replace inside pnt
            if mid_in:
                if pnt0_in:
                    diff = np.sqrt((x_mid - x0)**2 + (y_mid - y0)**2)
                    x0 = x_mid
                    y0 = y_mid
                else:
                    diff = np.sqrt((x_mid - x1)**2 + (y_mid - y1)**2)
                    x1 = x_mid
                    y1 = y_mid

            # Middle pnt outside ==> Replace outside pnt
            else:
                if not pnt0_in:
                    diff = np.sqrt((x_mid - x0)**2 + (y_mid - y0)**2)
                    x0 = x_mid
                    y0 = y_mid
                else:
                    diff = np.sqrt((x_mid - x1)**2 + (y_mid - y1)**2)
                    x1 = x_mid
                    y1 = y_mid

            iters += 1

        return x_mid, y_mid, iters

    def gen_sem_probmap(self, pc: np.array, sem_clss: list):
        '''
        Generates a probabilistic map of a given semantic class modeled as a
        Dirichlet distribution for ever grid map element.

        Args:
            pc: dim (N, M) [x, y, ..., sem].
            sem_clss: List of string specifying the semantic to extract
                      (e.g. ['road']).
        '''
        sem_idxs = [self.sem_idxs[sem_cls] for sem_cls in sem_clss]
        # Partition point cloud into 'semantic' and 'non-semantic' components
        pc_sem, pc_not_sem = self.partition_semantic_pc(
            pc, sem_idxs, self.sem_idx)
        # Count number of 'semantic' and 'non-semantic' points in each element
        gridmap_sem = self.gen_gridmap_count_map(pc_sem)
        gridmap_not_sem = self.gen_gridmap_count_map(pc_not_sem)

        gridmaps = [gridmap_sem, gridmap_not_sem]
        probmap_sem, _ = self.dirichlet_dist_expectation(gridmaps)

        return probmap_sem

    def gen_intensity_map(self, pc: np.array, sem_cls: str):
        '''
        Generates a normalized intensity map from points belonging to a
        semantic class.

        Args:
            pc: dim (N, M) [x, y, z, i, ... ].
        '''
        sem_idxs = [self.sem_idxs[sem_cls]]
        # Partition point cloud into 'semantic' and 'non-semantic' components
        pc_sem, _ = self.partition_semantic_pc(pc, sem_idxs, self.sem_idx)

        pc_int = pc_sem[:, 3]
        gridmap_int_sum = self.gen_gridmap_count_map(pc_sem, weights=pc_int)
        gridmap_count = self.gen_gridmap_count_map(pc_sem)

        # Summed intensity --> average intensity
        gridmap_int = gridmap_int_sum / (gridmap_count + 1)

        return gridmap_int

    @staticmethod
    def partition_semantic_pc(pc_mat: np.array, sems: list,
                              sem_idx: int) -> np.array:
        '''
        Partitions a point cloud into 'semantic' and 'not semantic' components.
        Args:
            pc: (N, M) [x, y, ..., sem]
            sems: List of integers representing semanntics
            sem_idx: Column index of semantic information
        '''
        # Create a mask for all points having semantics
        mask = np.zeros(pc_mat.shape[0], dtype=bool)
        for sem in sems:
            mask = np.logical_or(mask, pc_mat[:, sem_idx] == sem)

        pc_sem = pc_mat[mask]
        inv_mask = np.invert(mask)
        pc_notsem = pc_mat[inv_mask]

        return pc_sem, pc_notsem

    def gen_gridmap_count_map(self,
                              pc: np.array,
                              weights: np.array = None) -> np.array:
        '''
        Generates a gridmap with number of points in each grid element.
        '''
        ij = pc[:, :2]
        gridmap_counts, _, _ = np.histogram2d(
            ij[:, 1],
            ij[:, 0],
            range=[[0, self.pixel_size], [0, self.pixel_size]],
            bins=[self.pixel_size, self.pixel_size],
            weights=weights)

        # Image to Cartesian coordinate axis direction
        gridmap_counts = np.flip(gridmap_counts, axis=0)

        return gridmap_counts

    @staticmethod
    def dirichlet_dist_expectation(gridmaps, obs_weight=1):
        '''
        Args:
            gridmaps: List of np.array() with observation counts for each
                      semantic.

        Returns:
            post_gridmaps: List of np.array() with posterior probability
                           gridmaps.
        '''
        n_gridmaps = len(gridmaps)
        gridmaps = np.stack(gridmaps)
        gridmaps *= obs_weight  # Consider prev. downsampling of observations

        # Uniform prior
        gridmaps += 1.

        alpha0 = np.sum(gridmaps, axis=0)
        gridmaps /= alpha0

        gridmaps = [gridmaps[idx] for idx in range(n_gridmaps)]

        return gridmaps

    @staticmethod
    def warp_dense_probmaps(probmaps, a_1, a_2, b_1, b_2):
        '''
        Warps a dense 2D array 'A' using polynomial warping.

        The warping is defined by transforming a point (i, j) in the original
        array to (i', j') in the warped array.

        Maximum warping limit approx. +-15% of length (80 px for 512 px input).

        Args:
            probmaps (3D float np.array) : Input dense arrays w. dim (n,h,w).
            i_orig (int) : Row coordinate of the original dense array.
            j_orig (int) : Column coordinate.
            i_warped (int) : Row coordinate of the warped point.
            j_warped (int) : Column coordinate

        Return:
            B (2D float np.array) : Warped dense array.
        '''
        # Get dimensionality
        N, I, J = probmaps.shape

        # For each grid point in B, find corresponding grid point in B and copy
        B = np.zeros((N, I, J))
        for i_warp in range(I):
            for j_warp in range(J):
                i = int(np.rint(a_1 * i_warp + a_2 * i_warp**2))
                j = int(np.rint(b_1 * j_warp + b_2 * j_warp**2))

                # Ensure that the transformed indices are in range
                if i < 0:
                    i = 0
                elif i >= I:
                    i = I - 1
                if j < 0:
                    j = 0
                elif j >= J:
                    j = J - 1

                # NOTE: First index correspond to ROWS, second to COLUMNS!
                B[:, j_warp, i_warp] = probmaps[:, j, i]

        return B

    def warp_sparse_points(self, pnts, a_1, a_2, b_1, b_2, i_mid, j_mid,
                           i_warp, j_warp):
        '''
        '''
        # NOTE No idea why, but the j warping must be reverse ...
        j_warp_rev = self.pixel_size - j_warp
        b_1_rev, b_2_rev = self.cal_warp_params(j_warp_rev, j_mid,
                                                self.pixel_size - 1)

        pnts_xy = list(zip(pnts[:, 0], pnts[:, 1]))

        pnts_xy = self.warp_points(pnts_xy, a_1, a_2, b_1_rev, b_2_rev,
                                   self.pixel_size, self.pixel_size)
        pnts_xy = [[i for i, _ in pnts_xy], [j for _, j in pnts_xy]]

        pnts[:, 0] = pnts_xy[0]
        pnts[:, 1] = pnts_xy[1]

        return pnts

    @staticmethod
    def warp_point(x, y, a_1, a_2, b_1, b_2, I, J):
        '''
        Transforms (x, y) in array coordinates to warped coordinates (x', y').

        The warping is defined by transforming a point (i, j) in the original
        array to (i', j') in the warped array.

        Maximum warping limit approx. +-15% of length (80 px for 512 px input).

        Args:
            x (float) : x-coordinate to be warped
            y (float) : y-coordinate
            i_orig (int) : Row coordinate of the original dense array.
            j_orig (int) : Column coordinate.
            i_warped (int) : Row coordinate of the warped point.
            j_warped (int) : Column coordinate
            I (int) : Number of rows
            J (int) : Number of columns

        Return:
            (float, float) : Tuple of the warped (x', y') coordinates.
        '''
        # Inverse function breaks down in case of no warping (a_2, b_2 = 0)
        if math.isclose(a_2, 0.0, abs_tol=1e-6):
            x_warped = x
        else:
            x_warped = int(
                np.rint((-a_1 + np.sqrt(a_1**2 + 4.0 * a_2 * x)) / (2 * a_2)))

        if math.isclose(b_2, 0.0, abs_tol=1e-6):
            y_warped = y
        else:
            y_warped = int(
                np.rint((-b_1 + np.sqrt(b_1**2 + 4.0 * b_2 * y)) / (2 * b_2)))

        # Ensure that the transformed coordinates are in range
        if x_warped < 0:
            x_warped = 0
        elif x_warped >= I:
            x_warped = I - 1
        if y_warped < 0:
            y_warped = 0
        elif y_warped >= J:
            y_warped = J - 1

        return (x_warped, y_warped)

    def warp_points(self, pnt_list, a_1, a_2, b_1, b_2, I, J):
        '''
        Transforms a set of points (x, y) in array coordinates to warped
        coordinates (x', y').

        Args:
            pnt_list : List of points [(i, j)_0, (i, j)_1, ...].
            i_orig (int) : Row coordinate of the original dense array.
            j_orig (int) : Column coordinate.
            i_warped (int) : Row coordinate of the warped point.
            j_warped (int) : Column coordinate
            I (int) : Number of rows
            J (int) : Number of columns

        Return:
            List of warped points [(i', j')_0, (i', j')_1, ...].
        '''
        warped_pnt_list = []
        # Warp point location one-by-one
        for pnt in pnt_list:
            i_new, j_new = self.warp_point(pnt[0], pnt[1], a_1, a_2, b_1, b_2,
                                           I, J)
            warped_pnt_list.append((i_new, j_new))

        return warped_pnt_list

    @staticmethod
    def get_random_warp_params(mean_ratio, max_ratio, I, J):
        '''
        Returns random warping parameters sampled from a Gaussian distribution.

        Args:
            mean_ratio (float) : Normalized value specifying mean of distr.
            max_ratio (float) : Normalized value specifying maximum warping.
            I (int) : Dimension of image frame.
            J (int) :

        Returns:
            Warp parameters (i_warp, j_warp) as an int tuple.
        '''

        max_val = max_ratio * (I / 2.0)
        mean_val = mean_ratio * max_val

        i_warp = np.random.normal(mean_val, max_val)
        j_warp = np.random.normal(mean_val, max_val)

        if abs(i_warp) > max_val:
            i_warp = max_val
        if abs(j_warp) > max_val:
            j_warp = max_val

        # Random sign
        if random.random() < 0.5:
            i_warp = -i_warp
        if random.random() < 0.5:
            j_warp = -j_warp

        I_mid = int(I / 2)
        J_mid = int(J / 2)

        return (I_mid + i_warp, J_mid + j_warp)

    @staticmethod
    def cal_warp_params(idx_0, idx_1, idx_max):
        '''
        Calculates the polynomial warping coefficients (a_1, a_2).

        The coefficient is found by solving the following second-order
        polynomial equations
            idx_1 = a_0 + a_1 * idx_0 + a_2 * idx_0^2

        Having the boundary conditions
            1) idx_1 = 0 and idx_0 = 0
            2) idx_1 = idx_max and idx_0 = idx_max
            3) idx_1 = idx_1_t and idx_0 = idx_0_t
        idx_0_t, idx_1_t correspond to a specified index.

        Args:
            idx_0 (int) : Original coordinate.
            idx_1 (int) : Transformed coordinate.
            idx_max (int) : Length of coordinate range.

        Returns:
            (float, float) : Tuple of warping parameters (a_1, a_2)
        '''
        a_1 = (idx_1 - idx_0**2 / idx_max) / (idx_0 * (1.0 - idx_0 / idx_max))
        a_2 = (1.0 - a_1) / idx_max
        return (a_1, a_2)

    def warp_trajs(self, trajs, a_1, a_2, b_1, b_2, i_mid, j_mid, i_warp,
                   j_warp):
        '''
        Warp a set of trajectories using the same parameters.

        Args:
            trajs: List of np.array pose matrices (N,3).
        '''
        trajs_warped = []
        for traj in trajs:
            traj = self.warp_sparse_points(traj, a_1, a_2, b_1, b_2, i_mid,
                                           j_mid, i_warp, j_warp)
            trajs_warped.append(traj)
        return trajs_warped

    @staticmethod
    def extract_pc_dict(pcs: dict):
        pc_past = pcs['pc_present']
        pc_future = pcs['pc_future']
        pc_full = pcs['pc_full']
        return pc_past, pc_future, pc_full

    @staticmethod
    def extract_ego_traj_dict(trajs: dict) -> tuple:
        ego_traj_past = trajs['ego_traj_present']
        ego_traj_future = trajs['ego_traj_future']
        ego_traj_full = trajs['ego_traj_full']
        return ego_traj_past, ego_traj_future, ego_traj_full

    @staticmethod
    def extract_other_traj_dicts(trajs: dict) -> tuple:
        other_trajs_past = trajs['other_trajs_present']
        other_trajs_future = trajs['other_trajs_future']
        other_trajs_full = trajs['other_trajs_full']
        return other_trajs_past, other_trajs_future, other_trajs_full

    @staticmethod
    def extract_gt_lane_dicts(trajs: dict) -> tuple:
        gt_lane_trajs = trajs['gt_lanes']
        return gt_lane_trajs

    @staticmethod
    def extract_aug_dict(augs: dict):
        max_trans_radius = augs['max_translation_radius']
        zoom_threshold = augs['zoom_threshold']
        return max_trans_radius, zoom_threshold

    @staticmethod
    def rotation_matrix_3d(ang):
        return np.array([[np.cos(ang), -np.sin(ang), 0],
                         [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])

    def pos2grid(self, pc_mat, view_size):
        '''
        Args:
            pc_mat: np.array (N, M)
                    [x, y, ...]
        '''
        pc_mat[:,
               0:2] = np.floor(pc_mat[:, 0:2] / view_size * self.pixel_size +
                               0.5 * self.pixel_size)

        return pc_mat

    @abstractmethod
    def viz_bev(self):
        '''
        Implement this function for visualizing the BEV representations
        created.
        '''
        pass
