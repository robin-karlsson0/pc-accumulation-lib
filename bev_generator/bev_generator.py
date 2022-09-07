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
                 zoom_thresh: float = 0.):
        '''
        '''
        # View frame size in [m]
        self.view_size = view_size
        # View frame size in [px]
        self.pixel_size = pixel_size

        # Random augmentation parameters
        self.max_trans_radius = max_trans_radius
        self.zoom_thresh = zoom_thresh
        if self.max_trans_radius > 0. or self.zoom_thresh > 0.:
            self.do_aug = True
        else:
            self.do_aug = False

        print("NOTE: Removes points above ego-vehicle height!")

    @abstractmethod
    def generate_bev(self,
                     pc_present: np.array,
                     pc_future: np.array,
                     poses_present: np.array,
                     poses_future: np.array,
                     do_warping: bool = False):
        '''
        Implement this function to generate actual BEV representations from the
        preprocessed semantic point cloud and poses.
        '''
        pass

    def generate(self,
                 pcs: dict,
                 poses: dict,
                 rot_ang: float = 0.,
                 trans_dx: float = 0.,
                 trans_dy: float = 0.,
                 zoom_scalar: float = 1.,
                 do_warping: bool = False):
        '''
        '''
        # Extract semantic point cloud and pose information
        pc_present, pc_future, pc_full = self.extract_pc_dict(pcs)
        poses_present, poses_future, poses_full = self.extract_pose_dict(poses)

        aug_view_size = zoom_scalar * self.view_size

        if do_warping is False:
            rot_ang = 0.5 * np.pi
            if len(poses_present) > 1:
                dx = poses_present[-1][0] - poses_present[-2][0]
                dy = poses_present[-1][1] - poses_present[-2][1]
                rot_ang += np.arctan2(dy, dx)
            rot_ang = np.pi - rot_ang

        pc_present, poses_present = self.preprocess_pc_and_poses(
            pc_present, poses_present, rot_ang, trans_dx, trans_dy,
            aug_view_size)

        if pc_future is not None:
            pc_future, poses_future = self.preprocess_pc_and_poses(
                pc_future, poses_future, rot_ang, trans_dx, trans_dy,
                aug_view_size)

            pc_full, poses_full = self.preprocess_pc_and_poses(
                pc_full, poses_full, rot_ang, trans_dx, trans_dy,
                aug_view_size)

        bev = self.generate_bev(pc_present, pc_future, pc_full, poses_present,
                                poses_future, poses_full, do_warping)

        return bev

    def preprocess_pc_and_poses(self, pc, poses, rot_ang, trans_dx, trans_dy,
                                aug_view_size):
        '''
        Applies transformations and converts to gridmap coordinates.
        '''
        # Apply transformations (rot, trans, zoom)
        pc = self.geometric_transform(pc, rot_ang, trans_dx, trans_dy,
                                      aug_view_size)
        poses = self.geometric_transform(poses, rot_ang, trans_dx, trans_dy,
                                         aug_view_size)

        # Remove points above ego-vehicle height (for bridges, tunnels etc.)
        mask = pc[:, 2] < 1.
        pc = pc[mask]

        # Metric to pixel coordinates
        pc = self.pos2grid(pc, aug_view_size)
        poses = self.pos2grid(poses, aug_view_size)

        return pc, poses

    def generate_rand_aug(self,
                          pcs: dict,
                          poses: dict,
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

        bev = self.generate(pcs, poses, rot_ang, trans_dx, trans_dy,
                            zoom_scalar, do_warping)

        return bev

    def generate_multiproc(self, bev_gen_inputs):
        '''
        '''
        pcs, poses = bev_gen_inputs

        if self.do_aug:
            bev = self.generate_rand_aug(pcs, poses)
        else:
            bev = self.generate(pcs, poses)

        return bev

    def generate_rand_aug_multiproc(self, bev_gen_inputs):
        '''
        '''
        pcs, poses = bev_gen_inputs

        bev = self.generate_rand_aug(pcs, poses, do_warping=True)

        return bev

    def geometric_transform(self, pc_mat: np.array, rot_ang: float,
                            trans_dx: float, trans_dy: float,
                            aug_view_size: float) -> np.array:
        '''
        Function to transform both point cloud and pose matrices.

        Args:
            pc_mat: Matrix of N row vectors (x,y,z, ... ) having dim (N, 3 + M)
            rot_ang:
            trans_dx:
            trans_dy:
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

    def gen_sem_probmap(self, pc: np.array, sem_cls: str):
        '''
        Generates a probabilistic map of a given semantic class modeled as a
        Dirichlet distribution for ever grid map element.

        Args:
            pc: dim (N, M) [x, y, ..., sem].
            sem_cls: String specifying the semantic to extract (e.g. 'road').
        '''
        sem_idxs = [self.sem_idxs[sem_cls]]
        # Partition point cloud into 'semantic' and 'non-semantic' components
        pc_sem, pc_not_sem = self.partition_semantic_pc(pc, sem_idxs)
        # Count number of 'semantic' and 'non-semantic' points in each element
        gridmap_sem = self.gen_gridmap_count_map(pc_sem)
        gridmap_not_sem = self.gen_gridmap_count_map(pc_not_sem)

        gridmaps = [gridmap_sem, gridmap_not_sem]
        probmap_sem, _ = self.dirichlet_dist_expectation(gridmaps)

        return probmap_sem

    @staticmethod
    def partition_semantic_pc(pc_mat: np.array, sems: list) -> np.array:
        '''
        Partitions a point cloud into 'semantic' and 'not semantic' components.
        Args:
            pc: (N, M) [x, y, ..., sem]
            sems: List of integers representing semanntics
        '''
        # Create a mask for all points having semantics
        mask = np.zeros(pc_mat.shape[0], dtype=bool)
        for sem in sems:
            mask = np.logical_or(mask, pc_mat[:, -1] == sem)

        pc_sem = pc_mat[mask]
        inv_mask = np.invert(mask)
        pc_notsem = pc_mat[inv_mask]

        return pc_sem, pc_notsem

    def gen_gridmap_count_map(self, pc: np.array) -> np.array:
        '''
        Generates a gridmap with number of points in each grid element.
        '''
        ij = pc[:, :2]
        gridmap_counts, _, _ = np.histogram2d(
            ij[:, 1],
            ij[:, 0],
            range=[[0, self.pixel_size], [0, self.pixel_size]],
            bins=[self.pixel_size, self.pixel_size])

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

    @staticmethod
    def extract_pc_dict(pcs: dict):
        pc_past = pcs['pc_present']
        pc_future = pcs['pc_future']
        pc_full = pcs['pc_full']
        return pc_past, pc_future, pc_full

    @staticmethod
    def extract_pose_dict(poses: dict):
        poses_past = poses['poses_present']
        poses_future = poses['poses_future']
        poses_full = poses['poses_full']
        return poses_past, poses_future, poses_full

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
