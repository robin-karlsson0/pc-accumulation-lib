import numpy as np
import open3d as o3d
import PIL.Image as Image

from sem_pc_accum import SemanticPointCloudAccumulator


class Kitti360SemanticPointCloudAccumulator(SemanticPointCloudAccumulator):

    def __init__(self, horizon_dist: float, calib_params: dict,
                 icp_threshold: float, semseg_onnx_path: str,
                 semseg_filters: list, sem_idxs: dict, use_gt_sem: bool,
                 bev_params: dict):
        """
        Args:
            horizon_dist (float): maximum distance that ego vehicle traveled
                within an accumulated pointcloud. If ego vehicle travels more
                than this, past pointclouds will be discarded.
            calib_params: h_velo_cam: np.array,
                          p_cam_frame: np.array,
                          p_velo_frame: np.array,
                          c_x, c_y, f_x, f_y: int
                              calib_params['c_x'] --> c_x
                              calib_params['c_y'] --> c_y
                              calib_params['f_x'] --> f_x
                              calib_params['f_y'] --> f_y
            icp_threshold (float): not used if using ground truth ego pose
            semseg_onnx_path (str): path to onnx file defining semseg model
            semseg_filters (list[int]): List of semantic class idxs to filter
                out from point cloud.
            sem_idxs (dict): mapping semseg class to str
            bev_params (dict):
        """
        super().__init__(horizon_dist, icp_threshold, semseg_onnx_path,
                         semseg_filters, sem_idxs, use_gt_sem, bev_params)

        # Calibration parameters
        self.H_velo_cam = calib_params['h_velo_cam']
        self.P_cam_frame = calib_params['p_cam_frame']
        self.P_velo_frame = calib_params['p_velo_frame']

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
        if self.use_gt_sem:
            rgb, pc, sem_gt = observations[0]
            sem_pc, pose, semseg, T_new_prev = self.obs2sem_vec_space(
                rgb, pc, sem_gt)
        else:
            rgb, pc, _ = observations[0]
            sem_pc, pose, semseg, T_new_prev = self.obs2sem_vec_space(rgb, pc)

        # Transform previous poses and sem_pcs to new ego coordinate system
        if len(self.poses) > 0:
            self.update_poses(T_new_prev)
            self.update_sem_pcs(T_new_prev)

        self.sem_pcs.append(sem_pc)
        self.poses.append(pose)
        self.rgbs.append(rgb)
        self.semsegs.append(semseg)

        # Remove observations beyond "memory horizon"
        idx = 0  # Default value for no removed observations
        if len(self.poses) > 1:
            idx, path_length = self.remove_observations()

            print(f'    #pc {len(self.sem_pcs)} |',
                  f'path length {path_length:.2f}')

        # Number of observations removed
        return idx

    def obs2sem_vec_space(self,
                          rgb: Image,
                          pc: np.array,
                          sem_gt: np.array = None) -> tuple:
        '''
        Converts a new observation to a semantic point cloud in the common
        vector space.

        The function maintains the most recent pointcloud and transformation
        for the next observation update.

        Args:
            rgb: RGB image.
            pc: Point cloud as row vector matrix w. dim (N, 4)
                [x, y, z, intensity]
            sem_gt: Ground truth semantic class for each point (N, 1)
                    If 'None' --> Compute semantics from image

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
            target, source, self.icp_threshold, self.icp_trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T_new_prev = reg_p2l.transformation
        T_new_origin = np.matmul(self.T_prev_origin, T_new_prev)

        # Semantic point cloud
        if sem_gt is None:
            semseg = self.semseg_model.pred(rgb)[0, 0]
            pc_velo_rgb = self.gen_semantic_pc(pc, np.array(rgb),
                                               self.P_velo_frame)
            pc_velo_sem = self.gen_semantic_pc(pc, np.expand_dims(semseg, -1),
                                               self.P_velo_frame)  # (N, 5)
            pc_velo_rgbsem = np.concatenate((pc_velo_rgb, pc_velo_sem[:, -1:]),
                                            axis=1)
        else:
            semseg = None
            N = sem_gt.shape[0]
            pc_velo_rgb = np.zeros((N, 3))
            pc_velo_sem = sem_gt
            pc_velo_rgbsem = np.concatenate(
                (pc, pc_velo_rgb, pc_velo_sem[:, -1:]), axis=1)

        # Filter out unwanted points according to semantics
        # TODO do this earlier to reduce computation?
        pc_velo_rgbsem = self.filter_semseg_pc(pc_velo_rgbsem)

        # Pose of new observations always ego-centered
        pose = [0., 0., 0.]

        self.T_prev_origin = T_new_origin
        self.pcd_prev = pcd_new

        return pc_velo_rgbsem, pose, semseg, T_new_prev
