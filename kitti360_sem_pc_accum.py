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

        # Transform point cloud 'ego --> abs' homogeneous coordinates
        N = pc_velo_rgbsem.shape[0]
        pc_velo_homo = np.concatenate((pc_velo_rgbsem[:, :3], np.ones((N, 1))),
                                      axis=1)
        # Replace spatial coordinates
        pc_velo_rgbsem[:, :3] = pc_velo_homo[:, :3]

        # Filter out unwanted points according to semantics
        # TODO do this earlier to reduce computation?
        pc_velo_rgbsem = self.filter_semseg_pc(pc_velo_rgbsem)

        # Compute pose in 'absolute' coordinates
        # Pose = Project origin in ego ref. frame --> abs
        pose = np.array([[0., 0., 0., 1.]]).T
        pose = pose.T[0][:-1]  # Remove homogeneous coordinate
        pose = pose.tolist()

        self.T_prev_origin = T_new_origin
        self.pcd_prev = pcd_new

        return pc_velo_rgbsem, pose, semseg, T_new_prev