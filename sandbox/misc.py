import open3d as o3d
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from copy import deepcopy
import numpy as np
from PIL import Image
import os.path as osp


def show_pointcloud(xyz, boxes=None, pc_colors=None):
    """

    Args:
        xyz (np.ndarray): (N, 3)
        boxes (list): list of boxes, each box is denoted by coordinates of its 8 vertices - np.ndarray (8, 3)
        pc_colors (np.ndarray): (N, 3) - r, g, b
    """
    def create_cube(vers):
        # vers: (8, 3)
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # front
            [4, 5], [5, 6], [6, 7], [7, 4],  # back
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting front & back
            [0, 2], [1, 3]  # denote forward face
        ]
        colors = [[1, 0, 0] for i in range(len(lines))]  # red
        cube = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vers),
            lines=o3d.utility.Vector2iVector(lines),
        )
        cube.colors = o3d.utility.Vector3dVector(colors)
        return cube

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if pc_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    if boxes is not None:
        o3d_cubes = [create_cube(box) for box in boxes]
        o3d.visualization.draw_geometries([pcd, *o3d_cubes])
    else:
        o3d.visualization.draw_geometries([pcd])


class NuScenesAnnotation:
    def __init__(self, translation, size, rotation, category_name, **kwargs):
        self.pose = transform_matrix(translation, Quaternion(rotation))  # @ init, == global_from_box | 4x4
        self.__global_pose = deepcopy(self.pose)
        self.category_name = category_name
        self.wlh = size  # length == x-axis

    def transform(self, target_from_global):
        self.pose = target_from_global @ self.pose  # == target_from_box

    def __gen_vertices_in_local(self):
        # generate 8 vertices in box's local frame
        # origin @ box's center
        # x-axis == box' heading dir
        # z-axis == vertical, up
        w, l, h = self.wlh
        xs = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float) * l/2.0
        ys = np.array([-1, 1, 1, -1, -1, 1, 1, -1], dtype=float) * w/2.0
        zs = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float) * h/2.0
        vers = np.concatenate([xs.reshape(-1, 1), ys.reshape(-1, 1), zs.reshape(-1, 1)], axis=1)
        return vers

    def gen_vertices(self):
        vers = self.__gen_vertices_in_local()
        vers = self.pose @ np.concatenate([vers, np.ones((vers.shape[0], 1))], axis=1).T
        vers = vers[:3].T  # (8, 3)
        return vers

    def find_points_inside(self, pc):
        """
        Find points inside this box
        Args:
            pc (np.ndarray): (N, 3) - X, Y, Z, in the SAME reference frame as the box

        Returns:
            mask_inside (np.ndarray): (N,) - bool, True if points is inside
        """
        box_from_ref = np.linalg.inv(self.pose)
        pc_in_box = box_from_ref @ np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1).T
        w, l, h = self.wlh
        pc_in_box = pc_in_box[:3, :].T / np.array([[l, w, h]], dtype=float)  # (N, 3)
        mask_inside = np.all(np.abs(pc_in_box) < (0.5 + 1e-3), axis=1)
        return mask_inside
