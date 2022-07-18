from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os.path as osp
import open3d as o3d
from utils.nuscenes_utils import print_dict
from pyquaternion import Quaternion

from misc import show_pointcloud, NuScenesAnnotation
from utils.nuscenes_utils import NuScenesCamera


nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/', version='v1.0-mini', verbose=False)
scene = nusc.scene[0]
# nusc.render_scene(scene['token'])
sample = nusc.sample[10]
# print_dict(sample)
# nusc.render_pointcloud_in_image(sample['token'], pointsensor_channel='LIDAR_TOP')

# ######################
# Display point cloud
# ######################
lidar_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
pcl_path = osp.join(nusc.dataroot, lidar_record['filename'])
pc_ = LidarPointCloud.from_file(pcl_path)  # pc is in LIDAR_TOP frame
pc = pc_.points[:3, :].T  # (N, 3) - X, Y, Z in LIDAR_TOP frame

# #################################
# Display a Bbox & pts inside
# #################################
anns = []
for i, ann_token in enumerate(sample['anns']):
    ann = nusc.get('sample_annotation', ann_token)  # ann's pose is GLOBAL coord sys
    if 'vehicle.car' in ann['category_name'] and ann['num_lidar_pts'] > 40:
        anns.append(NuScenesAnnotation(**ann))
        print_dict(ann)
        print('----\n\n')
# ---
# map anns & pc to flat_vehicle_coord (z-axis == z-axis of the global frame)
# ---

# NOTE: timestamp of sample (i.e. keyframe) == timestamp of sample annotations == timestamp of sample_data (cuz synced)
# NOTE: sample_data's timestamp is used to access ego vehicle pose

# ego vehicle <-- lidar
# transformation from lidar to ego vehicle
lidar_calib_record = nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
ego_from_lidar = transform_matrix(lidar_calib_record['translation'], Quaternion(lidar_calib_record['rotation']))

# # compute rotation from 3D vehicle pose to "flat" vehicle pose (parallel to global z plane).
# ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
# rotation_vehicle_flat_from_vehicle = np.dot(
#     Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
#     Quaternion(pose_record['rotation']).inverse.rotation_matrix)
# vehicle_flat_from_vehicle = np.eye(4)
# vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle

pc_in_ego = ego_from_lidar @ np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1).T
pc_in_ego = pc_in_ego[:3, :].T  # (N, 3) - X, Y, Z in vehicle flat coord

# ego vehicle <-- global <-- box's local frame
pose_record = nusc.get('ego_pose', lidar_record['ego_pose_token'])
global_from_ego = transform_matrix(pose_record['translation'],
                                   Quaternion(pose_record['rotation']))  # @ timestamp of LiDAR
ego_from_global = np.linalg.inv(global_from_ego)

_boxes = []
mask_in_boxes = np.zeros(pc_in_ego.shape[0], dtype=bool)
for anno in anns:
    anno.transform(ego_from_global)
    mask_in_box = anno.find_points_inside(pc_in_ego)
    mask_in_boxes = np.logical_or(mask_in_boxes, mask_in_box)
    _boxes.append(anno.gen_vertices())
pc_colors = np.zeros((pc.shape[0], 3), dtype=float)
pc_colors[mask_in_boxes, 2] = 1  # blue for points inside boxes

show_pointcloud(pc_in_ego, _boxes, pc_colors)

# ####################################
# project Bbox & pts inside onto img
# ####################################
cam_record = nusc.get('sample_data', sample['data']['CAM_FRONT'])
cam = NuScenesCamera(nusc, cam_record)
# ---
# map pc to camera frame
# ---
pc_in_glob = global_from_ego @ np.concatenate([pc_in_ego, np.ones((pc_in_ego.shape[0], 1))], axis=1).T
pc_in_cam = np.linalg.inv(cam.glob_from_self) @ pc_in_glob
pc_in_cam = pc_in_cam[:3, :].T
pc_pixel_uv, mask_in_img = cam.project_pts3d(pc_in_cam)
pc_feat = np.zeros((pc.shape[0], 3), dtype=float)
pc_feat[mask_in_img] = cam.get_pts_feat(pc_pixel_uv[mask_in_img], method='nearest') / 255.
show_pointcloud(pc_in_ego, _boxes, pc_feat)

mask_display = mask_in_img & mask_in_boxes
# display only n pts
num_display = 5
_inds = np.arange(mask_display.shape[0])
_inds = _inds[mask_display]
_inds = _inds[:num_display]  # to display only 10 pts
final_mask_display = np.zeros(mask_display.shape[0], dtype=bool)
final_mask_display[_inds] = True

print(pc_feat[final_mask_display])
disp_x, disp_y = pc_pixel_uv[final_mask_display, 0], pc_pixel_uv[final_mask_display, 1]
fig, ax = plt.subplots()
ax.imshow(cam.img)
ax.scatter(disp_x, disp_y)
for i in range(num_display):
    ax.annotate(str(i), (disp_x[i], disp_y[i]))
plt.show()


