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


nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/', version='v1.0-mini', verbose=False)
scene = nusc.scene[0]
# nusc.render_scene(scene['token'])
sample = nusc.sample[10]
print_dict(sample)
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
    if 'vehicle' in ann['category_name']:
        anns.append(NuScenesAnnotation(**ann))
        print_dict(ann)
        print('----\n\n')
# ---
# map anns & pc to flat_vehicle_coord (z-axis == z-axis of the global frame)
# ---

# NOTE: timestamp of sample (i.e. keyframe) == timestamp of sample annotations == timestamp of sample_data (cuz synced)
# NOTE: sample_data's timestamp is used to access ego vehicle pose

# vehicle flat <-- vehicle <-- lidar
# transformation from lidar to ego vehicle
lidar_calib_record = nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
vehicle_from_lidar = transform_matrix(lidar_calib_record['translation'], Quaternion(lidar_calib_record['rotation']))

# compute rotation from 3D vehicle pose to "flat" vehicle pose (parallel to global z plane).
pose_record = nusc.get('ego_pose', lidar_record['ego_pose_token'])
global_from_vehicle = transform_matrix(pose_record['translation'],
                                       Quaternion(pose_record['rotation']))  # @ timestamp of LiDAR
ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
rotation_vehicle_flat_from_vehicle = np.dot(
    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
vehicle_flat_from_vehicle = np.eye(4)
vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle

vehicle_flat_from_lidar = vehicle_flat_from_vehicle @ vehicle_from_lidar
pc_in_vf = vehicle_flat_from_lidar @ np.concatenate([pc, np.ones((pc.shape[0], 1))], axis=1).T
pc_in_vf = pc_in_vf[:3, :].T  # (N, 3) - X, Y, Z in vehicle flat coord

# vehicle flat <-- vehicle <-- global <-- box's local frame
vehicle_from_global = np.linalg.inv(global_from_vehicle)
vehicle_flat_from_global = vehicle_flat_from_vehicle @ vehicle_from_global
_boxes = []
for anno in anns:
    anno.transform(vehicle_flat_from_global)
    _boxes.append(anno.gen_vertices())

show_pointcloud(pc_in_vf, _boxes)


# TODO: project Bbox & pts inside onto img



