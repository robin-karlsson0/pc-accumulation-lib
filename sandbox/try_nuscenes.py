from nuscenes.nuscenes import NuScenes
import numpy as np
import matplotlib.pyplot as plt

from misc import show_pointcloud, NuScenesAnnotation
from utils.nuscenes_utils import NuScenesCamera, NuScenesLidar, homo_transform


nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/', version='v1.0-mini', verbose=False)
scene = nusc.scene[0]
# nusc.render_scene(scene['token'])
sample = nusc.sample[10]

# ##########################################################
# Get point cloud in LiDAR frame & map it ego vehicle frame
# ##########################################################
lidar_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
lidar = NuScenesLidar(nusc, lidar_record)
pc = lidar.get_pointcloud(nusc, sample, num_sweeps=5)  # in LIDAR frame
if pc.shape[1] > 3:
    pc = pc[:, :3]

pc_in_ego = homo_transform(lidar.ego_from_self, pc)

# #################################
# Display Annotations & pts inside
# #################################
anns = []
for i, ann_token in enumerate(sample['anns']):
    ann = nusc.get('sample_annotation', ann_token)  # ann's pose is GLOBAL coord sys
    if 'vehicle.car' in ann['category_name'] and ann['num_lidar_pts'] > 40:
        anns.append(NuScenesAnnotation(**ann))
# ---
# map anns to ego vehicle frame @ timestamp of LiDAR
# ---
# ego vehicle <-- global <-- box's local frame
glob_from_ego = lidar.glob_from_ego
ego_from_glob = np.linalg.inv(glob_from_ego)

_boxes = []
mask_in_boxes = np.zeros(pc_in_ego.shape[0], dtype=bool)
for anno in anns:
    anno.transform(ego_from_glob)
    mask_in_box = anno.find_points_inside(pc_in_ego)
    mask_in_boxes = np.logical_or(mask_in_boxes, mask_in_box)
    _boxes.append(anno.gen_vertices())
pc_colors = np.zeros((pc.shape[0], 3), dtype=float)
pc_colors[mask_in_boxes, 2] = 1  # blue for points inside boxes

# show_pointcloud(pc_in_ego, _boxes, pc_colors)

# ####################################
# project pc to 6 images
# ####################################
cam_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
cameras = [NuScenesCamera(nusc, nusc.get('sample_data', sample['data'][channel])) for channel in cam_channels]
# ---
# get pc's feat from camera image
# ---
pc_in_glob = glob_from_ego @ np.concatenate([pc_in_ego, np.ones((pc_in_ego.shape[0], 1))], axis=1).T  # (4, N)
pc_feat = np.zeros((pc.shape[0], 3), dtype=float)
pc_in_img_counter = np.zeros(pc.shape[0], dtype=float)  # to count number of images a point projected onto (<=2)
for camera in cameras:
    pc_in_cam = np.linalg.inv(camera.glob_from_self) @ pc_in_glob
    pc_in_cam = pc_in_cam[:3, :].T
    pc_pixel_uv, mask_in_img = camera.project_pts3d(pc_in_cam)
    # compute point feat from image
    pc_in_img_counter[mask_in_img] = pc_in_img_counter[mask_in_img] + 1.
    pc_feat[mask_in_img] = pc_feat[mask_in_img] + camera.get_pts_feat(pc_pixel_uv[mask_in_img], method='nearest') / 255.

# average pc_feat by number of image a point projected onto
assert np.all(pc_in_img_counter < 3)
mask_in_img = pc_in_img_counter > 0
pc_feat[mask_in_img] = pc_feat[mask_in_img] / pc_in_img_counter[mask_in_img, np.newaxis]

show_pointcloud(pc_in_ego, _boxes, pc_feat)

# mask_display = mask_in_img & mask_in_boxes
# # display only n pts
# num_display = None
# if num_display:
#     _inds = np.arange(mask_display.shape[0])
#     _inds = _inds[mask_display]
#     _inds = _inds[:num_display]  # to display only 10 pts
#     final_mask_display = np.zeros(mask_display.shape[0], dtype=bool)
#     final_mask_display[_inds] = True
#     print(pc_feat[final_mask_display])
# else:
#     final_mask_display = mask_display
#
# disp_x, disp_y = pc_pixel_uv[final_mask_display, 0], pc_pixel_uv[final_mask_display, 1]
# fig, ax = plt.subplots()
# ax.imshow(cam.img)
# ax.scatter(disp_x, disp_y)
# if num_display:
#     for i in range(num_display):
#         ax.annotate(str(i), (disp_x[i], disp_y[i]))
# plt.show()

