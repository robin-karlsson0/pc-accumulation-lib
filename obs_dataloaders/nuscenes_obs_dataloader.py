import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from datasets.nuscenes_utils import (NuScenesCamera, NuScenesLidar,
                                     homo_transform, inst_centric_get_sweeps,
                                     load_data_to_tensor,
                                     render_ego_centric_map)
from obs_dataloaders.obs_dataloader import ObservationDataloader


class NuScenesDataloader(ObservationDataloader):

    def __init__(self, nusc, scene_ids=None, batch_size=1, num_sweeps=5):
        """
        Args:
            nusc: Instance of NuScenes
            scene_ids (list): indicies of chosen scenes (1 scenes == 1 sequence in KITTI term)
            batch_size (int):
            num_sweeps (int): number of non-keyframe pointclouds prior to a keyframe pointcloud that are
                merged to the keyframe one. Default: 5
            version (str):
        """
        super().__init__(None, batch_size)
        self.nusc = nusc
        self.num_sweeps = num_sweeps
        self.cam_channels = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        # self.cam_channels = ['CAM_FRONT']
        if scene_ids is None:
            scene_ids = range(self.nusc.scene)

        self.sample_tokens = []
        for scene_idx in scene_ids:
            scene = self.nusc.scene[scene_idx]
            sample_token = scene['first_sample_token']
            #            # Test code for extracting map
            #            sample = self.nusc.get('sample', sample_token)
            #
            #            # TMP
            #            sd_record = self.nusc.get('sample_data',
            #                                      sample['data']['LIDAR_TOP'])
            #            sample = self.nusc.get('sample', sample_token)
            #            scene = self.nusc.get('scene', sample['scene_token'])
            #            log = self.nusc.get('log', scene['log_token'])
            #            map_ = self.nusc.get('map', log['map_token'])  # dict
            #            map_mask = map_['mask']  # nuscenes.utils.map_mask.MapMask object
            #            pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
            #            map = render_ego_centric_map(map_mask, pose)
            #
            #            from nuscenes.map_expansion.map_api import NuScenesMap
            #            nusc_map = NuScenesMap(dataroot='/home/robin/datasets/nuscenes',
            #                                   map_name='singapore-onenorth')
            #
            #            ego_center_x, ego_center_y, _ = pose['translation']
            #            patch_box = (ego_center_x, ego_center_y, 80, 80)  # [m]
            #
            #            ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
            #            yaw_deg = -math.degrees(ypr_rad[0])
            #            # Make forward direction up by rotating 90 deg
            #            # TODO: Confirm this is true
            #            yaw_deg += 45 + 90
            #
            #            patch_angle = yaw_deg  # 248  # Default orientation where North is up
            #            layer_names = ['drivable_area', 'walkway']
            #            canvas_size = (256, 256)  # [px]
            #            # fig, ax = nusc_map.render_map_mask(patch_box, patch_angle, layer_names, canvas_size, figsize=(12,4), n_row=1)
            #            layers = nusc_map.get_map_mask(patch_box,
            #                                           patch_angle,
            #                                           layer_names,
            #                                           canvas_size=(256, 256))
            #
            #            map = np.zeros(canvas_size)
            #            for layer in layers:
            #                map[layer == 1] = np.max(map) + 1
            #
            #            import matplotlib.pyplot as plt
            #            plt.imshow(map)
            #            plt.gca().invert_yaxis()
            #            plt.show()

            while sample_token != '':
                self.sample_tokens.append(sample_token)
                # move on
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']

        # PC matrix column indices
        self.int_idx = 3
        self.sweep_idx = 5
        self.inst_idx = 6
        self.cls_idx = 7

        # Point cloud range to extract (use large enough value)
        VAL = 1000
        self.pc_range = [-VAL, -VAL, -VAL, VAL, VAL, VAL]

    def __len__(self):
        return len(self.sample_tokens)

    def read_obs(self, idx):
        """
        Args:
            idx (int):
        Returns:
            obs (dict):
                images (list[PIL]):
                pc (np.ndarray): (N, 7)
                    [0] --> [2]: (x, y, z) in ego vehicle frame
                    [3]        : Intensity
                    [4] --> [5]: Pixel (u, v) coordinates
                    [6]        : Object instance idx
                pc_cam_idx (np.ndarray): (N,) index of camera where each point
                                         is projected onto
                ego_at_lidar_ts (np.ndarray): (4, 4) ego vehicle pose w.r.t
                                              global frame @ timestamp of lidar
                meta (dict):
                inst_tokens (list[str]): Object identifier across samples
                inst_cls (list[int]): Object instance class (0: car, etc.)
                inst_center (list[np.array]): Object (x,y,z) in global frame
        """
        sample_token = self.sample_tokens[idx]
        obs = dict()
        sample = self.nusc.get('sample', sample_token)
        obs['meta'] = {
            'sample_token': self.sample_tokens[idx],
            'scene_token': sample['scene_token'],
            'cam_channels': self.cam_channels
        }

        #################################################
        # Get pointcloud & map it to EGO VEHICLE frame
        #################################################
        map_point_feat2idx = {
            'sweep_idx': self.sweep_idx,
            'inst_idx': self.inst_idx,
            'cls_idx': self.cls_idx,
        }
        cfg = {
            'n_sweeps':
            self.num_sweeps,
            'center_radius':
            2.0,
            'in_box_tolerance':
            5e-2,
            'return_instances_last_box':
            True,
            'point_cloud_range':
            self.pc_range,
            'detection_classes':
            ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
             'motorcycle', 'bicycle', 'pedestrian'),  # Movable classes
            'map_point_feat2idx':
            map_point_feat2idx
        }
        out = inst_centric_get_sweeps(self.nusc, sample_token, **cfg)
        load_data_to_tensor(out)
        pc = out['points']  # In lidar frame

        lidar_sensor = NuScenesLidar(
            self.nusc, self.nusc.get('sample_data',
                                     sample['data']['LIDAR_TOP']))
        obs['ego_at_lidar_ts'] = lidar_sensor.glob_from_ego
        pc_in_ego = homo_transform(lidar_sensor.ego_from_self,
                                   pc[:, :3])  # (N, 3)

        # Intensity
        pc_int = pc[:, self.int_idx:self.int_idx + 1]

        # Instance idx
        pc_inst = pc[:, self.inst_idx:self.inst_idx + 1]

        ############################
        #  Project pc to 6 images
        ############################
        # NOTE: for pts projected onto 2 images, their pixel coords take value
        # of the last projection (i.e. overwritten)
        pc_in_glob = homo_transform(lidar_sensor.glob_from_ego,
                                    pc_in_ego)  # (N, 3)
        cameras = [
            NuScenesCamera(
                self.nusc, self.nusc.get('sample_data',
                                         sample['data'][channel]))
            for channel in self.cam_channels
        ]
        obs['images'] = [cam.img for cam in cameras]

        pc_uv, pc_cam_idx = np.zeros(
            (pc.shape[0], 2), dtype=float), -np.ones(pc.shape[0], dtype=int)
        for j, cam in enumerate(cameras):
            pc_in_cam = homo_transform(np.linalg.inv(cam.glob_from_self),
                                       pc_in_glob)  # (N, 3) - in camera frame
            uv, mask_in_img = cam.project_pts3d(pc_in_cam)  # (N, 2), (N,)
            # save projection result
            pc_uv[mask_in_img] = uv[mask_in_img]
            pc_cam_idx[mask_in_img] = j

        obs['pc_cam_idx'] = pc_cam_idx

        obs['pc'] = np.concatenate([pc_in_ego, pc_int, pc_uv, pc_inst], axis=1)

        #######################
        #  GT bounding boxes
        #######################
        obs['inst_tokens'] = out['instances_token']
        obs['inst_cls'] = [int(cls.item()) for cls in out['instances_name']]
        obs['inst_center'] = out['instances_center']

        return obs
