import numpy as np
from nuscenes.nuscenes import NuScenes
from obs_dataloaders.obs_dataloader import ObservationDataloader
from datasets.nuscenes_utils import NuScenesCamera, NuScenesLidar, homo_transform


class NuScenesDataloader(ObservationDataloader):
    def __init__(self, dataroot, scene_ids=None, batch_size=1, num_sweeps=5, version='v1.0-mini'):
        """
        Args:
            dataroot (str):
            scene_ids (list): indicies of chosen scenes (1 scenes == 1 sequence in KITTI term)
            batch_size (int):
            num_sweeps (int): number of non-keyframe pointclouds prior to a keyframe pointcloud that are
                merged to the keyframe one. Default: 5
            version (str):
        """
        super().__init__(dataroot, batch_size)
        self.nusc = NuScenes(dataroot=dataroot, version=version, verbose=False)
        self.num_sweeps = num_sweeps
        self.cam_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
                             'CAM_BACK_RIGHT']
        # self.cam_channels = ['CAM_FRONT']
        if scene_ids is None:
            scene_ids = range(self.nusc.scene)

        self.sample_tokens = []
        for scene_idx in scene_ids:
            scene = self.nusc.scene[scene_idx]
            sample_token = scene['first_sample_token']
            while sample_token != '':
                self.sample_tokens.append(sample_token)
                # move on
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']

    def __len__(self):
        return len(self.sample_tokens)

    def read_obs(self, idx):
        """

        Args:
            idx (int):

        Returns:
            obs (dict):
                images (list[PIL]):
                pc (np.ndarray): (N, 3+2[+1]) - X, Y, Z in EGO VEHICLE, pixel_u, pixel_v, [time-lag w.r.t keyframe]
                pc_cam_idx (np.ndarray): (N,) - index of camera where each point projected onto
                ego_at_lidar_ts (np.ndarray): (4, 4) ego vehicle pose w.r.t global frame @ timestamp of lidar
                meta (dict)
        """
        obs = dict()
        sample = self.nusc.get('sample', self.sample_tokens[idx])
        obs['meta'] = {'sample_token': self.sample_tokens[idx], 'scene_token': sample['scene_token'],
                       'cam_channels': self.cam_channels}

        # ##############################################
        # get pointcloud & map it to EGO VEHICLE frame
        # ##############################################
        lidar_sensor = NuScenesLidar(self.nusc, self.nusc.get('sample_data', sample['data']['LIDAR_TOP']))
        obs['ego_at_lidar_ts'] = lidar_sensor.glob_from_ego
        pc = lidar_sensor.get_pointcloud(self.nusc, sample, self.num_sweeps)  # in LiDAR frame
        pc_time = pc[:, [-1]] if pc.shape[1] > 3 else None
        pc_in_ego = homo_transform(lidar_sensor.ego_from_self, pc[:, :3])  # (N, 3)

        # #######################
        # project pc to 6 images
        # #######################
        # NOTE: for pts projected onto 2 images, their pixel coords take value of the last projection (i.e. overwritten)
        pc_in_glob = homo_transform(lidar_sensor.glob_from_ego, pc_in_ego)  # (N, 3)
        cameras = [NuScenesCamera(self.nusc, self.nusc.get('sample_data', sample['data'][channel])) for channel in
                   self.cam_channels]
        obs['images'] = [cam.img for cam in cameras]

        pc_uv, pc_cam_idx = np.zeros((pc.shape[0], 2), dtype=float), -np.ones(pc.shape[0], dtype=int)
        for j, cam in enumerate(cameras):
            pc_in_cam = homo_transform(np.linalg.inv(cam.glob_from_self), pc_in_glob)  # (N, 3) - in camera frame
            uv, mask_in_img = cam.project_pts3d(pc_in_cam)  # (N, 2), (N,)
            # save projection result
            pc_uv[mask_in_img] = uv[mask_in_img]
            pc_cam_idx[mask_in_img] = j

        obs['pc_cam_idx'] = pc_cam_idx
        if pc_time is not None:
            obs['pc'] = np.concatenate([pc_in_ego, pc_uv, pc_time], axis=1)
        else:
            obs['pc'] = np.concatenate([pc_in_ego, pc_uv], axis=1)

        return obs


