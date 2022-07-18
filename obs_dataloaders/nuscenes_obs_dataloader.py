from nuscenes.nuscenes import NuScenes
from obs_dataloaders.obs_dataloader import ObservationDataloader


class NuScenesDataloader(ObservationDataloader):
    def __init__(self, dataroot, scene_ids=None, batch_size=1, version='v1.0-mini'):
        """
        Args:
            dataroot (str):
            scene_ids (list): indicies of chosen scenes (1 scenes == 1 sequence in KITTI term)
            batch_size (int):
            version (str):
        """
        super().__init__(dataroot, batch_size)
        self.nusc = NuScenes(dataroot=dataroot, version=version, verbose=False)
        if scene_ids is None:
            scene_ids = range(self.nusc.scene)

        self.sample_tokens = []
        for scene_idx in scene_ids:
            scene = self.nusc.scene[scene_idx]
            sample_token = scene['first_sample_token']
            while sample_token is not None:
                self.sample_tokens.append(sample_token)
                # move on
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']

    def __len__(self):
        return len(self.sample_tokens)

    def read_obs(self, idx):
        sample_token = self.sample_tokens[idx]
        # TODO: obs == (cameras, pc, lidar, ego_pose @ lidar timestamp
        pass

