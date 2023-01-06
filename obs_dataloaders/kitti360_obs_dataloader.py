import os

import numpy as np
import PIL.Image as Image
from datasets.kitti360_utils import (conv_semantic_ids, read_pc_bin_file,
                                     read_sem_gt_bin_file)

from obs_dataloaders.obs_dataloader import ObservationDataloader


class Kitti360Dataloader(ObservationDataloader):
    '''
    Class for reading observations from the KITTI 360 dataset.

    Generates a list of observations [(RGB image, pointcloud), ... ] when used
    in a for loop:

        dataloader = Kitti360Dataloader(...)
        for observations in dataloader:
            --> [(RGB image, pointcloud), ... ]

    '''

    def __init__(self, root_path: str, batch_size: int, sequences: list,
                 start_idxs: list, end_idxs: list):
        '''
        All image and pointcloud data for specified sequences is stored as file
        paths for simple access by index.

        Each sequence beginning and end is cut out according to indices
        specified in 'start_idxs' and 'end_idxs'.

        Example:
            ------------------------------------------------------
            |        Sequence string       | Start idx | End idx |
            ------------------------------------------------------
            | "2013_05_28_drive_0000_sync" | 130       | 11400   |
            | "2013_05_28_drive_0002_sync" | 4613      | 18997   |
            | "2013_05_28_drive_0003_sync" | 40        | 770     |
            |                           ...                      |
            ------------------------------------------------------

        Args:
            root_path: Path to dataset root.
            batch_size: Number of observations to output in every iteration.
            sequences: List of strings representing sequences to use.
            start_idxs: List of sequence starting indices.
            end_idxs: List of sequence ending indices.
        '''
        super().__init__(root_path, batch_size)

        # Store file paths in list for access by index
        self.pc_paths = []
        self.img_paths = []
        self.sem_gt_paths = []

        for seq_idx in range(len(sequences)):

            seq_str = sequences[seq_idx]
            pc_dir = os.path.join('data_3d_raw', seq_str, 'velodyne_points',
                                  'data')
            img_dir = os.path.join('data_2d_raw', seq_str, 'image_00',
                                   'data_rect')
            sem_gt_dir = os.path.join('data_3d_semantics', 'raw', seq_str,
                                      'labels')

            seq_start_idx = start_idxs[seq_idx]
            seq_end_idx = end_idxs[seq_idx]

            for idx in range(seq_start_idx, seq_end_idx):
                idx_str = self.idx2str(idx)

                pc_path = os.path.join(pc_dir, idx_str + '.bin')
                self.pc_paths.append(pc_path)

                img_path = os.path.join(img_dir, idx_str + '.png')
                self.img_paths.append(img_path)

                sem_gt_path = os.path.join(sem_gt_dir, idx_str + '.bin')
                self.sem_gt_paths.append(sem_gt_path)

        self.idx2idx = self.gen_idx_mapping()

    def __len__(self):
        return len(self.pc_paths)

    def read_obs(self, idx):
        '''
        '''
        pc_path = os.path.join(self.root_path, self.pc_paths[idx])
        pc = read_pc_bin_file(pc_path)

        img_path = os.path.join(self.root_path, self.img_paths[idx])
        img = Image.open(img_path)

        sem_gt_path = os.path.join(self.root_path, self.sem_gt_paths[idx])
        sem_gt = read_sem_gt_bin_file(sem_gt_path)
        if sem_gt is None:
            print(f"Missing GT sem: {sem_gt_path}")
            N = pc.shape[0]
            sem_gt = np.zeros((N, 1))

        sem_gt = conv_semantic_ids(sem_gt, self.idx2idx)

        obs = (img, pc, sem_gt)
        return obs

    @staticmethod
    def idx2str(idx: int):
        '''
        Convert frame idx integer to string with leading zeros
        '''
        return f"{idx:010d}"

    @staticmethod
    def gen_idx_mapping():
        '''
        Transforms 'id' --> 'trainId'
            idx2idx[id] --> trainId

        Ref: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
        '''
        idx2idx = {
            0: 2,  # unlabeled
            1: 255,  # ego vehicle
            2: 255,  # rectification border
            3: 255,  # out of roi
            4: 2,  # static
            5: 2,  # dynamic
            6: 9,  # ground
            7: 0,  # road
            8: 1,  # sidewalk
            9: 9,  # parking
            10: 9,  # rail track
            11: 2,  # building
            12: 3,  # wall
            13: 4,  # fence
            14: 2,  # guard rail
            15: 2,  # bridge
            16: 2,  # tunnel
            17: 5,  # pole
            18: 5,  # polegroup
            19: 6,  # traffic light
            20: 7,  # traffic sign
            21: 8,  # vegetation
            22: 9,  # terrain
            23: 10,  # sky
            24: 11,  # person
            25: 12,  # rider
            26: 13,  # car
            27: 14,  # truck
            28: 15,  # bus
            29: 14,  # caravan
            30: 14,  # trailer
            31: 16,  # train
            32: 17,  # motorcycle
            33: 18,  # bicycle
            34: 2,  # garage
            35: 4,  # gate
            36: 2,  # stop
            37: 5,  # smallpole
            38: 5,  # lamp
            39: 2,  # trash bin
            40: 2,  # vending machine
            41: 2,  # box
            42: 2,  # unknown constructio
            43: 13,  # unknown vehicle
            44: 2,  # unknown object
            -1: 13,  # license plat
        }
        return idx2idx


if __name__ == '__main__':

    kitti360_path = '/home/robin/datasets/KITTI-360'
    batch_size = 5
    sequences = [
        '2013_05_28_drive_0000_sync',
        '2013_05_28_drive_0002_sync',
        '2013_05_28_drive_0003_sync',
        '2013_05_28_drive_0004_sync',
        '2013_05_28_drive_0005_sync',
        '2013_05_28_drive_0006_sync',
        '2013_05_28_drive_0007_sync',
        '2013_05_28_drive_0009_sync',
        '2013_05_28_drive_0010_sync',
    ]
    start_idxs = [130, 4613, 40, 90, 50, 120, 0, 90, 0]
    end_idxs = [11400, 1899, 770, 11530, 6660, 9698, 2960, 13945, 3540]

    dataloader = Kitti360Dataloader(kitti360_path, batch_size, sequences,
                                    start_idxs, end_idxs)

    for obss in dataloader:

        print(obss)
