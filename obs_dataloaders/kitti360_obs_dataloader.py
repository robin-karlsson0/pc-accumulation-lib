import os

import PIL.Image as Image
from datasets.kitti360_utils import read_pc_bin_file

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

        for seq_idx in range(len(sequences)):

            seq_str = sequences[seq_idx]
            pc_dir = os.path.join('data_3d_raw', seq_str, 'velodyne_points',
                                  'data')
            img_dir = os.path.join('data_2d_raw', seq_str, 'image_00',
                                   'data_rect')

            seq_start_idx = start_idxs[seq_idx]
            seq_end_idx = end_idxs[seq_idx]

            for idx in range(seq_start_idx, seq_end_idx):

                idx_str = self.idx2str(idx)
                pc_path = os.path.join(pc_dir, idx_str + '.bin')
                self.pc_paths.append(pc_path)

                img_path = os.path.join(img_dir, idx_str + '.png')
                self.img_paths.append(img_path)

    def __len__(self):
        return len(self.pc_paths)

    def read_obs(self, idx):
        '''
        '''
        pc_path = os.path.join(self.root_path, self.pc_paths[idx])
        pc = read_pc_bin_file(pc_path)

        img_path = os.path.join(self.root_path, self.img_paths[idx])
        img = Image.open(img_path)

        obs = (img, pc)
        return obs

    @staticmethod
    def idx2str(idx: int):
        '''
        Convert frame idx integer to string with leading zeros
        '''
        return f"{idx:010d}"


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
