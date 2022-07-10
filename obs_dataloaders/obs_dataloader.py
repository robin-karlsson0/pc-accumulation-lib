from abc import ABC, abstractmethod


class ObservationDataloader(ABC):
    '''
    Abstract class for reading observations from datasets.
    '''

    def __init__(self, root_path: str, batch_size: int):
        '''
        Args:
            root_path: Path to dataset root.
            batch_size: Number of observations to output in every iteration.
        '''
        self.root_path = root_path
        self.batch_size = batch_size

    @abstractmethod
    def read_obs(self, idx):
        '''
        Implement a function that returns a single observation using 'idx'.

        Observation: Tuple with an RGB image (pil.Image) and a point cloud
                     (np.array) w. dim (#points, #feats).
        '''
        pass

    @abstractmethod
    def __len__(self):
        '''
        Implement a function that returns the total number of observations.
        '''
        pass

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        '''
        Returns a list of observations [obs_1, obs_2, ... ]
            Typically an image and point cloud pair: (pil.Image, np.array)
        '''
        if self.idx + self.batch_size <= len(self):

            obss = []
            for _ in range(self.batch_size):
                obs = self.read_obs(self.idx)
                self.idx += 1
                obss.append(obs)

            return obss
        else:
            raise StopIteration
