import matplotlib as mpl

mpl.use('agg')  # Must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from .bev_generator import BEVGenerator


class SemBEVGenerator(BEVGenerator):
    '''
    '''

    def __init__(self,
                 sem_idxs: dict,
                 view_size: int,
                 pixel_size: int,
                 max_trans_radius: float = 0.,
                 zoom_thresh: float = 0.,
                 do_warp: bool = False):
        '''
        Args:
            sem_layers: ['road', 'intensity', 'elevation'] etc.
        '''
        super().__init__(view_size, pixel_size, max_trans_radius, zoom_thresh,
                         do_warp)

        # Dictionary with semantic --> index mapping
        self.sem_idxs = sem_idxs

    def generate_bev(self, pc_present: np.array, pc_future: np.array,
                     pc_full: np.array, poses_present: np.array,
                     poses_future: np.array, poses_full: np.array):
        '''
        Args:
            pc_present: Semantic point cloud matrix w. dim (N, 8)
                        [x, y, z, i, r, g, b, sem]
            pc_future:
            poses_present: Pose matrix w. dim (N, 3) [x, y, z]
            poses_future:
        '''
        dynamic_filter = [
            self.sem_idxs['car'],
            self.sem_idxs['truck'],
            self.sem_idxs['bus'],
            self.sem_idxs['motorcycle'],
        ]
        pc_present_dynamic, pc_present_static = self.partition_semantic_pc(
            pc_present, dynamic_filter)

        probmap_present_road = self.gen_sem_probmap(pc_present_static, 'road')

        intmap_present_road = self.gen_intensity_map(pc_present_static, 'road')

        if pc_future is not None:
            pc_future_dynamic, pc_future_static = self.partition_semantic_pc(
                pc_future, dynamic_filter)

            probmap_future_road = self.gen_sem_probmap(pc_future_static,
                                                       'road')
            intmap_future_road = self.gen_intensity_map(
                pc_future_static, 'road')

            pc_full_dynamic, pc_full_static = self.partition_semantic_pc(
                pc_full, dynamic_filter)

            probmap_full_road = self.gen_sem_probmap(pc_full_static, 'road')
            intmap_full_road = self.gen_intensity_map(pc_full_static, 'road')

        # Warp all probability maps and poses
        if self.do_warp:
            i_mid = int(self.pixel_size / 2)
            j_mid = i_mid
            # I_crop, J_crop = pixel_size
            i_warp, j_warp = self.get_random_warp_params(
                0.15, 0.30, self.pixel_size, self.pixel_size)
            a_1, a_2 = self.cal_warp_params(i_warp, i_mid, self.pixel_size - 1)
            b_1, b_2 = self.cal_warp_params(j_warp, j_mid, self.pixel_size - 1)

            maps = [probmap_present_road, intmap_present_road]
            if pc_future is not None:
                maps.append(probmap_future_road)
                maps.append(probmap_full_road)
                maps.append(intmap_future_road)
                maps.append(intmap_full_road)

            maps = np.stack(maps)
            maps = self.warp_dense_probmaps(maps, a_1, a_2, b_1, b_2)
            poses_present = self.warp_sparse_points(poses_present, a_1, a_2,
                                                    b_1, b_2, i_mid, j_mid,
                                                    i_warp, j_warp)
            probmap_present_road = maps[0]
            intmap_present_road = maps[1]

            if pc_future is not None:
                probmap_future_road = maps[2]
                poses_future = self.warp_sparse_points(poses_future, a_1, a_2,
                                                       b_1, b_2, i_mid, j_mid,
                                                       i_warp, j_warp)
                probmap_full_road = maps[3]
                poses_full = self.warp_sparse_points(poses_full, a_1, a_2, b_1,
                                                     b_2, i_mid, j_mid, i_warp,
                                                     j_warp)
                intmap_future_road = maps[4]
                intmap_full_road = maps[5]

        # Reduce storage size
        probmap_present_road = probmap_present_road.astype(np.float16)
        bev = {
            'road_present': probmap_present_road,
            'poses_present': poses_present,
            'intensity_present': intmap_present_road,
        }

        if pc_future is not None:
            probmap_future_road = probmap_future_road.astype(np.float16)
            probmap_full_road = probmap_full_road.astype(np.float16)
            bev.update({
                'road_future': probmap_future_road,
                'poses_future': poses_future,
                'road_full': probmap_full_road,
                'poses_full': poses_full,
                'intensity_future': intmap_future_road,
                'intensity_full': intmap_full_road,
            })

        return bev

    def viz_bev(self, bev, file_path, rgbs=[], semsegs=[]):
        '''
        '''
        present_road = bev['road_present']
        poses_present = bev['poses_present']
        present_intensity = bev['intensity_present']

        H = self.pixel_size

        num_imgs = len(rgbs)
        num_cols = num_imgs if num_imgs > 3 else 3
        num_rows = 3 if num_imgs > 0 else 2

        if 'road_future' in bev.keys():
            future_road = bev['road_future']
            poses_future = bev['poses_future']
            full_road = bev['road_full']
            poses_full = bev['poses_full']
            future_intensity = bev['intensity_future']
            full_intensity = bev['intensity_full']

            size_per_fig = 6
            _ = plt.figure(figsize=(size_per_fig * num_cols,
                                    size_per_fig * num_rows))

            # Road semantic
            plt.subplot(num_rows, num_cols, 1)
            plt.imshow(present_road, vmin=0, vmax=1)
            plt.plot(poses_present[:, 0], H - poses_present[:, 1], 'k-')

            plt.subplot(num_rows, num_cols, 2)
            plt.imshow(future_road, vmin=0, vmax=1)
            plt.plot(poses_future[:, 0], H - poses_future[:, 1], 'r-')

            plt.subplot(num_rows, num_cols, 3)
            plt.imshow(full_road, vmin=0, vmax=1)
            plt.plot(poses_full[:, 0], H - poses_full[:, 1], 'b-')

            # Intensity
            plt.subplot(num_rows, num_cols, num_cols + 1)
            plt.imshow(present_intensity, vmin=0, vmax=1)

            plt.subplot(num_rows, num_cols, num_cols + 2)
            plt.imshow(future_intensity, vmin=0, vmax=1)

            plt.subplot(num_rows, num_cols, num_cols + 3)
            plt.imshow(full_intensity, vmin=0, vmax=1)

            if num_imgs > 0:
                for idx in range(num_imgs):
                    plt.subplot(num_rows, num_cols, 2 * num_cols + idx + 1)
                    plt.imshow(rgbs[idx])
                    semseg = semsegs[idx]
                    if semseg is not None:
                        plt.imshow(semsegs[idx] == 0,
                                   alpha=0.5,
                                   vmin=0,
                                   vmax=1)

        else:

            _ = plt.figure(figsize=(6, 6))

            plt.imshow(present_road, vmin=0, vmax=1)
            plt.plot(poses_present[:, 0], H - poses_present[:, 1], 'k-')

        plt.tight_layout()

        plt.savefig(file_path)
        plt.clf()
        plt.close()
