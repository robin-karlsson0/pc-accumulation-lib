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
                 do_warp: bool = False,
                 int_scaler: float = 1.,
                 int_sep_scaler: float = 1.,
                 int_mid_threshold: float = 0.5,
                 rgb_fill: int = 0):
        '''
        Args:
            sem_layers: ['road', 'intensity', 'elevation'] etc.
        '''
        super().__init__(view_size, pixel_size, max_trans_radius, zoom_thresh,
                         do_warp, int_scaler, int_sep_scaler,
                         int_mid_threshold)

        # Dictionary with semantic --> index mapping
        self.sem_idxs = sem_idxs

        self.rgb_fill = rgb_fill

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

        # RGB
        r_present, g_present, b_present = self.get_rgb_maps(pc_present)
        r_future, g_future, b_future = self.get_rgb_maps(pc_future)
        r_full, g_full, b_full = self.get_rgb_maps(pc_full)
        r_present /= 255.
        g_present /= 255.
        b_present /= 255.
        r_future /= 255.
        g_future /= 255.
        b_future /= 255.
        r_full /= 255.
        g_full /= 255.
        b_full /= 255.

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

            pc_full_dynamic, pc_full_static = self.partition_semantic_pc(
                pc_full, dynamic_filter)

            probmap_full_road = self.gen_sem_probmap(pc_full_static, 'road')

        # Warp all probability maps and poses
        if self.do_warp:
            i_mid = int(self.pixel_size / 2)
            j_mid = i_mid
            # I_crop, J_crop = pixel_size
            i_warp, j_warp = self.get_random_warp_params(
                0.15, 0.30, self.pixel_size, self.pixel_size)
            a_1, a_2 = self.cal_warp_params(i_warp, i_mid, self.pixel_size - 1)
            b_1, b_2 = self.cal_warp_params(j_warp, j_mid, self.pixel_size - 1)

            maps = [
                probmap_present_road,
                intmap_present_road,
                r_present,
                g_present,
                b_present,
            ]
            if pc_future is not None:
                maps.append(probmap_future_road)
                maps.append(probmap_full_road)
                maps.append(intmap_future_road)
                maps.append(intmap_full_road)
                maps.append(r_future)
                maps.append(g_future)
                maps.append(b_future)
                maps.append(r_full)
                maps.append(g_full)
                maps.append(b_full)

            maps = np.stack(maps)
            maps = self.warp_dense_probmaps(maps, a_1, a_2, b_1, b_2)
            poses_present = self.warp_sparse_points(poses_present, a_1, a_2,
                                                    b_1, b_2, i_mid, j_mid,
                                                    i_warp, j_warp)
            probmap_present_road = maps[0]
            intmap_present_road = maps[1]
            r_present = maps[2]
            g_present = maps[3]
            b_present = maps[4]

            if pc_future is not None:
                probmap_future_road = maps[5]
                poses_future = self.warp_sparse_points(poses_future, a_1, a_2,
                                                       b_1, b_2, i_mid, j_mid,
                                                       i_warp, j_warp)
                probmap_full_road = maps[6]
                poses_full = self.warp_sparse_points(poses_full, a_1, a_2, b_1,
                                                     b_2, i_mid, j_mid, i_warp,
                                                     j_warp)
                intmap_future_road = maps[7]
                intmap_full_road = maps[8]

                r_future = maps[9]
                g_future = maps[10]
                b_future = maps[11]
                r_full = maps[12]
                g_full = maps[13]
                b_full = maps[14]

        # Transform intensity map to more discriminative range
        intmap_present_road = self.road_marking_transform(
            intmap_present_road, self.int_scaler, self.int_sep_scaler,
            self.int_mid_threshold)

        rgb_present = np.stack((r_present, g_present, b_present))

        # Reduce storage size
        probmap_present_road = probmap_present_road.astype(np.float16)
        rgb_present = rgb_present.astype(np.float16)
        intmap_present_road = intmap_present_road.astype(np.float16)
        bev = {
            'road_present': probmap_present_road,
            'poses_present': poses_present,
            'intensity_present': intmap_present_road,
            'rgb_present': rgb_present,
        }

        if pc_future is not None:
            # Transform intensity map to more discriminative range
            intmap_future_road = self.road_marking_transform(
                intmap_future_road, self.int_scaler, self.int_sep_scaler,
                self.int_mid_threshold)
            intmap_full_road = self.road_marking_transform(
                intmap_full_road, self.int_scaler, self.int_sep_scaler,
                self.int_mid_threshold)

            rgb_future = np.stack((r_future, g_future, b_future))
            rgb_full = np.stack((r_future, g_future, b_future))

            # Reduce storage size
            probmap_future_road = probmap_future_road.astype(np.float16)
            probmap_full_road = probmap_full_road.astype(np.float16)
            intmap_future_road = intmap_future_road.astype(np.float16)
            intmap_full_road = intmap_full_road.astype(np.float16)
            rgb_future = rgb_future.astype(np.float16)
            rgb_full = rgb_full.astype(np.float16)
            bev.update({
                'road_future': probmap_future_road,
                'poses_future': poses_future,
                'road_full': probmap_full_road,
                'poses_full': poses_full,
                'intensity_future': intmap_future_road,
                'intensity_full': intmap_full_road,
                'rgb_future': rgb_future,
                'rgb_full': rgb_full,
            })

        return bev

    def viz_bev(self, bev, file_path, rgbs=[], semsegs=[]):
        '''
        '''
        present_road = bev['road_present']
        poses_present = bev['poses_present']
        present_intensity = bev['intensity_present']
        present_rgb = bev['rgb_present']

        H = self.pixel_size

        num_imgs = len(rgbs)
        num_cols = num_imgs if num_imgs > 3 else 3
        num_rows = 4 if num_imgs > 0 else 3

        # dim (3, H, W) --> (H, W, 3)
        present_rgb = np.transpose(present_rgb, (1, 2, 0))
        present_rgb = (present_rgb * 255).astype(np.int)

        if 'road_future' in bev.keys():
            future_road = bev['road_future']
            poses_future = bev['poses_future']
            full_road = bev['road_full']
            poses_full = bev['poses_full']
            future_intensity = bev['intensity_future']
            full_intensity = bev['intensity_full']
            future_rgb = bev['rgb_future']
            full_rgb = bev['rgb_full']

            future_rgb = np.transpose(future_rgb, (1, 2, 0))
            full_rgb = np.transpose(full_rgb, (1, 2, 0))
            future_rgb = (future_rgb * 255).astype(np.int)
            full_rgb = (full_rgb * 255).astype(np.int)

            size_per_fig = 6
            _ = plt.figure(figsize=(size_per_fig * num_cols,
                                    size_per_fig * num_rows))

            # Road semantic
            plt.subplot(num_rows, num_cols, 1)
            plt.imshow(present_road, vmin=0, vmax=1)
            plt.plot(poses_present[:, 0], H - poses_present[:, 1], 'r-')

            plt.subplot(num_rows, num_cols, 2)
            plt.imshow(future_road, vmin=0, vmax=1)
            plt.plot(poses_future[:, 0], H - poses_future[:, 1], 'r-')

            plt.subplot(num_rows, num_cols, 3)
            plt.imshow(full_road, vmin=0, vmax=1)
            plt.plot(poses_full[:, 0], H - poses_full[:, 1], 'r-')

            # Intensity
            plt.subplot(num_rows, num_cols, num_cols + 1)
            plt.imshow(present_intensity, vmin=0, vmax=1)
            plt.plot(poses_present[:, 0], H - poses_present[:, 1], 'r-')

            plt.subplot(num_rows, num_cols, num_cols + 2)
            plt.imshow(future_intensity, vmin=0, vmax=1)
            plt.plot(poses_future[:, 0], H - poses_future[:, 1], 'r-')

            plt.subplot(num_rows, num_cols, num_cols + 3)
            plt.imshow(full_intensity, vmin=0, vmax=1)
            plt.plot(poses_full[:, 0], H - poses_full[:, 1], 'r-')

            # Intensity
            plt.subplot(num_rows, num_cols, 2 * num_cols + 1)
            plt.imshow(present_rgb)
            plt.plot(poses_present[:, 0], H - poses_present[:, 1], 'r-')

            plt.subplot(num_rows, num_cols, 2 * num_cols + 2)
            plt.imshow(future_rgb)
            plt.plot(poses_future[:, 0], H - poses_future[:, 1], 'r-')

            plt.subplot(num_rows, num_cols, 2 * num_cols + 3)
            plt.imshow(full_rgb)
            plt.plot(poses_full[:, 0], H - poses_full[:, 1], 'r-')

            if num_imgs > 0:
                for idx in range(num_imgs):
                    plt.subplot(num_rows, num_cols, 3 * num_cols + idx + 1)
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

    def road_marking_transform(self, intensity_map: np.array,
                               int_scaler: float, int_sep_scaler: float,
                               int_mid_threshold: float):
        '''
        KITTI-360
            int_scaler = 20
            int_sep_scaler = 20
            int_mid_threshold = 0.5
        NuScenes
            int_scaler = 1
            int_sep_scaler = 30
            int_mid_threshold = 0.12

        Args:
            intensity_map: Value interval (0, 1)
        '''
        intensity_map = int_scaler * self.sigmoid(
            int_sep_scaler * (intensity_map - int_mid_threshold))
        # Normalization
        intensity_map[intensity_map > 1.] = 1.
        return intensity_map

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def get_rgb_maps(self, pc: np.array):
        '''
        Generates a set of RGB color maps based on the count of color values
        within the point cloud.

        Args:
            pc: Point cloud with RGB values (N, 8)
                [x, y, z, int, r, g, b, sem]
        '''
        # Initialize grid map with a list at each grid element
        red = []
        green = []
        blue = []
        for j in range(self.pixel_size):
            red.append([])
            green.append([])
            blue.append([])
            for i in range(self.pixel_size):
                red[j].append([])
                green[j].append([])
                blue[j].append([])

        # For each point, add color value to elements
        for idx in range(pc.shape[0]):
            i = pc[idx, 0].astype(int)
            j = pc[idx, 1].astype(int)
            r = pc[idx, 4]
            g = pc[idx, 5]
            b = pc[idx, 6]
            j_rev = self.pixel_size - 1 - j
            red[j_rev][i].append(r)
            green[j_rev][i].append(g)
            blue[j_rev][i].append(b)

        # Create RGB color maps from color value counts
        red_map = np.zeros((self.pixel_size, self.pixel_size))
        green_map = np.zeros((self.pixel_size, self.pixel_size))
        blue_map = np.zeros((self.pixel_size, self.pixel_size))
        for j in range(self.pixel_size):
            for i in range(self.pixel_size):
                j_rev = self.pixel_size - 1 - j
                # Fill element with default color if missing points
                if len(red[j_rev][i]) == 0:
                    red[j_rev][i].append(self.rgb_fill)
                    green[j_rev][i].append(self.rgb_fill)
                    blue[j_rev][i].append(self.rgb_fill)
                red_map[j_rev, i] = np.median(red[j_rev][i])
                green_map[j_rev, i] = np.median(green[j_rev][i])
                blue_map[j_rev, i] = np.median(blue[j_rev][i])

        return red_map, green_map, blue_map
