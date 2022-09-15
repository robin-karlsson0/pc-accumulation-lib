import matplotlib as mpl

mpl.use('agg')  # Must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from .bev_generator import BEVGenerator


class RGBBEVGenerator(BEVGenerator):
    '''
    '''

    def __init__(self,
                 view_size: int,
                 pixel_size: int,
                 rgb_fill: int = 0,
                 max_trans_radius: float = 0.,
                 zoom_thresh: float = 0.,
                 do_warp: bool = False):
        '''
        '''
        super().__init__(view_size, pixel_size, max_trans_radius, zoom_thresh,
                         do_warp)

        # TODO Implement illumination normalization module

        self.rgb_fill = rgb_fill

    def generate_bev(self,
                     pc_present: np.array,
                     pc_future: np.array,
                     poses_present: np.array,
                     poses_future: np.array,
                     do_warping: bool = False):
        '''
        '''
        r_present, g_present, b_present = self.get_rgb_maps(pc_present)
        r_future, g_future, b_future = self.get_rgb_maps(pc_future)

        r_present /= 255.
        g_present /= 255.
        b_present /= 255.
        r_future /= 255.
        g_future /= 255.
        b_future /= 255.

        # Warp all probability maps and poses
        if do_warping:
            i_mid = int(self.pixel_size / 2)
            j_mid = i_mid
            # I_crop, J_crop = pixel_size
            i_warp, j_warp = self.get_random_warp_params(
                0.15, 0.30, self.pixel_size, self.pixel_size)
            a_1, a_2 = self.cal_warp_params(i_warp, i_mid, self.pixel_size - 1)
            b_1, b_2 = self.cal_warp_params(j_warp, j_mid, self.pixel_size - 1)

            rgbmaps = np.stack([
                r_present,
                g_present,
                b_present,
                r_future,
                g_future,
                b_future,
            ])
            rgbmaps = self.warp_dense_probmaps(rgbmaps, a_1, a_2, b_1, b_2)

            r_present = rgbmaps[0]
            g_present = rgbmaps[1]
            b_present = rgbmaps[2]
            r_future = rgbmaps[3]
            g_future = rgbmaps[4]
            b_future = rgbmaps[5]

            poses_present = self.warp_sparse_points(poses_present, a_1, a_2,
                                                    b_1, b_2, i_mid, j_mid,
                                                    i_warp, j_warp)
            poses_future = self.warp_sparse_points(poses_future, a_1, a_2, b_1,
                                                   b_2, i_mid, j_mid, i_warp,
                                                   j_warp)

        rgb_present = np.stack((r_present, g_present, b_present))
        rgb_future = np.stack((r_future, g_future, b_future))

        # Reduce storage size
        rgb_present = rgb_present.astype(np.float16)
        rgb_future = rgb_future.astype(np.float16)

        bev = {
            # Probability maps
            'rgb_present': rgb_present,
            'rgb_future': rgb_future,
            # Poses
            'poses_present': poses_present,
            'poses_future': poses_future,
        }

        return bev

    def viz_bev(self, bev, file_path):
        '''
        '''

        # Probmaps
        rgb_present = bev['rgb_present']
        rgb_future = bev['rgb_future']
        # Poses
        poses_present = bev['poses_present']
        poses_future = bev['poses_future']

        # dim (3, H, W) --> (H, W, 3)
        rgb_present = np.transpose(rgb_present, (1, 2, 0))
        rgb_future = np.transpose(rgb_future, (1, 2, 0))

        rgb_present = (rgb_present * 255).astype(np.int)
        rgb_future = (rgb_future * 255).astype(np.int)

        H = self.pixel_size

        _ = plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(rgb_present)
        plt.plot(poses_present[:, 0], H - poses_present[:, 1], 'b-')

        plt.subplot(1, 2, 2)
        plt.imshow(rgb_future)
        plt.plot(poses_future[:, 0], H - poses_future[:, 1], 'r-')

        plt.tight_layout()

        plt.savefig(file_path)
        plt.clf()
        plt.close()

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
