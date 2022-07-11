import os
import time

import matplotlib as mpl

mpl.use('agg')  # Must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

from utils.bev_data_aug import (cal_warp_params, get_random_warp_params,
                                warp_dense, warp_points)


def rotation_matrix_3d(ang):
    return np.array([[np.cos(ang), -np.sin(ang), 0],
                     [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])


def crop_view(pc, view_size):
    '''
    Args:
        pc: np.array (N, M)
                [x, y, ...]
    '''
    mask = np.logical_and(pc[:, 0] > -0.5 * view_size,
                          pc[:, 0] < 0.5 * view_size)
    pc = pc[mask]
    mask = np.logical_and(pc[:, 1] > -0.5 * view_size,
                          pc[:, 1] < 0.5 * view_size)
    pc = pc[mask]

    return pc


def pos2grid(pc, view_size, pixel_size):
    '''
    Args:
        pc: np.array (N, M)
                [x, y, ...]
    '''
    pc[:,
       0:2] = np.floor(pc[:, 0:2] / view_size * pixel_size + 0.5 * pixel_size)

    return pc


def separate_semantic_pc(pc, sems):
    '''
    Args:
        pc: np.array (N, M)
                [x, y, ..., sem]
        sems: List of integers representing semanntics
    '''
    # Create a mask for all points having semantics
    mask = np.zeros(pc.shape[0], dtype=bool)
    for sem in sems:
        mask = np.logical_or(mask, pc[:, -1] == sem)

    pc_sem = pc[mask]
    inv_mask = np.invert(mask)
    pc_notsem = pc[inv_mask]

    return pc_sem, pc_notsem


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gen_gridmap_count_map(pc, pixel_size):
    '''
    '''
    ij = pc[:, :2]
    gridmap_counts, _, _ = np.histogram2d(ij[:, 1],
                                          ij[:, 0],
                                          range=[[0, pixel_size],
                                                 [0, pixel_size]],
                                          bins=[pixel_size, pixel_size])

    # Image to Cartesian coordinate axis direction
    gridmap_counts = np.flip(gridmap_counts, axis=0)

    return gridmap_counts


def dirichlet_dist_expectation(
    gridmaps,
    obs_weight=1,
):
    '''
    Args:
        obs_gridmaps: List of np.array() with likelihood probability gridmaps.
    
    Returns:
        post_gridmaps: List of np.array() with posterior probability gridmaps.
    '''
    n_gridmaps = len(gridmaps)
    gridmaps = np.stack(gridmaps)
    gridmaps *= obs_weight  # Consider previous downsampling of observations

    # Uniform prior
    gridmaps += 1.

    alpha0 = np.sum(gridmaps, axis=0)
    gridmaps /= alpha0

    gridmaps = [gridmaps[idx] for idx in range(n_gridmaps)]

    return gridmaps


def get_rgb_maps(pc: np.array, pixel_size: int, fill: int = 0):
    '''
    Args:
        pc: Point cloud with RGB values (N, 8)
            [x, y, z, int, r, g, b, sem]
    '''
    red = []
    green = []
    blue = []
    for j in range(pixel_size):
        red.append([])
        green.append([])
        blue.append([])
        for i in range(pixel_size):
            red[j].append([])
            green[j].append([])
            blue[j].append([])

    for idx in range(pc.shape[0]):
        i = pc[idx, 0].astype(int)
        j = pc[idx, 1].astype(int)
        r = pc[idx, 4]
        g = pc[idx, 5]
        b = pc[idx, 6]
        j_rev = pixel_size - 1 - j
        red[j_rev][i].append(r)
        green[j_rev][i].append(g)
        blue[j_rev][i].append(b)

    red_map = np.zeros((pixel_size, pixel_size))
    green_map = np.zeros((pixel_size, pixel_size))
    blue_map = np.zeros((pixel_size, pixel_size))
    for j in range(pixel_size):
        for i in range(pixel_size):
            j_rev = pixel_size - 1 - j
            if len(red[j_rev][i]) == 0:
                red[j_rev][i].append(fill)
                green[j_rev][i].append(fill)
                blue[j_rev][i].append(fill)
            red_map[j_rev, i] = np.median(red[j_rev][i])
            green_map[j_rev, i] = np.median(green[j_rev][i])
            blue_map[j_rev, i] = np.median(blue[j_rev][i])

    return red_map, green_map, blue_map


def gen_view(pc_past, pc_future, poses_past, poses_future, rot_ang, trans_dx,
             trans_dy, zoom_scalar, view_size, pixel_size):
    '''
    Args:
        pc_past: np.array (N, 8)
                 [x, y, z, i, r, g, b, sem]

    Returns:
        view: Dict with gridmaps for past and future observations.
    '''
    ###################
    #  Generate view
    ###################
    # Apply transformations
    rot_mat = rotation_matrix_3d(rot_ang)
    xyz = pc_past[:, :3]
    xyz = np.matmul(rot_mat, xyz.T).T
    pc_past[:, :3] = xyz
    xyz = pc_future[:, :3]
    xyz = np.matmul(rot_mat, xyz.T).T
    pc_future[:, :3] = xyz
    xyz = poses_past[:, :3]
    xyz = np.matmul(rot_mat, xyz.T).T
    poses_past[:, :3] = xyz
    xyz = poses_future[:, :3]
    xyz = np.matmul(rot_mat, xyz.T).T
    poses_future[:, :3] = xyz
    pc_past[:, 0] += trans_dx
    pc_past[:, 1] += trans_dy
    pc_future[:, 0] += trans_dx
    pc_future[:, 1] += trans_dy
    poses_past[:, 0] += trans_dx
    poses_past[:, 1] += trans_dy
    poses_future[:, 0] += trans_dx
    poses_future[:, 1] += trans_dy

    # Randomly change view size while keeping pixels same ==> Zoom invariance
    view_size = zoom_scalar * view_size
    pc_past = crop_view(pc_past, view_size)
    pc_future = crop_view(pc_future, view_size)
    poses_past = crop_view(poses_past, view_size)
    poses_future = crop_view(poses_future, view_size)

    # Transform position (x, y) --> grid element (i, j)
    pc_past = pos2grid(pc_past, view_size, pixel_size)
    pc_future = pos2grid(pc_future, view_size, pixel_size)
    poses_past = pos2grid(poses_past, view_size, pixel_size)
    poses_future = pos2grid(poses_future, view_size, pixel_size)

    # Separate semantics
    ROAD_SEM = 0
    SIDEWALK_SEM = 1
    CAR_SEM = 13
    TRUCK_SEM = 14
    BUS_SEM = 15
    MOTORCYCLE_SEM = 17
    # NOTE: Assumes 'sky', 'person', 'rider', 'train', 'bicycle' semantics are
    # already removed
    dynamic_filter = [CAR_SEM, TRUCK_SEM, BUS_SEM, MOTORCYCLE_SEM]
    pc_past_dynamic, pc_past_static = separate_semantic_pc(
        pc_past, dynamic_filter)
    _, pc_future_static = separate_semantic_pc(pc_future, dynamic_filter)

    pc_past_road, pc_past_notroad = separate_semantic_pc(
        pc_past_static, [ROAD_SEM])
    pc_past_sidwalk, pc_past_notsidewalk = separate_semantic_pc(
        pc_past_static, [SIDEWALK_SEM])
    pc_future_road, pc_future_notroad = separate_semantic_pc(
        pc_future_static, [ROAD_SEM])

    ##############
    #  RGB maps
    ##############
    red_map_past, green_map_past, blue_map_past = get_rgb_maps(pc_past_static,
                                                               pixel_size,
                                                               fill=255)
    red_map_future, green_map_future, blue_map_future = get_rgb_maps(
        pc_future_static, pixel_size, fill=255)

    red_map_past /= 255.
    green_map_past /= 255.
    blue_map_past /= 255.
    red_map_future /= 255.
    green_map_future /= 255.
    blue_map_future /= 255.

    # rgb = np.stack(
    #     (red_median, green_median, blue_median), axis=-1).astype(int)
    # plt.imshow(rgb)

    ###################
    #  Elevation maps
    ###################
    LIDAR_HEIGHT_FROM_GROUND = 1.7  # [m]
    elevmap_past_counts = np.zeros((pixel_size, pixel_size), dtype=int)
    elevmap_past_mean = np.zeros((pixel_size, pixel_size), dtype=float)

    for idx in range(pc_past_static.shape[0]):
        i = pc_past_static[idx, 0].astype(int)
        j = pc_past_static[idx, 1].astype(int)
        z = pc_past_static[idx, 2]
        elevmap_past_counts[pixel_size - 1 - j, i] += 1
        elevmap_past_mean[pixel_size - 1 - j, i] += z

    # Get average elevation
    elevmap_past_mean /= (elevmap_past_counts + 1e-14)
    elevmap_past_mean[elevmap_past_counts == 0] = -LIDAR_HEIGHT_FROM_GROUND

    elevmap_dynamic_counts = np.zeros((pixel_size, pixel_size), dtype=int)
    elevmap_dynamic_mean = np.zeros((pixel_size, pixel_size), dtype=float)

    for idx in range(pc_past_dynamic.shape[0]):
        i = pc_past_dynamic[idx, 0].astype(int)
        j = pc_past_dynamic[idx, 1].astype(int)
        z = pc_past_dynamic[idx, 2]
        elevmap_dynamic_counts[pixel_size - 1 - j, i] += 1  # [i, j]
        elevmap_dynamic_mean[pixel_size - 1 - j, i] += z

    # Get average elevation
    elevmap_dynamic_mean /= (elevmap_dynamic_counts + 1e-14)
    elevmap_dynamic_mean[elevmap_dynamic_counts ==
                         0] = -LIDAR_HEIGHT_FROM_GROUND

    ##########################
    #  Lidar intensity maps
    ##########################
    intensitymap_past_counts = np.zeros((pixel_size, pixel_size), dtype=int)
    intensitymap_past_mean = np.zeros((pixel_size, pixel_size), dtype=float)

    for idx in range(pc_past_road.shape[0]):
        i = pc_past_road[idx, 0].astype(int)
        j = pc_past_road[idx, 1].astype(int)
        intensity = pc_past_road[idx, 3]
        intensity = 4 * sigmoid(20 * (intensity - 0.5))
        intensitymap_past_counts[pixel_size - 1 - j, i] += 1
        intensitymap_past_mean[pixel_size - 1 - j, i] += intensity

    # Get average intensity
    intensitymap_past_mean /= (intensitymap_past_counts + 1e-14)
    intensitymap_past_mean[intensitymap_past_mean > 1.] = 1.

    intensitymap_future_counts = np.zeros((pixel_size, pixel_size), dtype=int)
    intensitymap_future_mean = np.zeros((pixel_size, pixel_size), dtype=float)

    for idx in range(pc_future_road.shape[0]):
        i = pc_future_road[idx, 0].astype(int)
        j = pc_future_road[idx, 1].astype(int)
        intensity = pc_future_road[idx, 3]
        intensity = 4 * sigmoid(20 * (intensity - 0.5))
        intensitymap_future_counts[pixel_size - 1 - j, i] += 1
        intensitymap_future_mean[pixel_size - 1 - j, i] += intensity

    # Get average intensity
    intensitymap_future_mean /= (intensitymap_future_counts + 1e-14)
    intensitymap_future_mean[intensitymap_future_mean > 1.] = 1.

    # Bin points by semantics
    gridmap_past_road = gen_gridmap_count_map(pc_past_road, pixel_size)
    gridmap_past_notroad = gen_gridmap_count_map(pc_past_notroad, pixel_size)
    gridmap_future_road = gen_gridmap_count_map(pc_future_road, pixel_size)
    gridmap_future_notroad = gen_gridmap_count_map(pc_future_notroad,
                                                   pixel_size)

    gridmap_past_sidewalk = gen_gridmap_count_map(pc_past_sidwalk, pixel_size)
    gridmap_past_notsidewalk = gen_gridmap_count_map(pc_past_notsidewalk,
                                                     pixel_size)

    gridmap_dynamic = gen_gridmap_count_map(pc_past_dynamic, pixel_size)
    gridmap_static = gen_gridmap_count_map(pc_past_static, pixel_size)

    gridmaps = [gridmap_past_road, gridmap_past_notroad]
    gridmaps = dirichlet_dist_expectation(
        gridmaps,
        obs_weight=1,
    )
    gridmap_past_road, gridmap_past_notroad = gridmaps

    gridmaps = [gridmap_future_road, gridmap_future_notroad]
    gridmaps = dirichlet_dist_expectation(
        gridmaps,
        obs_weight=1,
    )
    gridmap_future_road, gridmap_future_notroad = gridmaps

    gridmaps = [gridmap_past_sidewalk, gridmap_past_notsidewalk]
    gridmaps = dirichlet_dist_expectation(
        gridmaps,
        obs_weight=1,
    )
    gridmap_past_sidewalk, gridmap_past_notsidewalk = gridmaps

    gridmaps = [gridmap_dynamic, gridmap_static]
    gridmaps = dirichlet_dist_expectation(
        gridmaps,
        obs_weight=1,
    )
    gridmap_dynamic, gridmap_static = gridmaps

    # Discard non-dynamic information: p(dynamic) --> [0, 1]
    gridmap_dynamic[gridmap_dynamic < 0.5] = 0.5
    gridmap_dynamic -= 0.5
    gridmap_dynamic *= 2.

    mask = gridmap_dynamic < 0.1
    elevmap_dynamic_mean[mask] = -LIDAR_HEIGHT_FROM_GROUND

    #############
    #  Warping
    #############
    i_mid = int(pixel_size / 2)
    j_mid = i_mid
    # I_crop, J_crop = pixel_size
    i_warp, j_warp = get_random_warp_params(0.15, 0.30, pixel_size,
                                            pixel_size)  # 0.15, 0.30
    a_1, a_2 = cal_warp_params(i_warp, i_mid, pixel_size - 1)
    b_1, b_2 = cal_warp_params(j_warp, j_mid, pixel_size - 1)

    arrays = np.stack([
        gridmap_past_road,
        gridmap_past_sidewalk,
        gridmap_future_road,
        gridmap_dynamic,
        elevmap_past_mean,
        elevmap_dynamic_mean,
        intensitymap_past_mean,
        intensitymap_future_mean,
        red_map_past,
        green_map_past,
        blue_map_past,
        red_map_future,
        green_map_future,
        blue_map_future,
    ])
    arrays = warp_dense(arrays, a_1, a_2, b_1, b_2)
    gridmap_past_road = arrays[0]
    gridmap_past_sidewalk = arrays[1]
    gridmap_future_road = arrays[2]
    gridmap_dynamic = arrays[3]
    elevmap_past_mean = arrays[4]
    elevmap_dynamic_mean = arrays[5]
    intensitymap_past_mean = arrays[6]
    intensitymap_future_mean = arrays[7]
    red_map_past = arrays[8]
    green_map_past = arrays[9]
    blue_map_past = arrays[10]
    red_map_future = arrays[11]
    green_map_future = arrays[12]
    blue_map_future = arrays[13]

    # NOTE No idea why, but the j warping must be reverse ...
    j_warp_rev = pixel_size - j_warp
    b_1_rev, b_2_rev = cal_warp_params(j_warp_rev, j_mid, pixel_size - 1)

    pnts = list(zip(poses_past[:, 0], poses_past[:, 1]))
    pnts = warp_points(pnts, a_1, a_2, b_1_rev, b_2_rev, pixel_size,
                       pixel_size)
    pnts = [[i for i, _ in pnts], [j for _, j in pnts]]
    poses_past[:, 0] = pnts[0]
    poses_past[:, 1] = pnts[1]
    pnts = list(zip(poses_future[:, 0], poses_future[:, 1]))
    pnts = warp_points(pnts, a_1, a_2, b_1_rev, b_2_rev, pixel_size,
                       pixel_size)
    pnts = [[i for i, _ in pnts], [j for _, j in pnts]]
    poses_future[:, 0] = pnts[0]
    poses_future[:, 1] = pnts[1]

    # Reduce storage size
    gridmap_past_road = gridmap_past_road.astype(np.float16)
    gridmap_past_sidewalk = gridmap_past_sidewalk.astype(np.float16)
    gridmap_future_road = gridmap_future_road.astype(np.float16)
    gridmap_dynamic = gridmap_dynamic.astype(np.float16)
    elevmap_past_mean = elevmap_past_mean.astype(np.float16)
    elevmap_dynamic_mean = elevmap_dynamic_mean.astype(np.float16)
    intensitymap_past_mean = intensitymap_past_mean.astype(np.float16)
    intensitymap_future_mean = intensitymap_future_mean.astype(np.float16)
    red_map_past = red_map_past.astype(np.float16)
    green_map_past = green_map_past.astype(np.float16)
    blue_map_past = blue_map_past.astype(np.float16)
    red_map_future = red_map_future.astype(np.float16)
    green_map_future = green_map_future.astype(np.float16)
    blue_map_future = blue_map_future.astype(np.float16)

    bev = {
        'gridmap_past_road': gridmap_past_road,
        # 'gridmap_past_notroad': gridmap_past_notroad,
        'gridmap_past_sidewalk': gridmap_past_sidewalk,
        # 'gridmap_past_notsidewalk': gridmap_past_notsidewalk,
        'gridmap_future_road': gridmap_future_road,
        # 'gridmap_future_notroad': gridmap_future_notroad,
        'gridmap_dynamic': gridmap_dynamic,
        # 'gridmap_notdynamic': gridmap_notdynamic,
        'elevmap_past_mean': elevmap_past_mean,
        # 'elevmap_dynamic_mean': elevmap_dynamic_mean,
        'intensitymap_past_mean': intensitymap_past_mean,
        'intensitymap_future_mean': intensitymap_future_mean,
        # RGB maps
        'red_map_past': red_map_past,
        'green_map_past': green_map_past,
        'blue_map_past': blue_map_past,
        'red_map_future': red_map_future,
        'green_map_future': green_map_future,
        'blue_map_future': blue_map_future,
        # Poses
        'poses_past': poses_past,
        'poses_future': poses_future,
    }

    return bev


def gen_aug_view(inputs):
    '''
    '''
    pc_past = inputs['pc_present']
    pc_future = inputs['pc_future']
    poses_past = inputs['poses_present']
    poses_future = inputs['poses_future']
    view_size = inputs['view_size']
    pixel_size = inputs['pixel_size']
    max_trans_radius = inputs['max_translation_radius']
    zoom_threshold = inputs['zoom_threshold']

    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    rot_ang = 2 * np.pi * np.random.random()
    trans_r = max_trans_radius * np.random.random()
    trans_ang = 2 * np.pi * np.random.random()
    trans_dx = trans_r * np.cos(trans_ang)
    trans_dy = trans_r * np.sin(trans_ang)
    zoom_scalar = np.random.normal(0, 0.1)
    if zoom_scalar < -zoom_threshold:
        zoom_scalar = -zoom_threshold
    elif zoom_scalar > zoom_threshold:
        zoom_scalar = zoom_threshold
    zoom_scalar = 1 + zoom_scalar

    bev = gen_view(pc_past, pc_future, poses_past, poses_future, rot_ang,
                   trans_dx, trans_dy, zoom_scalar, view_size, pixel_size)

    return bev


def viz_bev(bev, file_path):
    '''
    NOTE: Will leak memory if 'figsize' is set
    '''
    gridmap_past_road = bev['gridmap_past_road']
    gridmap_past_sidewalk = bev['gridmap_past_sidewalk']
    gridmap_future_road = bev['gridmap_future_road']

    gridmap_dynamic = bev['gridmap_dynamic']

    elevmap_past_mean = bev['elevmap_past_mean']

    intensitymap_past_mean = bev['intensitymap_past_mean']

    intensitymap_future_mean = bev['intensitymap_future_mean']

    red_map_past = bev['red_map_past']
    green_map_past = bev['green_map_past']
    blue_map_past = bev['blue_map_past']
    red_map_future = bev['red_map_future']
    green_map_future = bev['green_map_future']
    blue_map_future = bev['blue_map_future']

    poses_past = bev['poses_past']
    poses_future = bev['poses_future']

    # RGB maps are assumed to be float arrays w. values (0, 1)
    rgb_past = np.stack((red_map_past, green_map_past, blue_map_past), axis=-1)
    rgb_future = np.stack((red_map_future, green_map_future, blue_map_future),
                          axis=-1)
    rgb_past = (rgb_past * 255).astype(int)
    rgb_future = (rgb_future * 255).astype(int)

    H, _ = gridmap_past_road.shape

    fig = plt.figure(figsize=(32, 18))
    plt.subplot(2, 5, 1)
    plt.imshow(gridmap_past_road, vmin=0, vmax=1)
    plt.plot(poses_past[:, 0], H - poses_past[:, 1], 'k-')
    plt.subplot(2, 5, 2)
    plt.imshow(gridmap_past_sidewalk, vmin=0, vmax=1)
    plt.subplot(2, 5, 3)
    plt.imshow(intensitymap_past_mean, vmin=0, vmax=1)
    plt.subplot(2, 5, 4)
    plt.imshow(gridmap_dynamic, vmin=0, vmax=1)
    plt.subplot(2, 5, 5)
    plt.imshow(elevmap_past_mean, vmin=-2, vmax=2)
    plt.subplot(2, 5, 6)
    plt.imshow(gridmap_future_road, vmin=0, vmax=1)
    plt.plot(poses_future[:, 0], H - poses_future[:, 1], 'r-')
    # plt.subplot(2,5,7)
    # plt.imshow(intensitymap_future_max, vmin=0, vmax=1)
    plt.subplot(2, 5, 8)
    plt.imshow(intensitymap_future_mean, vmin=0, vmax=1)

    plt.subplot(2, 5, 9)
    plt.imshow(rgb_past)
    plt.subplot(2, 5, 10)
    plt.imshow(rgb_future)

    plt.tight_layout()

    plt.savefig(file_path)
    plt.clf()
    plt.close()
