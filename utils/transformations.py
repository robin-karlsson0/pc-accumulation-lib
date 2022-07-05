import numpy as np


def velo2frame(pc_velo, P_velo_frame):
    """
    Transforms point cloud from 'velodyne' to 'image frame' coordinates.

    Args:
        pc_velo: np.array (N, 3)
        P_velo_frame: np.array (3, 4)
    """
    # Covnert point cloud to homogeneous coordinates
    pc_num = pc_velo.shape[0]
    pc_homo_velo = np.concatenate((pc_velo, np.ones((pc_num, 1))), axis=1)
    pc_homo_velo = pc_homo_velo.T

    # Transform point cloud 'velodyne' --> 'frame'
    pc_homo_frame = np.matmul(P_velo_frame, pc_homo_velo)
    pc_homo_frame = pc_homo_frame.T

    return pc_homo_frame


def velo2img(pc_velo, P_velo_frame, img_h, img_w, max_depth=np.inf):
    """
    Compures image coordinates for points and returns the point cloud contained
    in the image.

    Args:
        pc_velo: np.array (N, 4) [x, y, z, i]
        P_velo_frame: np.array (3, 4)
        img_h: int
        img_w: int
        max_depth: float
    
    Returns:
        pc_velo_frame: np.array (M, 6) [x, y, z, i, img_i, img_j]
    """
    pc_frame = velo2frame(pc_velo[:, :3], P_velo_frame)

    depth = pc_frame[:, 2]
    depth[depth == 0] = -1e-6
    u = np.round(pc_frame[:, 0] / np.abs(depth)).astype(int)
    v = np.round(pc_frame[:, 1] / np.abs(depth)).astype(int)

    # Generate mask for points within image
    mask = np.logical_and(
        np.logical_and(np.logical_and(u >= 0, u < img_w), v >= 0), v < img_h)
    mask = np.logical_and(np.logical_and(mask, depth > 0), depth < max_depth)
    # Convert to column vectors
    u = u[:, np.newaxis]
    v = v[:, np.newaxis]

    pc_velo_img = np.concatenate([pc_velo, u, v], axis=1)
    pc_velo_img = pc_velo_img[mask]

    return pc_velo_img


def gen_semantic_pc(pc_velo, semantic_map, P_velo_frame):
    """
    Returns a subset of points with semantic content from the semantic map.

    Args:
        P_velo_frame: np.array (3, 4) Velodyne coords --> Image frame coords.
        semantic_map: np.array (h, w, k) w. K layers.

    Returns:
        pc_velo_sem: np.array (M, 4+K) [x, y, z, i, sem_1, ... , sem_K]
    """
    img_h, img_w, _ = semantic_map.shape

    pc_velo_img = velo2img(pc_velo, P_velo_frame, img_h, img_w)

    u = pc_velo_img[:, -2].astype(int)
    v = pc_velo_img[:, -1].astype(int)

    sem = semantic_map[v, u, :]

    pc_velo_sem = np.concatenate([pc_velo_img[:, :4], sem], axis=1)

    return pc_velo_sem
