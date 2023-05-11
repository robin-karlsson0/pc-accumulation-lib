import matplotlib.pyplot as plt
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap

# Needed to reset style set by NuScenesMap
plt.style.use('default')


def crop_centerline_poses(pose_list: list, bbox: tuple):
    '''
    Args:
        pose_list: [(x,y,z), ...]
        bbox: (x0, y0, x1, y1) global coordinates (i.e. map).
    '''

    for poses in pose_list:
        # 'x' range
        mask = np.logical_and(poses[:, 0] > bbox[0], poses[:, 0] < bbox[2])
        poses = poses[mask]
        # 'y' range
        mask = np.logical_and(poses[:, 1] > bbox[1], poses[:, 1] < bbox[3])
        poses = poses[mask]

    return pose_list


def get_centerlines(dataroot: str,
                    map_name: str,
                    bbox: tuple = None,
                    resolution_meters: float = 1.) -> list:
    '''
    Args:
        bbox: (x0, y0, x1, y1) global coordinates (i.e. map).

    '''
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_name)
    poses = nusc_map.discretize_centerlines(resolution_meters)

    # Optionally crop out lines outside the specified bounding box
    if bbox is not None:
        poses = crop_centerline_poses(poses, bbox)

    return poses


def render_centerlines(pose: dict) -> np.array:
    '''
     '''
    from nuscenes.map_expansion.bitmap import BitMap

    nusc_map = NuScenesMap(dataroot='/home/robin/datasets/nuscenes',
                           map_name='boston-seaport')

    bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
    fig, ax = nusc_map.render_centerlines(resolution_meters=0.5,
                                          figsize=1,
                                          bitmap=bitmap)

    plt.show()


if __name__ == "__main__":

    render_centerlines(None)
