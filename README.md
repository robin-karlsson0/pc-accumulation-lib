# Semantic point cloud accumulation
Library for temporal accumulation of semantic point clouds generated from image and lidar senor data

## Dependencies
- Open3D
- ONNX Runtime
- PyTorch

## TLDR

1. Install dependencies (for Ubuntu 20.04)

```
pip install open3d
```
Ref: http://www.open3d.org/docs/release/getting_started.html

```
pip install onnxruntime-gpu
```
Ref: https://onnxruntime.ai/

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
Ref: https://pytorch.org/get-started/locally/

```
pip install nuscenes-devkit
pip install einops
pip install lovely-tensors
```

2. Change path to KITTI-360 datasets in the run scripts (`run_pc_accum.py` and `run_bev_gen.py`)
```
#################
#  Sample data
#################
kitti360_path = '/home/robin/datasets/KITTI-360'
```
2. Run point cloud accumulation example
```
python run_pc_accum.py
```
3. Run BEV generation example
```
python run_bev_gen.py
```

## Generated 'accumulated observations' dataset format

```
dataset/
    subdir000/
        bev_000.pkl.gz
        ...
    ...

bev_000.pkl.gz (dict)
    # Present (past --> present)
    ['road_present']      --> np.array (256,256)
    ['intensity_present'] --> np.array (256,256)
    ['rgb_present']       --> np.array (256,256)
    ['dynamic_present']   --> np.array (256,256)
    ['elevation_present'] --> np.array (256,256)
    # Future (present --> future)
    ['road_future']       --> np.array (256,256)
    ['intensity_future']  --> np.array (256,256)
    ['rgb_future']        --> np.array (256,256)
    ['dynamic_future']    --> np.array (256,256)
    ['elevation_future']  --> np.array (256,256)
    # Full (past --> future)
    ['road_full']         --> np.array (256,256)
    ['intensity_full']    --> np.array (256,256)
    ['rgb_full']          --> np.array (256,256)
    ['dynamic_full']      --> np.array (256,256)
    ['elevation_full']    --> np.array (256,256)
    # Trajectories
    ['trajs_present']     --> List of np.array (N,2) [x,y] in img coords.
    ['trajs_future']      --> List of np.array (N,2) [x,y] in img coords.
    ['trajs_full']        --> List of np.array (N,2) [x,y] in img coords.
    # Optional GT lanes
    ['gt_lanes']          --> List of np.array (N,2) [x,y] in img coords.

```


## Calibration parameters

The current implementation supports one forward-facing camera and a 360 deg lidar. The following calibration parameters need to be specified following the formulation used in KITTI-360.

Transformation matrix $H_{lidar2cam} \in \mathcal{R}^{4 \times 4}$.

Transformation matrix $P_{cam2img} \in \mathcal{R}^{3 \times 4}$

Focal lengths (?) $c_x, c_y, f_x, f_y \in \mathcal{R}$

These parameters are read from the KITTI-360 dataset as follows

```
######################
#  Calibration info
######################
h_cam_velo, h_velo_cam = get_transf_matrices(kitti360_path)
p_cam_frame = get_camera_intrinsics(kitti360_path)
p_velo_frame = np.matmul(p_cam_frame, h_velo_cam)
c_x = p_cam_frame[0, 2]
c_y = p_cam_frame[1, 2]
f_x = p_cam_frame[0, 0]
f_y = p_cam_frame[1, 1]

calib_params = {}
calib_params['h_velo_cam'] = h_velo_cam
calib_params['p_cam_frame'] = p_cam_frame
calib_params['p_velo_frame'] = p_velo_frame
calib_params['c_x'] = c_x
calib_params['c_y'] = c_y
calib_params['f_x'] = f_x
calib_params['f_y'] = f_y
```