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
pip3 install torch torchvision torchaudio
```
Ref: https://pytorch.org/get-started/locally/

```
pip install nuscenes-devkit
```

2. Download the image semantic segmentation ONNX model

https://drive.google.com/file/d/1-pOhaAX_elDQrhPKz8wZnzAOuUeWCQrQ/view?usp=share_link

3. Run KITTI360 BEV generation
```
python run_kitti360_bev_gen.py dataset_path semseg_path

dataset_path: Path to the KITTI-360 dataset root directory (e.g. `/home/robin/datasets/KITTI-360`)
semseg_path: path to the downloaded image semantic segmentation ONNX model (e.g. `./semseg_rn50_160k_cm.onnx`)
```

See available command line arguments for additional run options.

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
    # Sample information
    ['scene_idx']         --> int
    ['map']               --> str
    ['ego_global_x']      --> float
    ['ego_global_y']      --> float
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