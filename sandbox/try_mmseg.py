import numpy as np
import torch
import torchvision
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from nuscenes.nuscenes import NuScenes
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt


palette = get_palette('cityscapes')
print(palette)

mmseg_root = '/home/user/Desktop/python_ws/mmsegmentation'
config_file = osp.join(mmseg_root, 'configs', 'deeplabv3', 'deeplabv3_r18-d8_512x1024_80k_cityscapes.py')
ckpt_file = osp.join(mmseg_root, 'checkpoints', 'deeplabv3_r18-d8_512x1024_80k_cityscapes_20201225_021506-23dffbe2.pth')
model = init_segmentor(config_file, ckpt_file, device='cuda:0')

nusc = NuScenes(dataroot='/home/user/dataset/nuscenes/', version='v1.0-mini', verbose=False)
scene = nusc.scene[0]
sample = nusc.sample[10]
cam_front = nusc.get('sample_data', sample['data']['CAM_FRONT'])
img = osp.join(nusc.dataroot, cam_front['filename'])

result = inference_segmentor(model, img)
show_result_pyplot(model, img, result, get_palette('cityscapes'))

img_ = Image.open(img)
img_road = img_ * np.expand_dims(result[0] == 0, -1)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_)
ax[1].imshow(img_road)
plt.show()
