'''Pose2pose'''
'''Download and extract video'''
import cv2
from pathlib import Path
import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(0)

save_dir = Path('./data/source/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

if len(os.listdir('./data/source/images'))<100:
    cap = cv2.VideoCapture(str(save_dir.joinpath('mv.mp4')))
    i = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if flag == False or i >= 1000:
            break
        cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(i))), frame)
        if i%100 == 0:
            print('Has generated %d picetures'%i)
        i += 1

'''Pose estimation (OpenPose)'''
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

openpose_dir = Path('./src/PoseEstimation/')

import sys
sys.path.append(str(openpose_dir))
sys.path.append('./src/utils')


# openpose
#from network.rtpose_vgg import gopenpose_diret_model
from evaluate.coco_eval import get_multiplier, get_outputs
from network.rtpose_vgg import get_model
# utils
from openpose_utils import remove_noise, get_pose


weight_name = './src/PoseEstimation/network/weight/pose_model.pth'

model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

'''check'''
img_path = sorted(img_dir.iterdir())[137]
import os
path = os.getcwd()
img = cv2.imread('./data/source/images/img_999.png')
shape_dst = np.min(img.shape[:2])
# offset
oh = (img.shape[0] - shape_dst) // 2
ow = (img.shape[1] - shape_dst) // 2

img = img[oh:oh + shape_dst, ow:ow + shape_dst]
img = cv2.resize(img, (512, 512))

plt.imshow(img[:, :, [2, 1, 0]])  # BGR -> RGB


multiplier = get_multiplier(img)
with torch.no_grad():
    paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')

r_heatmap = np.array([remove_noise(ht)
                      for ht in heatmap.transpose(2, 0, 1)[:-1]]) \
    .transpose(1, 2, 0)
heatmap[:, :, :-1] = r_heatmap
param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
label = get_pose(param, heatmap, paf)

plt.imshow(label)
plt.show()

'''make label images for pix2pix'''
test_img_dir = save_dir.joinpath('test_img')
test_img_dir.mkdir(exist_ok=True)
test_label_dir = save_dir.joinpath('test_label_ori')
test_label_dir.mkdir(exist_ok=True)

for idx in tqdm(range(200,400)):
    img_path = img_dir.joinpath('{:05}.png'.format(idx))
    img = cv2.imread(str(img_path))
    shape_dst = np.min(img.shape[:2])
    oh = (img.shape[0] - shape_dst) // 2
    ow = (img.shape[1] - shape_dst) // 2

    img = img[oh:oh + shape_dst, ow:ow + shape_dst]
    img = cv2.resize(img, (512, 512))
    multiplier = get_multiplier(img)
    with torch.no_grad():
        paf, heatmap = get_outputs(multiplier, img, model, 'rtpose')
    r_heatmap = np.array([remove_noise(ht)
                          for ht in heatmap.transpose(2, 0, 1)[:-1]]) \
        .transpose(1, 2, 0)
    heatmap[:, :, :-1] = r_heatmap
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    label,_ = get_pose(param, heatmap, paf)
    cv2.imwrite(str(test_img_dir.joinpath('{:05}.png'.format(idx))), img)
    cv2.imwrite(str(test_label_dir.joinpath('{:05}.png'.format(idx))), label)

torch.cuda.empty_cache()
