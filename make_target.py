import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

openpose_dir = Path('./src/PoseEstimation/')


save_dir = Path('./data/target/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)

if len(os.listdir('./data/target/images'))<100:
    cap = cv2.VideoCapture(str(save_dir.joinpath('mv2.mp4')))
    i = 0
    while (cap.isOpened()):
        flag, frame = cap.read()
        if flag == False :
            break
        cv2.imwrite(str(img_dir.joinpath('{:05}.png'.format(i))), frame)
        if i%100 == 0:
            print('Has generated %d picetures'%i)
        i += 1

import sys
sys.path.append(str(openpose_dir))
sys.path.append('./src/utils')
# openpose
from network.rtpose_vgg import get_model
from evaluate.coco_eval import get_multiplier, get_outputs

# utils
from openpose_utils import remove_noise, get_pose

weight_name = './src/PoseEstimation/network/weight/pose_model.pth'
print('load model...')
model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()
pass

save_dir = Path('./data/target/')
save_dir.mkdir(exist_ok=True)

img_dir = save_dir.joinpath('images')
img_dir.mkdir(exist_ok=True)


'''make label images for pix2pix'''
train_dir = save_dir.joinpath('train')
train_dir.mkdir(exist_ok=True)

train_img_dir = train_dir.joinpath('train_img')
train_img_dir.mkdir(exist_ok=True)
train_label_dir = train_dir.joinpath('train_label')
train_label_dir.mkdir(exist_ok=True)
train_head_dir = train_dir.joinpath('head_img')
train_head_dir.mkdir(exist_ok=True)

pose_cords = []
for idx in tqdm(range(2000)):
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
                          for ht in heatmap.transpose(2, 0, 1)[:-1]]).transpose(1, 2, 0)
    heatmap[:, :, :-1] = r_heatmap
    param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
    #TODO get_pose
    label, cord = get_pose(param, heatmap, paf)
    index = 13
    pose_cords.append(cord[index])
    crop_size = 25

    head = img[int(cord[index][1] - crop_size): int(cord[index][1] + crop_size),
           int(cord[index][0] - crop_size): int(cord[index][0] + crop_size), :]
    plt.imshow(head)
    plt.savefig(str(train_head_dir.joinpath('pose_{}.jpg'.format(idx))))

    cv2.imwrite(str(train_img_dir.joinpath('{:05}.png'.format(idx))), img)
    cv2.imwrite(str(train_label_dir.joinpath('{:05}.png'.format(idx))), label)

pose_cords = np.array(pose_cords, dtype=np.int)
np.save(str((save_dir.joinpath('pose.npy'))), pose_cords)
torch.cuda.empty_cache()
