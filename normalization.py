# *_*coding:utf-8 *_*
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

target_img = cv2.imread('./data/target/train/train_label/02497.png')[:,:,0]
source_img = cv2.imread('./data/source/test_label_ori/00000.png')[:,:,0]

plt.subplot(121)
plt.imshow(target_img)
plt.subplot(122)
plt.imshow(source_img)
plt.show()


def get_scale(label_img):
    any1 = label_img.any(axis=1)
    linspace1 = np.arange(len(any1))
    head_x, height = linspace1[list(any1)][0], len(linspace1[list(any1)])
    any0 = label_img[head_x, :] != 0
    linspace2 = np.arange(len(any0))
    head_y = int(np.mean(linspace2[list(any0)]))
    return (head_x,head_y),height

target_head,target_height = get_scale(target_img)
target_head_x = target_head[0]
target_head_y = target_head[1]

source_head,source_height = get_scale(source_img)

path = './data/source/test_label_ori/'
output = './data/source/test_label/'
for img_dir in tqdm(os.listdir(path)):
    img = cv2.imread(path+img_dir)
    source_rsize = cv2.resize(img,
                              (int(img.shape[0] * target_height / source_height),
                               int(img.shape[1] * target_height / source_height)))

    source_pad = np.pad(source_rsize, ((1000, 1000), (1000, 1000),(0,0)), mode='edge')

    source_head_rs, source_height_rs = get_scale(source_pad[:,:,0])
    source_head_rs_x = source_head_rs[0]
    source_head_rs_y = source_head_rs[1]

    new_source = source_pad[
                 (source_head_rs_x - target_head_x):(source_head_rs_x + (target_img.shape[0] - target_head_x)),
                 int((source_pad.shape[1] - target_img.shape[1])/2):int((source_pad.shape[1]-(source_pad.shape[1] - target_img.shape[1])/2))
                 ]

    cv2.imwrite(output+img_dir,new_source)