# EverybodyDanceNow reproduced in pytorch

Written by Peihuan Wu, Jinghong Lin, Yutao Liao, Wei Qing and Yan Xu, including normalization and face enhancement parts.<br>
<br>
We train and evaluate on Ubuntu 16.04, so if you don't have linux environment, you can set `nThreads=0` in `EverybodyDanceNow_reproduce_pytorch/src/config/train_opt.py`.

## Reference:

[nyoki-mtl](https://github.com/nyoki-mtl) pytorch-EverybodyDanceNow

[Lotayou](https://github.com/Lotayou) everybody_dance_now_pytorch

## Pre-trained models and source video
* Download vgg19-dcbb9e9d.pth.crdownload [here](https://drive.google.com/file/d/1JG-pLXkPmyx3o4L33rG5WMJKMoOjlXhl/view?usp=sharing) and put it in `./src/pix2pixHD/models/`  <br>

* Download pose_model.pth [here](https://drive.google.com/file/d/1DDBQsoZ94N4NRKxZbwyEXt7Tz8KqgS_w/view?usp=sharing) and put it in `./src/PoseEstimation/network/weight/`   <br>

* Source video can be download from [here](https://drive.google.com/file/d/1drRBJypNGqOZV9WFutEzDYXkEelUjZXh/view?usp=sharing)

* Download pre-trained vgg_16 for face enhancement [here](https://drive.google.com/file/d/180WgIzh0aV1Aayl_b1X7mIhVhDUcW3b1/view?usp=sharing) and put in `./face_enhancer/`

## Full process
#### Pose2vid network

![](/result/pic1.png)
#### Make source pictures
* Put source video mv.mp4 in `./data/source/` and run `make_source.py`, the label images and coordinate of head will save in `./data/source/test_label_ori/` and `./data/source/pose_souce.npy` (will use in step6). If you want to capture video by camera, you can directly run `./src/utils/save_img.py`
#### Make target pictures
* Put target video mv.mp4 in `./data/target/` and run `make_target.py`, `pose.npy` will save in `./data/target/`, which contain the coordinate of faces (will use in step6).
#### Train and use pose2vid network
* Run `train_pose2vid.py` and check loss and full training process in `./checkpoints/`
* If you break the traning and want to continue last training, set `load_pretrain = './checkpoints/target/` in `./src/config/train_opt.py`
* Run `normalization.py` rescale the label images, you can use two sample images from `./data/target/train/train_label/` and `./data/source/test_label_ori/` to complete normalization between two skeleton size
* Run `transfer.py` and get results in `./result`
#### Face enhancement network

![](/result/pic2.png)
#### Train and use face enhancement network
* Run `./face_enhancer/prepare.py` and check the results in `./data/face/test_sync` and `./data/face/test_real`.
* Run `./face_enhancer/main.py` train face enhancer and run`./face_enhancer/enhance.py` to gain results <br>
This is comparision in original (left), generated image before face enhancement (median) and after enhancement (right). FaceGAN can learn the residual error between the real picture and the generated picture faces.

#### Performance of face enhancement 
![](/result/37500_enhanced_full.png)
![](/result/37500_enhanced_head.png)

#### Gain results
* Run `make_gif.py` and make result pictures to gif picture

![Result](/result/output.gif)

## ToDo

- Pose estimation
    - [x] Pose
    - [x] Face
    - [ ] Hand
- [x] pix2pixHD
- [x] FaceGAN
- [ ] Temporal smoothing

## Environments
Ubuntu 16.04 <br>
Python 3.6.5 <br>
Pytorch 0.4.1  <br>
OpenCV 3.4.4  <br>


