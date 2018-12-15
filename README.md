# EverybodyDanceNow reproduced in pytorch

Written by Peihuan Wu, Jinghong Lin, Yutao Liao, Wei Qing and Yan Xu, reference by [pytorch-EverybodyDanceNow](https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow), we add normalization  and face enhancement parts.

## Result
![Result](output.gif)

## Pre-trained models and experimental video
* Download vgg19-dcbb9e9d.pth.crdownload and put it in `./src/pix2pixHD/models/`  <br>Link: https://pan.baidu.com/s/1XMZpSY_UOIwFbN1NXfKEpA   code：agum 

* Download pose_model.pth and put it in `./src/PoseEstimation/network/weight/`   <br>Link: https://pan.baidu.com/s/1V68pNSzeaey9OCtVkO_f4Q   code：yf2x 

* Source video can be download from ：https://pan.baidu.com/s/15_PJzFf-rRHMkwji9T20gQ  code：leos 

* Download pre-trained vgg_16 for face enhancement https://pan.baidu.com/s/1RrcLjEtl4yJ40-4h9sZaDQ  code：62y0 and put in `./face_enhancer/`

## Full process
* Put source video mv.mp4 in `./data/source/` and run `make_source_images.py`, the label images will save in `./data/source/test_label_ori/` 
* Put target video mv.mp4 in `./data/target/` and run `make_target_images.py`, pose.npy will save in `./data/target/`, which contain the coordinate of very face.
* Run `train_target_images.py` and check loss and full training process in `./checkpoints/`
* If you break the traning and want to continue last training, set `opt.load_pretrain = './checkpoints/target/`
* Run `normalization.py` rescale the label images, you can use two sample images from `./data/target/train/train_label/` and `./data/source/test_label_ori/` to complete normalization between two skeleton size
* Run `transfer.py` and get results in `./result`
* Create `./data/face/test_sync` and `./data/face/test_real`, then put the same person's generated pictures and the original pictures in them.
* Run `./face_enhancer/main.py` train face enhancer and run`./face_enhancer/enhance.py` to gain result
* Run `make_gif.py` and make result pictures to gif picture

## Environments
python: 3.6.5 <br>
pytorch: 0.4.1  <br>
opencv: 3.4.4  <br>
