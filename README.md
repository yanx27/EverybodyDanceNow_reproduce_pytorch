# Every body dance now pytorch

Reference by [pytorch-EverybodyDanceNow](https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow), we write normalization part and add some improvement.

## Result
![Result](output.gif)

## Pre-trained models and experimental video
* Download vgg19-dcbb9e9d.pth.crdownload and put it in `./src/pix2pixHD/models/`  <br>Link: https://pan.baidu.com/s/1XMZpSY_UOIwFbN1NXfKEpA   code：agum 

* Download pose_model.pth and put it in `./src/PoseEstimation/network/weight/`   <br>Link: https://pan.baidu.com/s/1V68pNSzeaey9OCtVkO_f4Q   code：yf2x 

* Source video can be download from ：https://pan.baidu.com/s/15_PJzFf-rRHMkwji9T20gQ  code：leos 

## Full process
* Put source video mv.mp4 in `./data/source/` and run `make_source_images.py`, the label images will save in `./data/source/test_label_ori/` 
* Put target video mv.mp4 in `./data/target/` and run `make_target_images.py`
* Run `train_target_images.py` and check loss and full training process in `./checkpoints/`
* If you break the traning and want to continue last training, set `opt.load_pretrain = './checkpoints/target/`
* Run `normalization.py` rescale the label images, you can use two sample images from `./data/target/train/train_label/` and `./data/source/test_label_ori/` to complete normalization between two skeleton size
* Run `transfer.py` and get results in `./result`
* Run `make_gif.py` and make result pictures to gif picture

## Environments
python: 3.5.4 <br>
pytorch: 0.4.1  <br>
opencv: 3.4.1  <br>