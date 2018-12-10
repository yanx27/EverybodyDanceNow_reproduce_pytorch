# Every body dance now pytorch

* Written by Peihuan Wu, Jinghong Lin, Wei Qing, Yutao Liao and Yan Xu

* Download vgg19-dcbb9e9d.pth.crdownload and put it in ./src/pix2pixHD/models/   <br>Link: https://pan.baidu.com/s/1XMZpSY_UOIwFbN1NXfKEpA   code：agum 

* Download pose_model.pth and put it in ./src/PoseEstimation/network/weight/   <br>Link: https://pan.baidu.com/s/1V68pNSzeaey9OCtVkO_f4Q   code：yf2x 

* Put source video mv.mp4 in `./data/source/` and run `make_source_images.py`
* Put target video mv.mp4 in `./data/target/` and run `make_target_images.py`
* Run `train_target_images.py` and check loss and full training process in `./checkpoints/`
* Run `transfer.py` and get results in `./result`
* Run `make_gif.py` and make result pictures to gif picture

* Reference by [pytorch-EverybodyDanceNow](https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow)
 
