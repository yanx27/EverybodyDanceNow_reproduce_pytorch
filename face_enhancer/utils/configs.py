"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

"""

import torch

class Config():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def debug(num=0):
    return Config(
        ngf=64,
        from_batch = num,
        # video_batch_size = 1,
        # video_length = 3,
        # clip_size=3,
        image_batch_size = 3,
        log_interval = 1,
        save_interval = 1,
        train_batches = 100000,
        dataset = 'D:/data/ntu_image_skeleton',
        log_folder='logs_debug',
        image_height = 256,
        image_width = None,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        is_debug = True,
        enable_visdom = False,
        random_flip = True,
        use_dropout = True,
        recon_loss_weight = 2,
        gen_lr = 2e-5,
        gen_loss_type = 'feature_matching_loss',
        # gen_loss_type = 'gan_loss',
	    # recon_loss_type='vgg_perceptual_loss',
	    recon_loss_type='L1_loss',
        dis_lr = 1e-5,
        dis_loss_type = 'least_square_loss',
        # dis_loss_type = 'gan_loss',

    )

def train(num=0):
    return Config(
        ngf=64,
        from_batch=num,
        # video_batch_size = 1,
        # video_length = 3,
        # clip_size=3,
        image_batch_size=3,
        log_interval=10,
        save_interval=200,
        train_batches=100000,
        dataset='D:/data/ntu_image_skeleton_clean_bg',
        log_folder='logs_train_ntu_256_clean_bg_#13',
        image_height=256,
        image_width=None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        is_debug=False,
        enable_visdom=True,
        random_flip=True,
        use_dropout=True,
        recon_loss_weight=2,
        gen_lr = 1e-5,
        gen_loss_type='feature_matching_loss',
	    # gen_loss_type = 'gan_loss',
	    # recon_loss_type='vgg_perceptual_loss',
	    recon_loss_type='L1_loss',
	    dis_lr = 1e-5,
        dis_loss_type = 'least_square_loss',
        # dis_loss_type = 'gan_loss',
        changes_upon_last_version=[
            'Try spectral norm',
            'Dang it I\'m out... (for now)?'
        ]
    )
