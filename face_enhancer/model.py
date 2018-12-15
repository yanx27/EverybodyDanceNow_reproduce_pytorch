import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional
import torch.utils.data
import numpy as np
from utils.spectral_norm import spectral_norm


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# transported from pix2pixHD project
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


"""
  Use CycleGAN's NLayerDiscriminator as a starting point...
  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

  The discriminator is fixed to process 64*64 input.
  params:
    @ n_layers: You can change this param to control the receptive field size
        n_layers  output  receptive field size
            4     13*13           34
            5      5*5            70
            6      1*1            256

  P.S. This implementation doesn't use sigmoid, so it must be trained with
  nn.BCEWithLogitLoss() instead of nn.BCELoss()!       
"""


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=4,
                 norm_layer=nn.BatchNorm2d,
                 use_sigmoid=True, use_bias=False):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                spectral_norm(norm_layer(ndf * nf_mult)),
                nn.LeakyReLU(0.2, True),
            ]

        # modify this part for feature extraction
        self.model = nn.Sequential(*sequence)  # Extract feature here

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence = [  # building a new sequence, not adding modules!
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            spectral_norm(norm_layer(ndf * nf_mult)),
            nn.LeakyReLU(0.2, True),
            # nn.Dropout(0.1),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=0)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.predictor = nn.Sequential(*sequence)

    # single label prediction
    def forward(self, x):
        return self.predictor(self.model(x)).squeeze()

    # high-level feature matching
    def extract_features(self, x):
        return self.model(x).squeeze()
