from torchvision.models import vgg
import torch
from torch import nn
import os
"""
  TODO: VGG perceptual loss module
  something is wrong? Must double check this time...
"""
class VGG_perceptual_loss(nn.Module):
    def __init__(self, pretrained=False, device='cuda'):
        super(VGG_perceptual_loss, self).__init__()
        self.device = device
        self.loss_function = nn.L1Loss()
        self.vgg_features = vgg.make_layers(vgg.cfg['D'])
        if pretrained:
            self.vgg_features.load_state_dict(
                torch.load('vgg16_pretrained_features.pth'))
        self.vgg_features.to(device)
        # freeze parameter update
        for params in self.vgg_features.parameters():
            params.requires_grad = False
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, input, target):
        # TODO: extract 16 layers of activations and return weighted L1-loss.
        loss = torch.tensor(0.).to(self.device)
        for name, module in self.vgg_features._modules.items():
            input = module(input)
            target = module(target)
            if name in self.layer_name_mapping:
                loss += self.loss_function(input, target)
        return 0.1 * loss  # recon loss should be on the same level as gen loss...


#
# if __name__ == '__main__':
# 	# from torchvision.models import vgg16
# 	# vggnet = vgg16()
# 	# vggnet.load_state_dict(torch.load('vgg16-397923af.pth'))
# 	# torch.save(vggnet.features.state_dict(), 'vgg16_pretrained_features.pth')
# 	vg = VGG_perceptual_loss(pretrained=True)
# 	print(vg)