import model
import dataset
from trainer import Trainer
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.io import imsave
from imageio import get_writer

image_transforms = transforms.Compose([
        Image.fromarray,
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
    ])
    
device = torch.device('cuda')


def load_models(directory, nd, nb):
    generator = model.GlobalGenerator(n_downsampling=nd, n_blocks=nb)
    gen_name = os.path.join(directory, 'final_generator.pth')

    if os.path.isfile(gen_name):
        gen_dict = torch.load(gen_name)
        generator.load_state_dict(gen_dict)
        
    return generator.to(device)

    
def torch2numpy(tensor):
        generated = tensor.detach().cpu().permute(1, 2, 0).numpy()
        generated[generated < -1] = -1
        generated[generated > 1] = 1
        generated = (generated + 1) / 2 * 255
        return generated.astype(np.uint8)
    
    
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    dataset_dir = '../data/face'
    pose_name = '../data/target/pose.npy'
    ckpt_dir = '../checkpoints/yxu_face'
    result_dir = './results'

    image_folder = dataset.ImageFolderDataset(dataset_dir, cache=os.path.join(dataset_dir, 'local.db'), is_test=True)
    face_dataset = dataset.FaceCropDataset(image_folder, pose_name, image_transforms, crop_size=48)
    length = len(face_dataset)
    print(length)
    
    path = 'dance_test_new_down2_res6'
    nd = int(path[path.find('down') + 4])
    nb = int(path[path.find('res') + 3])
    print(path, nd, nb)
    generator = load_models(os.path.join(ckpt_dir, path), nd, nb)

    video = []    
    for i in range(length):
        _, fake_head, top, bottom, left, right, real_full, fake_full \
            = face_dataset.get_full_sample(i)

        with torch.no_grad():
            fake_head.unsqueeze_(0)
            fake_head = fake_head.to(device)
            residual = generator(fake_head)
            enhanced = fake_head + residual

        enhanced.squeeze_()
        enhanced = torch2numpy(enhanced)
        fake_full_old = fake_full.copy()
        fake_full[top: bottom, left: right, :] = enhanced
        video.append(np.concatenate((real_full, fake_full_old, fake_full), axis=1))
               
    with get_writer('teaser.avi', fps=25) as w:
        for im in video:
            w.append_data(im)
