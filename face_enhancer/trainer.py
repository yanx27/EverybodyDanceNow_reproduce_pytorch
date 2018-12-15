import os
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from skimage.io import imsave

class Trainer(object):
    def __init__(self, ckpt_dir, log_dir, dataset, dataloader,
                 log_every=10, save_every=500,
                 max_batches=40000, gen_lr=1e-4, dis_lr=1e-4):

        if not os.path.isdir(ckpt_dir):
            os.mkdir(ckpt_dir)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.log_every = log_every
        self.save_every = save_every
        self.max_batches = max_batches
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.sampler = dataloader
        self.enumerator = None

        from face_enhancer.utils.perceptual_loss import VGG_perceptual_loss
        self.gen_loss = nn.MSELoss()
        self.recon_loss = VGG_perceptual_loss(pretrained=True, device=self.device)
        self.dis_loss = nn.MSELoss()  # LSGAN

        np.random.seed(233)

    # only keep samples
    def get_batch(self):
        if self.enumerator is None:
            self.enumerator = enumerate(self.sampler)

        batch_idx, batch = next(self.enumerator)
        b = {}
        for k, v in batch.items():
            b[k] = v.to(self.device)

        if batch_idx == len(self.sampler) - 1:
            self.enumerator = enumerate(self.sampler)

        return b

    @staticmethod
    def image2numpy(tensor):
        generated = tensor.detach().cpu().permute(1, 2, 0).numpy()
        generated[generated < -1] = -1
        generated[generated > 1] = 1
        generated = (generated + 1) / 2 * 255
        return generated.astype(np.uint8)

    @staticmethod
    def _init_logs():
        return {'gen_loss': 0, 'dis_loss': 0}

    def train_generator(self, g, d, g_opt):
        batch = self.get_batch()
        real_heads = batch['real_heads']
        fake_heads = batch['fake_heads']

        g_opt.zero_grad()

        residuals = g(fake_heads)
        enhanced_heads = fake_heads + residuals
        fake_features = d.extract_features(enhanced_heads)
        with torch.no_grad():
            real_features = d.extract_features(real_heads)
        gen_loss = self.gen_loss(fake_features, real_features)

        recon_loss = self.recon_loss(enhanced_heads, real_heads)
        # 20180924 change recon_loss weight
        # 20180929 ablation study on recon_loss weight
        gen_loss = gen_loss + 10 * recon_loss
        gen_loss_val = gen_loss.item()
        gen_loss.backward()
        g_opt.step()

        return gen_loss_val

    def train_discriminator(self, g, d, d_opt):
        batch = self.get_batch()
        real_heads = batch['real_heads']
        fake_heads = batch['fake_heads']

        d_opt.zero_grad()
        real_labels = d(real_heads)
        with torch.no_grad():
            ones = torch.ones_like(real_labels)
            zeros = torch.zeros_like(real_labels)

            # one sided label smoothing for vanilla gan
            ones.uniform_(.9, 1.1)
            zeros.uniform_(-.1, .1)

        dis_real_loss = self.dis_loss(real_labels, ones)
        dis_real_loss_val = dis_real_loss.item()

        residual = g(fake_heads)
        enhanced_heads = fake_heads + residual
        fake_labels = d(enhanced_heads)
        dis_fake_loss = self.dis_loss(fake_labels, zeros)
        dis_fake_loss_val = dis_fake_loss.item()

        dis_loss = dis_real_loss + dis_fake_loss
        dis_loss.backward()
        d_opt.step()

        return dis_real_loss_val + dis_fake_loss_val

    def save_models(self, generator, discriminator, batch_num):
        torch.save(generator.state_dict(), os.path.join(self.ckpt_dir, '%05d_generator.pth' % batch_num))
        torch.save(discriminator.state_dict(), os.path.join(self.ckpt_dir, '%05d_discriminator.pth' % batch_num))
        print('Model saved... Batch Num: %05d' % batch_num)

    def validate_and_save(self, generator, batch_num):
        idx = np.random.randint(self.dataset_len, size=(1,))[0]
        #print(idx)
        real_head, fake_head, top, bottom, left, right, real_full, fake_full \
            = self.dataset.get_full_sample(idx)

        with torch.no_grad():
            fake_head.unsqueeze_(0)
            real_head = real_head.to(self.device)
            fake_head = fake_head.to(self.device)
            residual = generator(fake_head)
            enhanced = fake_head + residual

        fake_head.squeeze_()
        enhanced.squeeze_()
        residual.squeeze_()
        image = torch.cat((real_head, fake_head, residual, enhanced), dim=2)
        image = self.image2numpy(image)
        imsave(os.path.join(self.log_dir, '%05d_enhanced_head.png' % batch_num), image)
        fake_full_old = fake_full.copy()
        fake_full[top: bottom, left: right, :] = self.image2numpy(enhanced)
        imsave(os.path.join(self.log_dir, '%05d_enhanced_full.png' % batch_num),
               np.concatenate((real_full, fake_full_old, fake_full), axis=1))

    def train(self, generator, discriminator, batch):
        generator = generator.to(self.device)
        discriminator = discriminator.to(self.device)
        generator.train()
        discriminator.train()

        opt_generator = optim.Adam(generator.parameters(),
                                   lr=self.gen_lr, betas=(0.5, 0.999),
                                   weight_decay=1e-5)
        opt_discriminator = optim.Adam(discriminator.parameters(),
                                       lr=self.dis_lr, betas=(0.5, 0.999),
                                       weight_decay=1e-5)

        logs = self._init_logs()
        start = time.time()
        while batch <= self.max_batches:
            batch += 1
            gen_loss = self.train_generator(generator, discriminator, opt_generator)
            dis_loss = self.train_discriminator(generator, discriminator, opt_discriminator)

            logs['gen_loss'] += gen_loss
            logs['dis_loss'] += dis_loss

            if batch % self.log_every == 0:
                log_string = "Batch %d" % batch
                for k, v in logs.items():
                    log_string += " [%s] %5.3f" % (k, v)

                log_string += ". Took %5.2f" % (time.time() - start)
                print(log_string)
                logs = self._init_logs()
                start = time.time()

            if batch % self.save_every == 0:
                self.validate_and_save(generator, batch)
                self.save_models(generator, discriminator, batch)
                torch.cuda.empty_cache()  # just for safety
