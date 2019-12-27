from model.ESRGAN import ESRGAN
import os
from glob import glob
import torch
from torchvision.utils import save_image
import torch.nn as nn


class Tester:
    def __init__(self, config, data_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = config.checkpoint_dir
        self.data_loader = data_loader
        self.scale_factor = config.scale_factor
        self.sample_dir = config.sample_dir
        self.num_epoch = config.num_epoch
        self.image_size = config.image_size
        self.upsampler = nn.Upsample(scale_factor=self.scale_factor)
        self.epoch = config.epoch
        self.build_model()

    def test(self):
        self.generator.eval()
        total_step = len(self.data_loader)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        for step, image in enumerate(self.data_loader):
            low_resolution = image['lr'].to(self.device)
            high_resolution = image['hr'].to(self.device)
            fake_high_resolution = self.generator(low_resolution)
            low_resolution = self.upsampler(low_resolution)
            print(f"[Batch {step}/{total_step}]... ")

            result = torch.cat((low_resolution, fake_high_resolution, high_resolution), 2)
            save_image(result, os.path.join(self.sample_dir, f"SR_{step}.png"))

    def build_model(self):
        self.generator = ESRGAN(3, 3, 64, scale_factor=self.scale_factor).to(self.device)
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            raise Exception(f"[!] No checkpoint in {self.checkpoint_dir}")

        generator = glob(os.path.join(self.checkpoint_dir, f'generator_{self.epoch - 1}.pth'))

        self.generator.load_state_dict(torch.load(generator[0]))