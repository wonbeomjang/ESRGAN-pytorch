from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
from random import random


class Datasets(Dataset):
    def __init__(self, image_size, scale):
        self.image_size = image_size
        self.scale = scale

        if not os.path.exists('datasets'):
            raise Exception(f"[!] dataset is not exited")

        self.image_file_name = sorted(os.listdir(os.path.join('datasets', 'hr')))

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        high_resolution = Image.open(os.path.join('datasets', 'hr', file_name)).convert('RGB')
        low_resolution = Image.open(os.path.join('datasets', 'lr', file_name)).convert('RGB')

        if random() > 0.5:
            high_resolution = TF.vflip(high_resolution)
            low_resolution = TF.vflip(low_resolution)

        if random() > 0.5:
            high_resolution = TF.hflip(high_resolution)
            low_resolution = TF.hflip(low_resolution)

        high_resolution = TF.to_tensor(high_resolution)
        low_resolution = TF.to_tensor(low_resolution)

        images = {'lr': low_resolution, 'hr': high_resolution}

        return images

    def __len__(self):
        return len(self.image_file_name)
