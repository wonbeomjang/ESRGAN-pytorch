from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import torchvision.transforms.functional as TF
from random import random


class Datasets(Dataset):
    def __init__(self, data_dir, image_size, scale):
        self.data_dir = data_dir
        self.image_size = image_size
        self.scale = scale

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_file_name = sorted(os.path.join(self.data_dir, 'hr'))

    def __getitem__(self, item):
        file_name = self.image_file_name[item]
        high_resolution = Image.open(os.path.join(self.data_dir, 'hr', file_name)).convert('RGB')
        low_resolution = Image.open(os.path.join(self.data_dir, 'lr', file_name)).convert('RGB')

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