import torch.utils.data
from PIL import Image
import os
from glob import glob
from torchvision import transforms


class Datasets(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size, scale):
        self.data_dir = data_dir
        self.image_size = image_size
        self.scale = scale

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(self.data_dir))

    def __getitem__(self, item):
        image = self.image_path[item]
        image = Image.open(image).convert('RGB')

        transform = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size // self.scale),
            transforms.ToTensor(),
        ])

        low_resolution = transform(image)
        hight_resolution = image

        image['lr'] = low_resolution
        image['hr'] = hight_resolution

        return image

    def __len__(self):
        len(self.image_path)
