from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms

class Datasets(Dataset):
    def __init__(self, data_dir, image_size, scale):
        self.data_dir = data_dir
        self.image_size = image_size
        self.scale = scale

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image = self.image_path[item]
        image = Image.open(image).convert('RGB')

        transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        transform_low_resolution = transforms.Compose([
            transforms.Resize(self.image_size // self.scale),
            transforms.ToTensor(),
        ])

        transform_high_resolution = transforms.Compose([
            # transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        image = transform(image)
        low_resolution = transform_low_resolution(image)
        high_resolution = transform_high_resolution(image)

        _image = {'lr': low_resolution, 'hr': high_resolution}

        return _image

    def __len__(self):
        return len(self.image_path)
