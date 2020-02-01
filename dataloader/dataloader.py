from torch.utils.data import DataLoader
from dataloader.datasets import Datasets
import torch


def get_loader(image_size, scale, batch_size, sample_batch_size):
    train_dataset = Datasets(image_size, scale)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
