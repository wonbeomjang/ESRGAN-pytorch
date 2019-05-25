from torch.utils.data import DataLoader
from dataloader.datasets import Datasets
from torch.utils.data.dataset import random_split
import torch


def get_loader(data_dir, image_size, scale, batch_size, sample_batch_size):
    dataset = Datasets(data_dir, image_size, scale)

    train_length = int(0.9 * len(dataset))
    test_length = len(dataset) - train_length

    train_dataset, test_dataset = random_split(dataset, (train_length, test_length))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=sample_batch_size,
                                              shuffle=False)
    return train_loader, test_loader
