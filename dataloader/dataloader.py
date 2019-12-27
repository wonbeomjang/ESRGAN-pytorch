from torch.utils.data import DataLoader
from dataloader.datasets import TrainDatasets, TestDataset
import torch


def get_loader(data_dir, image_size, scale, batch_size, sample_batch_size):
    train_dataset = TrainDatasets(data_dir, image_size, scale)
    test_dataset = TestDataset(data_dir, image_size, scale)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=sample_batch_size,
                                              shuffle=False)
    return train_loader, test_loader
