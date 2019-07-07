from src.train import Trainer
from src.test import Tester
from dataloader.dataloader import get_loader
import os
from config.config import get_config
from util.util import download_url, reformat_file, unzip_tar_file, unzip_zip_file


def main(config):
    # make directory not existed
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)

    # download dataset
    if not os.listdir(config.data_dir):
        url = config.dataset_url
        for i, url in enumerate(url):
            zip_path = os.path.join(config.data_dir, url.split('/')[-1])
            download_url(url, zip_path)
            if zip_path.endswith('tar'):
                unzip_tar_file(zip_path, config.data_dir)
            if zip_path.endswith('zip'):
                unzip_zip_file(zip_path, config.data_dir)
        reformat_file(config.data_dir)

    print(f"ESRGAN start")

    data_loader, val_data_loader = get_loader(config.data_dir, config.image_size, config.scale_factor,
                                              config.batch_size, config.sample_batch_size)
    trainer = Trainer(config, data_loader)
    trainer.train()

    tester = Tester(config, val_data_loader)
    tester.test()


if __name__ == "__main__":
    config = get_config()
    main(config)
