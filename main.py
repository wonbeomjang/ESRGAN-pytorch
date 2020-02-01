from src.train import Trainer
from dataloader.dataloader import get_loader
import os
from config.config import get_config


def main(config):
    # make directory not existed
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    print(f"ESRGAN start")

    data_loader = get_loader(config.image_size, config.scale_factor, config.batch_size, config.sample_batch_size)
    trainer = Trainer(config, data_loader)
    trainer.train()


if __name__ == "__main__":
    config = get_config()
    main(config)
