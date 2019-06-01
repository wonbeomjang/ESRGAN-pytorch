import torch
from loss.loss import GeneratorLoss, DiscriminatorLoss
from torch.optim.adam import Adam
from model.ESRGAN import ESRGAN
from model.Discriminator import Discriminator
import os
from glob import glob
from torch.autograd import Variable
from torchvision.utils import save_image


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Trainer:
    def __init__(self, config, data_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epoch = config.num_epoch
        self.epoch = config.epoch
        self.image_size = config.image_size
        self.data_loader = data_loader
        self.checkpoint_dir = config.checkpoint_dir
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.sample_dir = config.sample_dir
        self.nf = config.nf
        self.Tensor = torch.cuda.is_available()
        self.scale_factor = config.scale_factor
        self.build_model()

    def train(self):
        total_step = len(self.data_loader)
        optimizer_generator = Adam(self.generator.parameters(), lr=self.lr)
        optimizer_discriminator = Adam(self.discriminator.parameters(), lr=self.lr)
        discriminator_criterion = DiscriminatorLoss().to(self.device)
        generator_criterion = GeneratorLoss().to(self.device)
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.epoch, self.num_epoch):
            if not os.path.exists(os.path.join(self.sample_dir, str(epoch))):
                os.makedirs(os.path.join(self.sample_dir, str(epoch)))

            for step, image in enumerate(self.data_loader):
                low_resolution = image['lr'].to(self.device)
                high_resolution = image['hr'].to(self.device)


                # update generator
                fake_high_resolution = self.generator(low_resolution)
                discriminator_rf = self.discriminator(high_resolution, fake_high_resolution)
                discriminator_fr = self.discriminator(fake_high_resolution, high_resolution)
                generator_loss = generator_criterion(discriminator_rf, discriminator_fr)

                optimizer_generator.zero_grad()
                generator_loss.backward()
                optimizer_generator.step()

                # update discriminator
                discriminator_rf = self.discriminator(high_resolution, fake_high_resolution)
                discriminator_fr = self.discriminator(fake_high_resolution, high_resolution)

                discriminator_loss = discriminator_criterion(discriminator_rf, discriminator_fr)

                optimizer_discriminator.zero_grad()
                discriminator_loss.backward()
                optimizer_discriminator.step()

                if step % 10 == 0:
                    print(f"[Epoch {epoch}/{self.num_epoch}] [Batch {step}/{total_step}] "
                          f"[D loss {discriminator_loss}] [G loss {generator_loss}]")
                    if step % 50 == 0:
                        result = torch.cat((high_resolution, fake_high_resolution), 2)
                        save_image(result, os.path.join(self.sample_dir, str(epoch), f"SR_{step}.png"))

            torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, str(epoch),
                                                                 f"generator_{epoch}.pth"))
            torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, str(epoch),
                                                                 f"discriminator_{epoch}.pth"))

    def build_model(self):
        self.generator = ESRGAN(3, 3, 64, scale_facter=self.scale_factor).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            print(f"[!] No checkpoint in {self.checkpoint_dir}")
            return

        generator = glob(
            os.path.join(self.checkpoint_dir, f'ESRGAN_{self.epoch - 1}.pth'))
        discriminator = glob(
            os.path.join(self.checkpoint_dir, f'ESRGAN_{self.epoch - 1}.pth'))

        if not generator:
            print(f"[!] No checkpoint in epoch {self.epoch - 1}")
            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)
            return

        self.generator.load_state_dict(torch.load(generator[0]))
        self.discriminator.load_state_dict(torch.load(discriminator[0]))
