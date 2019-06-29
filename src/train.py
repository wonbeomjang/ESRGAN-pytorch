import torch
from torch.optim.adam import Adam
from model.ESRGAN import ESRGAN
from model.Discriminator import Discriminator
import os
from glob import glob
from util.util import LRScheduler, LambdaLR
import torch.nn as nn
from torchvision.utils import save_image
from loss.loss import PerceptionLoss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

    elif classname.find('InstanceNorm') != -1:
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
        self.scale_factor = config.scale_factor
        self.b1 = config.b1
        self.b2 = config.b2
        self.weight_decay = config.weight_decay

        if config.is_perceptual_oriented:
            self.content_loss_factor = config.p_content_loss_factor
            self.perceptual_loss_factor = config.p_perceptual_loss_factor
            self.adversarial_loss_factor = config.p_adversarial_loss_factor
            self.decay_batch_size = config.p_decay_batch_size
        else:
            self.content_loss_factor = config.g_content_loss_factor
            self.perceptual_loss_factor = config.g_perceptual_loss_factor
            self.adversarial_loss_factor = config.g_adversarial_loss_factor
            self.decay_batch_size = config.g_decay_batch_size

        self.build_model()
        self.optimizer_generator = Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2),
                                        weight_decay=self.weight_decay)
        self.optimizer_discriminator = Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2),
                                            weight_decay=self.weight_decay)


        self.lr_scheduler_generator = torch.optim.lr_scheduler.LambdaLR(self.optimizer_generator,
                                                                        LambdaLR(self.num_epoch, self.epoch
                                                                                 , len(self.data_loader),
                                                                                 self.decay_batch_size))
        self.lr_scheduler_discriminator = torch.optim.lr_scheduler.LambdaLR(self.optimizer_discriminator,
                                                                            LambdaLR(self.num_epoch, self.epoch
                                                                                     , len(self.data_loader),
                                                                                     self.decay_batch_size)
                                                                            )

    def train(self):
        total_step = len(self.data_loader)
        adversarial_criterion = nn.BCELoss().to(self.device)
        content_criterion = nn.L1Loss().to(self.device)
        perception_criterion = PerceptionLoss().to(self.device)
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.epoch, self.num_epoch):
            if not os.path.exists(os.path.join(self.sample_dir, str(epoch))):
                os.makedirs(os.path.join(self.sample_dir, str(epoch)))

            for step, image in enumerate(self.data_loader):
                low_resolution = image['lr'].to(self.device)
                high_resolution = image['hr'].to(self.device)

                real_labels = torch.ones((high_resolution.size(0), *self.discriminator.output_shape)).to(self.device)
                fake_labels = torch.zeros((high_resolution.size(0), *self.discriminator.output_shape)).to(self.device)

                ##########################
                #   training generator   #
                ##########################
                fake_high_resolution = self.generator(low_resolution)
                discriminator_rf = self.discriminator(high_resolution, fake_high_resolution)
                discriminator_fr = self.discriminator(fake_high_resolution, high_resolution)

                adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
                adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
                adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

                perception_loss = perception_criterion(high_resolution, fake_high_resolution)
                content_loss = content_criterion(high_resolution, fake_high_resolution)

                generator_loss = adversarial_loss * self.adversarial_loss_factor + \
                                 perception_loss * self.perceptual_loss_factor + \
                                 content_loss * self.content_loss_factor

                self.optimizer_generator.zero_grad()
                generator_loss.backward(retain_graph=True)
                self.optimizer_generator.step()

                ##########################
                # training discriminator #
                ##########################
                discriminator_rf = self.discriminator(high_resolution, fake_high_resolution)
                discriminator_fr = self.discriminator(fake_high_resolution, high_resolution)

                adversarial_loss_rf = adversarial_criterion(discriminator_rf, real_labels)
                adversarial_loss_fr = adversarial_criterion(discriminator_fr, fake_labels)
                discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

                self.optimizer_discriminator.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                self.optimizer_discriminator.step()

                if step % 50 == 0:
                    print(f"[Epoch {epoch}/{self.num_epoch}] [Batch {step}/{total_step}] "
                          f"[D loss {discriminator_loss.item()}] [G loss {generator_loss.item()}]")
                    if step % 100 == 0:
                        result = torch.cat((high_resolution, fake_high_resolution), 2)
                        save_image(result, os.path.join(self.sample_dir, str(epoch), f"SR_{step}.png"))

            torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, f"generator_{epoch}.pth"))
            torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, f"discriminator_{epoch}.pth"))

            self.lr_scheduler_generator.step()
            self.lr_scheduler_discriminator.step()

    def build_model(self):
        self.generator = ESRGAN(3, 3, 64, scale_factor=self.scale_factor).to(self.device)
        self.discriminator = Discriminator(self.image_size).to(self.device)
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            print(f"[!] No checkpoint in {self.checkpoint_dir}")
            return

        generator = glob(os.path.join(self.checkpoint_dir, f'generator_{self.epoch - 1}.pth'))
        discriminator = glob(os.path.join(self.checkpoint_dir, f'discriminator_{self.epoch - 1}.pth'))

        if not generator:
            print(f"[!] No checkpoint in epoch {self.epoch - 1}")
            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)
            return

        self.generator.load_state_dict(torch.load(generator[0]))
        self.discriminator.load_state_dict(torch.load(discriminator[0]))
