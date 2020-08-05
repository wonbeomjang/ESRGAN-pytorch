import torch
from torch.optim.adam import Adam
from model.ESRGAN import ESRGAN
from model.Discriminator import Discriminator
import os
from glob import glob
import torch.nn as nn
from torchvision.utils import save_image
from loss.loss import PerceptualLoss


class Trainer:
    def __init__(self, config, data_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epoch = config.num_epoch
        self.epoch = config.epoch
        self.image_size = config.image_size
        self.data_loader = data_loader
        self.checkpoint_dir = config.checkpoint_dir
        self.batch_size = config.batch_size
        self.sample_dir = config.sample_dir
        self.nf = config.nf
        self.scale_factor = config.scale_factor

        if config.is_perceptual_oriented:
            self.lr = config.p_lr
            self.content_loss_factor = config.p_content_loss_factor
            self.perceptual_loss_factor = config.p_perceptual_loss_factor
            self.adversarial_loss_factor = config.p_adversarial_loss_factor
            self.decay_iter = config.p_decay_iter
        else:
            self.lr = config.g_lr
            self.content_loss_factor = config.g_content_loss_factor
            self.perceptual_loss_factor = config.g_perceptual_loss_factor
            self.adversarial_loss_factor = config.g_adversarial_loss_factor
            self.decay_iter = config.g_decay_iter

        self.build_model()
        self.optimizer_generator = Adam(self.generator.parameters(), lr=self.lr, betas=(config.b1, config.b2),
                                        weight_decay=config.weight_decay)
        self.optimizer_discriminator = Adam(self.discriminator.parameters(), lr=self.lr, betas=(config.b1, config.b2),
                                            weight_decay=config.weight_decay)

        self.lr_scheduler_generator = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_generator, self.decay_iter)
        self.lr_scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_discriminator, self.decay_iter)

    def train(self):
        total_step = len(self.data_loader)
        adversarial_criterion = nn.BCEWithLogitsLoss().to(self.device)
        content_criterion = nn.L1Loss().to(self.device)
        perception_criterion = PerceptualLoss().to(self.device)
        self.generator.train()
        self.discriminator.train()

        for epoch in range(self.epoch, self.num_epoch):
            if not os.path.exists(os.path.join(self.sample_dir, str(epoch))):
                os.makedirs(os.path.join(self.sample_dir, str(epoch)))

            for step, image in enumerate(self.data_loader):
                low_resolution = image['lr'].to(self.device)
                high_resolution = image['hr'].to(self.device)

                real_labels = torch.ones((high_resolution.size(0), 1)).to(self.device)
                fake_labels = torch.zeros((high_resolution.size(0), 1)).to(self.device)

                ##########################
                #   training generator   #
                ##########################
                self.optimizer_generator.zero_grad()
                fake_high_resolution = self.generator(low_resolution)

                score_real = self.discriminator(high_resolution)
                score_fake = self.discriminator(fake_high_resolution)
                discriminator_rf = score_real - score_fake.mean()
                discriminator_fr = score_fake - score_real.mean()

                adversarial_loss_rf = adversarial_criterion(discriminator_rf, fake_labels)
                adversarial_loss_fr = adversarial_criterion(discriminator_fr, real_labels)
                adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

                perceptual_loss = perception_criterion(high_resolution, fake_high_resolution)
                content_loss = content_criterion(fake_high_resolution, high_resolution)

                generator_loss = adversarial_loss * self.adversarial_loss_factor + \
                                 perceptual_loss * self.perceptual_loss_factor + \
                                 content_loss * self.content_loss_factor

                generator_loss.backward()
                self.optimizer_generator.step()

                ##########################
                # training discriminator #
                ##########################

                self.optimizer_discriminator.zero_grad()

                score_real = self.discriminator(high_resolution)
                score_fake = self.discriminator(fake_high_resolution.detach())
                discriminator_rf = score_real - score_fake.mean()
                discriminator_fr = score_fake - score_real.mean()

                adversarial_loss_rf = adversarial_criterion(discriminator_rf, real_labels)
                adversarial_loss_fr = adversarial_criterion(discriminator_fr, fake_labels)
                discriminator_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

                discriminator_loss.backward()
                self.optimizer_discriminator.step()

                self.lr_scheduler_generator.step()
                self.lr_scheduler_discriminator.step()
                if step % 1000 == 0:
                    print(f"[Epoch {epoch}/{self.num_epoch}] [Batch {step}/{total_step}] "
                          f"[D loss {discriminator_loss.item():.4f}] [G loss {generator_loss.item():.4f}] "
                          f"[adversarial loss {adversarial_loss.item() * self.adversarial_loss_factor:.4f}]"
                          f"[perceptual loss {perceptual_loss.item() * self.perceptual_loss_factor:.4f}]"
                          f"[content loss {content_loss.item() * self.content_loss_factor:.4f}]"
                          f"")
                    if step % 5000 == 0:
                        result = torch.cat((high_resolution, fake_high_resolution), 2)
                        save_image(result, os.path.join(self.sample_dir, str(epoch), f"SR_{step}.png"))

            torch.save(self.generator.state_dict(), os.path.join(self.checkpoint_dir, f"generator_{epoch}.pth"))
            torch.save(self.discriminator.state_dict(), os.path.join(self.checkpoint_dir, f"discriminator_{epoch}.pth"))

    def build_model(self):
        self.generator = ESRGAN(3, 3, 64, scale_factor=self.scale_factor).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            self.makedirs = os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            print(f"[!] No checkpoint in {self.checkpoint_dir}")
            return

        generator = glob(os.path.join(self.checkpoint_dir, f'generator_{self.epoch - 1}.pth'))
        discriminator = glob(os.path.join(self.checkpoint_dir, f'discriminator_{self.epoch - 1}.pth'))

        if not generator:
            print(f"[!] No checkpoint in epoch {self.epoch - 1}")
            return

        self.generator.load_state_dict(torch.load(generator[0]))
        self.discriminator.load_state_dict(torch.load(discriminator[0]))
