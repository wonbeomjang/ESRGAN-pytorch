import torch.nn as nn
import torch
from torchvision.models.vgg import vgg16
from torch.nn.modules.loss import _WeightedLoss


class GeneratorLoss(_WeightedLoss):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False

        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, discriminator_rf, discriminator_fr, high_resolution, fake_high_resolution):

        perception_loss = self.l1_loss(self.loss_network(fake_high_resolution), self.loss_network(high_resolution))

        Exr = torch.mean(torch.log(discriminator_rf.sum(-1)).mul(-1)).mul(-1)
        Exf = torch.mean(torch.log(discriminator_fr)).mul(-1)
        adversarial_loss = Exr + Exf

        image_loss = self.l1_loss(fake_high_resolution, high_resolution)
        return adversarial_loss + perception_loss + 0.01 * image_loss


class DiscriminatorLoss(_WeightedLoss):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()

    def forward(self, discriminator_rf, discriminator_fr):
        Exr = torch.mean(torch.log(discriminator_rf)).mul(-1)
        Exf = torch.mean(torch.log(discriminator_fr.sum(-1)).mul(-1)).mul(-1)
        adversarial_loss = Exr + Exf
        return adversarial_loss
