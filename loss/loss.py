import torch.nn as nn
from torchvision.models.vgg import vgg16


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()

        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:29]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()
        self.upsampler = nn.Upsample(size=224)

    def forward(self, high_resolution, fake_high_resolution):
        high_resolution = self.upsampler(high_resolution)
        fake_high_resolution = self.upsampler(fake_high_resolution)
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss
