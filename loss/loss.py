import torch.nn as nn
import torch


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, discriminator_rf, discriminator_fr):
        Exr = torch.mean(torch.log(discriminator_rf).sum(-1).mul(-1)).mul(-1)
        Exf = torch.mean(torch.log(discriminator_fr)).mul(-1)
        return Exr + Exf


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, discriminator_rf, discriminator_fr):
        Exr = torch.mean(torch.log(discriminator_rf).sum(-1).mul(-1)).mul(-1)
        Exf = torch.mean(torch.log(discriminator_fr))
        return Exr + Exf
