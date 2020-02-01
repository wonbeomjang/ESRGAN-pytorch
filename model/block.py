import torch.nn as nn
import torch


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nf + 0 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(nf + 1 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=True), nn.LeakyReLU())

        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.layer1 = ResidualDenseBlock(nf, gc)
        self.layer2 = ResidualDenseBlock(nf, gc)
        self.layer3 = ResidualDenseBlock(nf, gc, )
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


def upsample_block(nf, scale_factor=2):
    block = []
    for _ in range(scale_factor//2):
        block += [
            nn.Conv2d(nf, nf * (2 ** 2), 1),
            nn.PixelShuffle(2),
            nn.ReLU()
        ]

    return nn.Sequential(*block)

