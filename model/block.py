import torch.nn as nn
import torch


def activation(act_type, inplace=True, negative_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace=inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(negative_slope=negative_slope)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=negative_slope)
    else:
        raise Exception(f"[!] Activation layer {act_type} is not found")
    return layer


def normalization(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batchnorm':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instancenorm':
        layer = nn.InstanceNorm2d(nc, affine=True)
    else:
        raise Exception(f"[!] Normalization layer {norm_type} is not found")
    return layer


def padding(pad_type, pad):
    pad_type = pad_type.lower()
    if pad == 0:
        return None
    if pad_type == 'reflection':
        layer = nn.ReflectionPad2d(pad)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(pad_type)
    elif pad_type == 'zero':
        layer = nn.ZeroPad2d(pad)
    else:
        raise Exception(f"[!] Padding layer {pad_type} is not found")
    return layer


def get_n_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) *(dilation - 1)
    pad = (kernel_size - 1) // 2
    return pad


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
               act_type='leakyrelu', pad_type='reflection', norm_type=None, negative_slope=0.2, n_prelu=1,
               inplace=True, n_padding=None):
    n_pad = n_padding if n_padding else get_n_padding(kernel_size, dilation)
    pad = padding(pad_type, n_pad) if pad_type else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
    norm = normalization(norm_type, out_channels) if norm_type else None
    act = activation(act_type, inplace=inplace, negative_slope=negative_slope, n_prelu=n_prelu) if act_type else None
    if (norm is None) and (act_type is None):
        return nn.Sequential(pad, conv)
    if pad_type is None:
        return nn.Sequential(conv, act)
    if norm is None:
        return nn.Sequential(pad, conv, act)
    else:
        return nn.Sequential(pad, conv, norm, act)


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, gc, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
                 res_scale=0.2, act_type='leakyrelu', last_act=None, pad_type='reflection', norm_type=None,
                 negative_slope=0.2, n_prelu=1, inplace=True):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = conv_block(in_channels + 0 * gc, gc, kernel_size, stride, dilation, groups,
                                 bias, act_type, pad_type, norm_type, negative_slope, n_prelu, inplace)
        self.layer2 = conv_block(in_channels + 1 * gc, gc, kernel_size, stride, dilation, groups,
                                 bias, act_type, pad_type, norm_type, negative_slope, n_prelu, inplace)
        self.layer3 = conv_block(in_channels + 2 * gc, gc, kernel_size, stride, dilation, groups,
                                 bias, act_type, pad_type, norm_type, negative_slope, n_prelu, inplace)
        self.layer4 = conv_block(in_channels + 3 * gc, gc, kernel_size, stride, dilation, groups,
                                 bias, act_type, pad_type, norm_type, negative_slope, n_prelu, inplace)
        self.layer5 = conv_block(in_channels + 4 * gc, in_channels, kernel_size, stride, dilation, groups,
                                 bias, last_act, pad_type, norm_type, negative_slope, n_prelu, inplace)

        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, gc, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
                 res_scale=0.2, act_type='leakyrelu', last_act=None, pad_type='reflection', norm_type=None,
                 negative_slope=0.2, n_prelu=1, inplace=True):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.layer1 = ResidualDenseBlock(in_channels, gc, kernel_size, stride, dilation, groups, bias, res_scale,
                                         act_type, last_act, pad_type, norm_type, negative_slope, n_prelu, inplace)
        self.layer2 = ResidualDenseBlock(in_channels, gc, kernel_size, stride, dilation, groups, bias, res_scale,
                                         act_type, last_act, pad_type, norm_type, negative_slope, n_prelu, inplace)
        self.layer3 = ResidualDenseBlock(in_channels, gc, kernel_size, stride, dilation, groups, bias, res_scale,
                                         act_type, last_act, pad_type, norm_type, negative_slope, n_prelu, inplace)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


def upsample_block(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
                 act_type='relu', pad_type='reflection', norm_type=None, negative_slope=0.2, n_prelu=1, inplace=True,
                 scale_facter=2, mode='nearest'):
        up_layer = nn.Upsample(scale_factor=scale_facter, mode=mode)
        conv = conv_block(in_channels, out_channels, kernel_size, stride, dilation, groups, bias, act_type,
                               pad_type, norm_type, negative_slope, n_prelu, inplace)
        return nn.Sequential(up_layer, conv)
