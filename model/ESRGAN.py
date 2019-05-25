from model.block import *


class ESRGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nf, gc=32, kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
                 res_scale=0.2, act_type='leakyrelu', last_act=None, pad_type='reflection', norm_type=None,
                 negative_slope=0.2, n_prelu=1, inplace=True, scale_facter=2, mode='nearest', n_basic_block=5):
        super(ESRGAN, self).__init__()

        self.conv1 = conv_block(in_channels, nf, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                       norm_type, negative_slope, n_prelu, inplace)

        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += ResidualInResidualDenseBlock(nf, gc, kernel_size, stride, dilation, groups,
                                                              bias, res_scale, act_type, last_act, pad_type, norm_type,
                                                              negative_slope, n_prelu, inplace)

        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                       norm_type, negative_slope, n_prelu, inplace)

        self.upsample = upsample_block(nf, nf, scale_facter=scale_facter)
        self.conv3 = conv_block(nf, nf, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                       norm_type, negative_slope, n_prelu, inplace)
        self.conv4 = conv_block(nf, out_channels, kernel_size, stride, dilation, groups, bias, act_type, pad_type,
                       norm_type, negative_slope, n_prelu, inplace)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.basic_block(x1)
        x3 = self.conv2(x2)
        x4 = self.upsample(x3 + x1)
        x5 = self.conv3(x4)
        x6 = self.conv4(x5)
        return x6
