import functools
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil


class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16):
        super(MSRResNet, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = mutil.make_layer(basic_block, nb)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_first, self.HRconv, self.conv_last], 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        return out
