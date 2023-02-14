import numpy as np
from torch import nn
from torch.nn import Conv2d, Module, Upsample

from models.stylegan2.model import EqualLinear


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleBlock1(Module):
    def __init__(self, in_c, out_c, in_s, out_s=None, is_noise=False):
        super(GradualStyleBlock1, self).__init__()
        self.out_c = out_c

        self.is_noise = is_noise
        if self.is_noise:
            modules = [Conv2d(in_c, 1, kernel_size=3, stride=1, padding=1)]
            self.convs = nn.Sequential(*modules)
            self.scale = nn.Sequential(*[Upsample(size=out_s, mode='bicubic'),
                                         nn.InstanceNorm2d(1)])
        else:
            num_pools = int(np.log2(in_s))
            modules = []
            modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU()]
            for i in range(num_pools - 1):
                modules += [
                    Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()
                ]
            self.convs = nn.Sequential(*modules)
            self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        if self.is_noise:
            # mid_s = int(np.sqrt(self.out_c))
            # x = x.view(-1, 1, mid_s, mid_s)
            x = self.scale(x)
        else:
            x = x.view(-1, self.out_c)
            x = self.linear(x)
        return x
