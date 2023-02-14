from models.stylegan2.model import StyledConv, ToRGB, ConstantInput
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from torchvision.models.resnet import resnet34

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.encoders.map2style import GradualStyleBlock


class BackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet IRSE-50 backbone.
    Note this class is designed to be used for the human facial domain.
    """
    def __init__(self, num_layers, mode='ir', n_styles=18, opts=None):
        super(BackboneEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16)
            self.styles.append(style)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        latents = []
        for j in range(self.style_count):
            latents.append(self.styles[j](x))
        out = torch.stack(latents, dim=1)
        return out


class ResNetBackboneEncoder(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet34 backbone.
    """
    def __init__(self, n_styles=18, opts=None):
        super(ResNetBackboneEncoder, self).__init__()
        stride = 2
        if opts.output_size == 128:
            stride = 1 

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.input_ch = opts.input_ch
        for i in range(self.style_count):
            if self.input_ch == -1:
                style = GradualStyleBlock(512, 512, 16)
            else:
                style = GradualStyleBlock(512, 128, 16)
            self.styles.append(style)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        latents = []

        for j in range(self.style_count):
            sty = self.styles[j](x)
            if self.input_ch == -1:
                latents.append(sty)
            else:
                latents.append(sty.repeat(1, 4))
        out = torch.stack(latents, dim=1)
        return out


class ResNetBackboneEncoder0(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet34 backbone.
    """

    def __init__(self, n_styles=18, opts=None):
        super(ResNetBackboneEncoder0, self).__init__()

        self.conv1 = nn.Conv2d(
            opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(
            pretrained=False, norm_layer=nn.InstanceNorm2d)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        dim = 256
        self.styles = nn.ModuleList()
        self.noises = nn.ModuleList()
        self.style_count = n_styles
        for i in range(self.style_count):
            style = GradualStyleBlock(512, 512, 16, None, False)
            self.styles.append(style)
            noise = GradualStyleBlock(512, 256, 16, dim, True)
            self.noises.append(noise)
            if i % 2 == 1:
                dim //= 2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        styles, noises = [], []
        for j in range(self.style_count):
            style = self.styles[j](x)
            styles = [style] + styles
            noise = self.noises[j](x)
            noises = [noise] + noises

        styles = torch.stack(styles, dim=1)
        noises = noises[1:]
        return styles, noises


class ResNetBackboneEncoder1(Module):
    """
    The simpler backbone architecture used by ReStyle where all style vectors are extracted from the final 16x16 feature
    map of the encoder. This classes uses the simplified architecture applied over an ResNet34 backbone.
    """

    def __init__(self, n_styles=18, opts=None, style_dim=512,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1]):
        super(ResNetBackboneEncoder1, self).__init__()

        self.conv0 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.InstanceNorm2d(64, affine=True)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(
            pretrained=False, norm_layer=nn.InstanceNorm2d)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        dim = 256
        self.style0 = nn.ModuleList()
        self.style1 = nn.ModuleList()
        self.style2 = nn.ModuleList()
        self.style_count = n_styles // 2
        for i in range(self.style_count):
            sty0 = GradualStyleBlock(512, style_dim, 16)
            self.style0.append(sty0)
            self.style1.append(nn.Upsample(size=(dim, dim)))
            sty2 = nn.Sequential(*[nn.Conv2d(512, 1, 3, 1, 1), 
                                   nn.Upsample(size=(dim, dim))])
            self.style2.append(sty2)
            dim //= 2


        # self.channels = {
        #     4: 512,
        #     8: 512,
        #     16: 512,
        #     32: 512,
        #     64: 256 * channel_multiplier,
        #     128: 128 * channel_multiplier,
        #     256: 64 * channel_multiplier,
        # }

        # self.input = ConstantInput(self.channels[4])
        # self.conv1 = StyledConv(
        #     self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        # )
        # self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False, out_chn=1)

        # self.convs = nn.ModuleList()
        # self.to_rgbs = nn.ModuleList()
        # self.norms = nn.ModuleList()
        # in_channel = self.channels[4]
        # for i in range(3, n_styles // 2 + 2):
        #     out_channel = self.channels[2 ** i]

        #     self.convs.append(
        #         StyledConv(
        #             in_channel,
        #             out_channel,
        #             3,
        #             style_dim,
        #             upsample=True,
        #             blur_kernel=blur_kernel,
        #         )
        #     )

        #     self.convs.append(
        #         StyledConv(
        #             out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
        #         )
        #     )

        #     self.to_rgbs.extend([ToRGB(out_channel, style_dim, out_chn=1), 
        #                          ToRGB(out_channel, style_dim, out_chn=1)])
        #     self.norms.extend([nn.InstanceNorm2d(out_channel), 
        #                        nn.InstanceNorm2d(out_channel)])

        #     in_channel = out_channel

    def forward(self, x):
        x, msk0 = x[:,:3].clone(), x[:, 3].clone()
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.body(x)

        style0, style1 = list(), list()
        msk = (msk0.unsqueeze(1)) * 255.
        for i in range(self.style_count):
            sty0 = self.style0[i](x)
            style0 = [sty0, sty0] + style0
            msk = self.style1[i](msk)
            nos = self.style2[i](x)
            style1 = [msk + nos, msk + nos] + style1
        style0 = torch.stack(style0, dim=1)
            
            # sty1 = self.style1[i](x)
            # style1.append(sty1.view(sty1.shape[0], -1))
        # style1 = self.style1(x).view(x.shape[0], -1, 512)

        # noises = list()
        # out = self.input(style1)
        # out = self.conv1(out, style1[:, 0])
        # noise = self.to_rgb1(out, style1[:, 1])
        # # print(noise.shape)
        # noises.append(noise)

        # i = 1
        # for conv1, conv2, to_rgb1, to_rgb2 in zip(self.convs[::2], self.convs[1::2], self.to_rgbs[::2], self.to_rgbs[1::2]):
        #     out = conv1(out, style1[:, i])
        #     noise = to_rgb1(out, style1[:, i + 1])
        #     # print(noise.shape)
        #     noises.append(noise)

        #     out = conv2(out, style1[:, i + 1])
        #     noise = to_rgb2(out, style1[:, i + 2])
        #     # print(noise.shape)
        #     noises.append(noise)
        #     i += 2

        return style0, style1[1:]
