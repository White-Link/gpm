# Copyright 2023 Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickaël Chen, Alain Rakotomamonjy

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Adapted code from PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# (file `src/models/resnet.py`).

# Originally licensed under the MIT License with the following notice.

# The MIT License (MIT)

# PyTorch StudioGAN:
# Copyright (c) 2020 MinGuk Kang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import torch

import torch.nn.functional as F

from torch.nn.utils.parametrizations import spectral_norm as sn

from gpm.networks.activations import ActivationDict, activation_factory
from gpm.networks.conv.utils import make_conv_block
from gpm.networks.utils import DecoderDict, EncoderDict


class SelfAttention(torch.nn.Module):
    """
    https://github.com/voletiv/self-attention-GAN-pytorch
    MIT License

    Copyright (c) 2019 Vikram Voleti

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, in_channels: int, spectral_norm: bool, additional_input: int | None = None):
        super().__init__()
        self.in_channels = in_channels

        self.conv1x1_theta = make_conv_block(torch.nn.Conv2d(in_channels, in_channels // 8, 1, stride=1, padding=0,
                                                             bias=False),
                                             None, spectral_norm=spectral_norm, additional_input=additional_input)
        self.conv1x1_phi = make_conv_block(torch.nn.Conv2d(in_channels, in_channels // 8, 1, stride=1, padding=0,
                                                           bias=False),
                                           None, spectral_norm=spectral_norm, additional_input=additional_input)
        self.conv1x1_g = make_conv_block(torch.nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0,
                                                         bias=False),
                                         None, spectral_norm=spectral_norm, additional_input=additional_input)
        self.conv1x1_attn = make_conv_block(torch.nn.Conv2d(in_channels // 2, in_channels, 1, stride=1, padding=0,
                                                            bias=False),
                                            None, spectral_norm=spectral_norm, additional_input=additional_input)
        self.maxpool = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.sigma = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is not None:
            c_args = [c]
        else:
            c_args = []
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x, *c_args)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.conv1x1_phi(x, *c_args)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x, *c_args)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.conv1x1_attn(attn_g, *c_args)
        return x + self.sigma * attn_g


class GenBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: ActivationDict, batch_norm: bool,
                 spectral_norm: bool):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(in_channels) if batch_norm else torch.nn.Identity()
        self.activation = activation_factory(activation)
        self.conv2d0 = make_conv_block(torch.nn.ConvTranspose2d(in_channels, out_channels, 1, stride=1, padding=0),
                                       None, spectral_norm=spectral_norm)
        self.conv2d1 = make_conv_block(torch.nn.ConvTranspose2d(in_channels, out_channels, 3, stride=1, padding=1),
                                       activation, batch_norm=batch_norm, spectral_norm=spectral_norm)
        self.conv2d2 = make_conv_block(torch.nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1),
                                       None, spectral_norm=spectral_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        x = self.bn(x)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


class DiscOptBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: ActivationDict, batch_norm: bool,
                 spectral_norm: bool, additional_input: int | None = None):
        super().__init__()
        self.conv2d0 = make_conv_block(torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0), None,
                                       spectral_norm=spectral_norm, additional_input=additional_input)
        self.conv2d1 = make_conv_block(torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1), activation,
                                       batch_norm=batch_norm, spectral_norm=spectral_norm,
                                       additional_input=additional_input)
        self.conv2d2 = make_conv_block(torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                                       torch.nn.AvgPool2d(2), spectral_norm=spectral_norm,
                                       additional_input=additional_input)
        self.bn = torch.nn.BatchNorm2d(in_channels) if batch_norm else torch.nn.Identity()
        self.activation = activation_factory(activation)
        self.average_pooling = torch.nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is not None:
            c_args = [c]
        else:
            c_args = []
        x0 = x
        x = self.conv2d1(x, *c_args)
        x = self.conv2d2(x, *c_args)
        x0 = self.average_pooling(x0)
        x0 = self.bn(x0)
        x0 = self.conv2d0(x0, *c_args)
        out = x + x0
        return out


class DiscBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: ActivationDict, batch_norm: bool,
                 spectral_norm: bool, downsample=True, additional_input: int | None = None):
        super(DiscBlock, self).__init__()
        self.downsample = downsample
        self.ch_mismatch = in_channels != out_channels

        if self.ch_mismatch or downsample:
            self.conv2d0 = make_conv_block(torch.nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0),
                                           torch.nn.AvgPool2d(2) if downsample else None, spectral_norm=spectral_norm,
                                           additional_input=additional_input)
            self.bn0 = torch.nn.BatchNorm2d(in_channels) if batch_norm else torch.nn.Identity()

        self.conv2d1 = make_conv_block(torch.nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1), activation,
                                       batch_norm=batch_norm, spectral_norm=spectral_norm,
                                       additional_input=additional_input)
        self.conv2d2 = make_conv_block(torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                                       torch.nn.AvgPool2d(2) if downsample else None, spectral_norm=spectral_norm,
                                       additional_input=additional_input)
        self.bn1 = torch.nn.BatchNorm2d(in_channels) if batch_norm else torch.nn.Identity()

        self.activation = activation_factory(activation)
        self.average_pooling = torch.nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is not None:
            c_args = [c]
        else:
            c_args = []

        x0 = x
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x, *c_args)
        x = self.conv2d2(x, *c_args)

        if self.downsample or self.ch_mismatch:
            x0 = self.bn0(x0)
            x0 = self.conv2d0(x0, *c_args)
        out = x + x0
        return out


class ResNetDecoderDict(DecoderDict):
    """
    Parameter dictionary for ResNet generators.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'width', 'batch_norm', 'spectral_norm', 'activation'}.issubset(self.keys())
        self.width: int  # Hidden width multipler
        self.batch_norm: bool  # Whether to include batch norm
        self.spectral_norm: bool  # Whether to include spectral norm
        self.attn_layers: list[int]  # List of layers IDs where attention should be added (default, no attention)
        if 'attn_layers' not in self:
            self.attn_layers = []
        self.activation: ActivationDict = ActivationDict(self.activation)  # Activation


class ResNetDecoder(torch.nn.Module):
    """
    ResNet generator.
    """

    def __init__(self, in_dim: int, out_channels: int, out_size: int, config: ResNetDecoderDict):
        super().__init__()
        g_conv_dim = config.width
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2,
                    g_conv_dim]
        }
        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim,
                    g_conv_dim]
        }
        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = in_dim
        self.in_dims = g_in_dims_collection[str(out_size)]
        self.out_dims = g_out_dims_collection[str(out_size)]
        self.bottom = bottom_collection[str(out_size)]
        self.num_blocks = len(self.in_dims)

        self.linear0 = torch.nn.Linear(self.z_dim, self.in_dims[0] * self.bottom * self.bottom, bias=True)
        if config.spectral_norm:
            self.linear0 = sn(self.linear0)

        blocks = []
        for index in range(self.num_blocks):
            blocks.append(GenBlock(self.in_dims[index], self.out_dims[index], config.activation, config.batch_norm,
                                   config.spectral_norm))
            if index + 1 in config.attn_layers:
                blocks.append(SelfAttention(self.out_dims[index], config.spectral_norm))

        self.blocks = torch.nn.Sequential(*blocks)

        self.bn4 = torch.nn.BatchNorm2d(self.out_dims[-1]) if config.batch_norm else torch.nn.Identity()
        self.activation = activation_factory(config.activation)
        self.conv2d5 = make_conv_block(torch.nn.ConvTranspose2d(self.out_dims[-1], out_channels, 3, stride=1,
                                                                padding=1),
                                       None, spectral_norm=config.spectral_norm)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        act = self.linear0(z)
        act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
        act = self.blocks(act)
        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        return act


class ResNetEncoderDict(EncoderDict):
    """
    Parameter dictionary for ResNet discriminators.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'width', 'batch_norm', 'spectral_norm', 'activation'}.issubset(self.keys())
        self.width: int  # Hidden width multipler
        self.batch_norm: bool  # Whether to include batch norm
        self.spectral_norm: bool  # Whether to include spectral norm
        self.attn_layers: list[int]  # List of layers IDs where attention should be added (default, no attention)
        if 'attn_layers' not in self:
            self.attn_layers = []
        self.activation: ActivationDict = ActivationDict(self.activation)  # Activation


class ResNetEncoder(torch.nn.Module):
    """
    ResNet discriminator.

    Can optionnally take as input an additional tensor with is preprocessed by an MLP, then used to modulate
    convolutional layers outputs.
    """

    def __init__(self, in_channels: int, in_size: int, out_dim: int, config: ResNetEncoderDict,
                 additional_input: int | None = None):
        super().__init__()
        d_conv_dim = config.width
        d_in_dims_collection = {
            "32": [in_channels] + [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [in_channels] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8],
            "128": [in_channels] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "256": [in_channels] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8,
                                    d_conv_dim * 16],
            "512": [in_channels] + [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8,
                                    d_conv_dim * 8, d_conv_dim * 16]
        }
        d_out_dims_collection = {
            "32": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "128": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "256": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16,
                    d_conv_dim * 16],
            "512": [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8,
                    d_conv_dim * 16, d_conv_dim * 16]
        }
        d_down = {
            "32": [True, True, False, False],
            "64": [True, True, True, True, False],
            "128": [True, True, True, True, True, False],
            "256": [True, True, True, True, True, True, False],
            "512": [True, True, True, True, True, True, True, False]
        }

        # Preprocessing MLP of 2nd output
        if additional_input is not None and additional_input > 0:
            self.c_mlp = torch.nn.Sequential(
                torch.nn.Linear(additional_input, additional_input * 2),
                torch.nn.SiLU(inplace=True),
                torch.nn.Linear(additional_input * 2, additional_input * 2),
                torch.nn.SiLU(inplace=True),
            )
            additional_input = additional_input * 2 if additional_input > 0 else None
        else:
            self.c_mlp = torch.nn.Identity()

        self.in_dims = d_in_dims_collection[str(in_size)]
        self.out_dims = d_out_dims_collection[str(in_size)]
        down = d_down[str(in_size)]

        blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                blocks.append(DiscOptBlock(self.in_dims[index], self.out_dims[index], config.activation,
                                           config.batch_norm, config.spectral_norm, additional_input=additional_input))
            else:
                blocks.append(DiscBlock(self.in_dims[index], self.out_dims[index], config.activation,
                                        config.batch_norm, config.spectral_norm, downsample=down[index],
                                        additional_input=additional_input))
            if index + 1 in config.attn_layers:
                blocks.append(SelfAttention(self.out_dims[index], config.spectral_norm,
                                            additional_input=additional_input))
        self.blocks = torch.nn.ModuleList(blocks)
        self.activation = activation_factory(config.activation)
        self.linear1 = torch.nn.Linear(self.out_dims[-1], out_dim)
        if config.spectral_norm:
            self.linear1 = sn(self.linear1)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        # Preprocessing of 2nd output
        if c is not None:
            c = self.c_mlp(c)
            c_args = [c]
        else:
            c_args = []
        h = x
        for block in self.blocks:
            h = block(h, *c_args)
        h = self.activation(h)
        h = torch.sum(h, dim=[2, 3])
        return self.linear1(h)
