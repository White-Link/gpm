# Copyright 2023 Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac,
# Mickaël Chen, Alain Rakotomamonjy

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

from gpm.networks.activations import ActivationDict
from gpm.networks.conv.utils import make_conv_block
from gpm.networks.utils import EncoderDict, DecoderDict


class DCGANEncoderDict(EncoderDict):
    """
    Parameter dictionary for DCGAN discriminators.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'width', 'batch_norm', 'spectral_norm', 'first_bias', 'activation'}.issubset(self.keys())
        self.width: int  # Hidden width multipler
        self.batch_norm: bool  # Whether to include batch norm
        self.spectral_norm: bool  # Whether to include spectral norm
        self.first_bias: bool  # Whether to add bias to the first convolution
        self.activation: ActivationDict = ActivationDict(self.activation)  # Activation


class DCGANDecoderDict(DecoderDict):
    """
    Parameter dictionary for DCGAN generators.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'width', 'batch_norm', 'spectral_norm', 'activation'}.issubset(self.keys())
        self.width: int  # Hidden width multipler
        self.batch_norm: bool  # Whether to include batch norm
        self.spectral_norm: bool  # Whether to include spectral norm
        self.activation: ActivationDict = ActivationDict(self.activation)  # Activation


class DCGANEncoder(torch.nn.Module):
    """
    DCGAN discriminator.

    Adpated the first layer to optionnally take as input 32x32 images.

    Can optionnally take as input an additional tensor with is preprocessed by an MLP, then used to modulate
    convolutional layers outputs.
    """
    def __init__(self, in_channels: int, in_size: int, out_dim: int, config: DCGANEncoderDict,
                 additional_input: int | None = None):
        super().__init__()
        self.out_dim = out_dim
        # Preprocessing MLP of 2nd output
        if additional_input is not None and additional_input > 0:
            self.c_mlp = torch.nn.Sequential(
                torch.nn.Linear(additional_input, additional_input * 2),
                torch.nn.SiLU(inplace=True),
                torch.nn.Linear(additional_input * 2, additional_input * 2),
                torch.nn.SiLU(inplace=True)
            )
            additional_input = additional_input * 2 if additional_input > 0 else None
        else:
            self.c_mlp = torch.nn.Identity()
        bias = not config.batch_norm
        if in_size == 64:
            first_conv = make_conv_block(torch.nn.Conv2d(in_channels, config.width, 4, 2, 1, bias=config.first_bias),
                                         config.activation, spectral_norm=config.spectral_norm,
                                         additional_input=additional_input)
        else:
            first_conv = make_conv_block(torch.nn.Conv2d(in_channels, config.width, 3, 1, 1, bias=config.first_bias),
                                         config.activation, spectral_norm=config.spectral_norm,
                                         additional_input=additional_input)
        self.conv = torch.nn.ModuleList([
            first_conv,
            make_conv_block(torch.nn.Conv2d(config.width, config.width * 2, 4, 2, 1, bias=bias), config.activation,
                            batch_norm=config.batch_norm, spectral_norm=config.spectral_norm,
                            additional_input=additional_input),
            make_conv_block(torch.nn.Conv2d(config.width * 2, config.width * 4, 4, 2, 1, bias=bias), config.activation,
                            batch_norm=config.batch_norm, spectral_norm=config.spectral_norm,
                            additional_input=additional_input),
            make_conv_block(torch.nn.Conv2d(config.width * 4, config.width * 8, 4, 2, 1, bias=bias), config.activation,
                            batch_norm=config.batch_norm, spectral_norm=config.spectral_norm,
                            additional_input=additional_input),
        ])
        self.last_conv = make_conv_block(torch.nn.Conv2d(config.width * 8, self.out_dim, 4, 1, 0, bias=False),
                                         torch.nn.Flatten(), spectral_norm=config.spectral_norm,
                                         additional_input=additional_input)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        skips = []
        h = x
        # Preprocessing of 2nd output
        if c is not None:
            c = self.c_mlp(c)
        for layer in self.conv:
            if c is None:
                h = layer(h)
            else:
                h = layer(h, c)
            skips.append(h)
        if c is None:
            h = self.last_conv(h)
        else:
            h = self.last_conv(h, c)
        return h


class DCGANDecoder(torch.nn.Module):
    def __init__(self, in_dim: int, out_channels: int, out_size: int, config: DCGANDecoderDict, skip: bool = False):
        super().__init__()
        coef = 2 if skip else 1
        bias = not config.batch_norm
        self.first_upconv = make_conv_block(torch.nn.ConvTranspose2d(in_dim, config.width * 8, 4, 1, 0, bias=bias),
                                            config.activation, batch_norm=config.batch_norm,
                                            spectral_norm=config.spectral_norm)
        if out_size == 64:
            last_conv = make_conv_block(torch.nn.ConvTranspose2d(config.width * coef, out_channels, 4, 2, 1,
                                                                 bias=False),
                                        None, spectral_norm=config.spectral_norm)
        else:
            last_conv = make_conv_block(torch.nn.ConvTranspose2d(config.width * coef, out_channels, 3, 1, 1,
                                                                 bias=False),
                                        None, spectral_norm=config.spectral_norm)
        self.conv = torch.nn.ModuleList([
            make_conv_block(torch.nn.ConvTranspose2d(config.width * 8 * coef, config.width * 4, 4, 2, 1, bias=bias),
                            config.activation, batch_norm=config.batch_norm, spectral_norm=config.spectral_norm),
            make_conv_block(torch.nn.ConvTranspose2d(config.width * 4 * coef, config.width * 2, 4, 2, 1, bias=bias),
                            config.activation, batch_norm=config.batch_norm, spectral_norm=config.spectral_norm),
            make_conv_block(torch.nn.ConvTranspose2d(config.width * 2 * coef, config.width, 4, 2, 1, bias=bias),
                            config.activation, batch_norm=config.batch_norm, spectral_norm=config.spectral_norm),
            last_conv
        ])

    def forward(self, z: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        h = self.first_upconv(z.view(*z.shape, 1, 1))
        for i, layer in enumerate(self.conv):
            if skip is not None:
                h = torch.cat([h, skip[i]], 1)
            h = layer(h)
        return h
