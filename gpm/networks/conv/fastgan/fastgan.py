# Copyright (C) 2023 Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac,
# Mickaël Chen, Alain Rakotomamonjy

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Adapted code from https://github.com/odegeasslbc/FastGAN-pytorch (file `models.py`).

# Initially released under the GNU GPL license and redistributed under the same license (see attached file).
# The original work did not include any copyright notice.
# Changes:
#  - Removed unused module definitions.
#  - Adapted the code to the coding style of this program (typing, spacing, naming, etc.).
#  - Added a modulation of many convolutional layers by a time-dependent vector.
#  - Adapted the generator and discriminator architectures to 32x32 and 64x64 images.


import torch
from gpm.networks.activations import ActivationDict

from gpm.networks.conv.utils import make_conv_block
from gpm.networks.utils import DecoderDict, EncoderDict, SequentialTuple


class PixelNorm(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class GLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1, x_2 = x.chunk(chunks=2, dim=1)
        return x_1 * torch.sigmoid(x_2)


class SEBlock(torch.nn.Module):
    def __init__(self, ch_in: int, ch_out: int, bias: bool, spectral_norm: bool, additional_input: int | None = None):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(4)
        self.conv_1 = make_conv_block(torch.nn.Conv2d(ch_in, ch_out, 4, stride=1, padding=0, bias=bias),
                                      torch.nn.SiLU(inplace=True), spectral_norm=spectral_norm,
                                      additional_input=additional_input)
        self.conv_2 = make_conv_block(torch.nn.Conv2d(ch_out, ch_out, 1, stride=1, padding=0, bias=bias),
                                      torch.nn.Sigmoid(), spectral_norm=spectral_norm,
                                      additional_input=additional_input)

    def forward(self, feat_small: torch.Tensor, feat_big: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is None:
            main = self.conv_2(self.conv_1(self.pool(feat_small)))
        else:
            main = self.conv_2(self.conv_1(self.pool(feat_small), c), c)
        return feat_big * main


class InitLayer(torch.nn.Module):
    def __init__(self, nz: int, channel: int, bias: bool, batch_norm: bool, spectral_norm: bool):
        super().__init__()
        self.init = torch.nn.Sequential(make_conv_block(torch.nn.ConvTranspose2d(nz, channel * 2, 4, stride=1,
                                                                                 padding=0, bias=bias),
                                                        GLU(), batch_norm=batch_norm, spectral_norm=spectral_norm))

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def up_block(in_planes: int, out_planes: int, bias: bool, batch_norm: bool, spectral_norm: bool) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode='nearest'),
                               make_conv_block(torch.nn.Conv2d(in_planes, out_planes * 2, 3, stride=1, padding=1,
                                                               bias=bias),
                                               GLU(), batch_norm=batch_norm, spectral_norm=spectral_norm))


def up_block_comp(in_planes: int, out_planes: int, bias: bool, batch_norm: bool,
                  spectral_norm: bool) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Upsample(scale_factor=2, mode='nearest'),
                               make_conv_block(torch.nn.Conv2d(in_planes, out_planes * 2, 3, stride=1, padding=1,
                                                               bias=bias),
                                               GLU(), batch_norm=batch_norm, spectral_norm=spectral_norm),
                               make_conv_block(torch.nn.Conv2d(out_planes, out_planes * 2, 3, stride=1, padding=1,
                                                               bias=bias),
                                               GLU(), batch_norm=batch_norm, spectral_norm=spectral_norm))


class DownBlockComp(torch.nn.Module):
    def __init__(self, in_planes: int, out_planes: int, act: ActivationDict, bias: bool, batch_norm: bool,
                 spectral_norm: bool, additional_input: int | None = None):
        super().__init__()
        self.main_1 = make_conv_block(torch.nn.Conv2d(in_planes, out_planes, 4, stride=2, padding=1, bias=bias), act,
                                      batch_norm=batch_norm, spectral_norm=spectral_norm,
                                      additional_input=additional_input)
        self.main_2 = make_conv_block(torch.nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1,
                                                      bias=bias),
                                      act, batch_norm=batch_norm, spectral_norm=spectral_norm,
                                      additional_input=additional_input)

        self.direct_avg_pool = torch.nn.AvgPool2d(2, 2)
        self.direct = make_conv_block(torch.nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, bias=bias), act,
                                      batch_norm=batch_norm, spectral_norm=spectral_norm,
                                      additional_input=additional_input)

    def forward(self, feat: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is None:
            main = self.main_2(self.main_1(feat))
            direct = self.direct(self.direct_avg_pool(feat))
        else:
            main = self.main_2(self.main_1(feat, c), c)
            direct = self.direct(self.direct_avg_pool(feat), c)
        return (main + direct) / 2


class FastGANDecoderDict(DecoderDict):
    """
    Parameter dictionary for FastGAN generators.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'width', 'batch_norm', 'spectral_norm'}.issubset(self.keys())
        self.width: int  # Hidden width multipler
        self.batch_norm: bool  # Whether to include batch norm
        self.spectral_norm: bool  # Whether to include spectral norm


class FastGANDecoder(torch.nn.Module):
    """
    FastGAN generator. Adapted the architecture for lower-dimensional images.
    """

    def __init__(self, in_dim: int, out_channels: int, out_size: int, config: FastGANDecoderDict):
        super().__init__()
        ngf = config.width
        self.im_size = out_size
        bias = not config.batch_norm
        batch_norm = config.batch_norm
        spectral_norm = config.spectral_norm

        nfc_multi = {4: 16, 8: 8, 16: 4, 32: 2, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * ngf)

        self.init = InitLayer(in_dim, nfc[4], bias, batch_norm, spectral_norm)

        self.feat_8 = up_block_comp(nfc[4], nfc[8], bias, batch_norm, spectral_norm)
        self.feat_16 = up_block(nfc[8], nfc[16], bias, batch_norm, spectral_norm)
        self.feat_32 = up_block_comp(nfc[16], nfc[32], bias, batch_norm, spectral_norm)
        self.to_big = make_conv_block(torch.nn.Conv2d(nfc[out_size], out_channels, 3, stride=1, padding=1, bias=False),
                                      None, spectral_norm=spectral_norm)

        if out_size > 32:
            self.feat_64 = up_block(nfc[32], nfc[64], bias, batch_norm, spectral_norm)
            self.se_64 = SEBlock(nfc[4], nfc[64], False, spectral_norm)
        if out_size > 64:
            self.feat_128 = up_block_comp(nfc[64], nfc[128], bias, batch_norm, spectral_norm)
            self.se_128 = SEBlock(nfc[8], nfc[128], False, spectral_norm)
        if out_size > 128:
            self.feat_256 = up_block(nfc[128], nfc[256], bias, batch_norm, spectral_norm)
            self.se_256 = SEBlock(nfc[16], nfc[256], False, spectral_norm)
        if out_size > 256:
            self.feat_512 = up_block_comp(nfc[256], nfc[512], bias, batch_norm, spectral_norm)
            self.se_512 = SEBlock(nfc[32], nfc[512], False, spectral_norm)
        if out_size > 512:
            self.feat_1024 = up_block(nfc[512], nfc[1024], bias, batch_norm, spectral_norm)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        if self.im_size == 32:
            return self.to_big(feat_32)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        if self.im_size == 64:
            return self.to_big(feat_64)

        feat_128 = self.se_128(feat_8, self.feat_128(feat_64))
        if self.im_size == 128:
            return self.to_big(feat_128)

        feat_256 = self.se_256(feat_16, self.feat_256(feat_128))
        if self.im_size == 256:
            return self.to_big(feat_256)

        feat_512 = self.se_512(feat_32, self.feat_512(feat_256))
        if self.im_size == 512:
            return self.to_big(feat_512)

        feat_1024 = self.feat_1024(feat_512)
        return self.to_big(feat_1024)


class FastGANEncoderDict(EncoderDict):
    """
    Parameter dictionary for FastGAN discriminators.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'width', 'batch_norm', 'spectral_norm', 'first_bias', 'activation'}.issubset(self.keys())
        self.width: int  # Hidden width multipler
        self.batch_norm: bool  # Whether to include batch norm
        self.spectral_norm: bool  # Whether to include spectral norm
        self.first_bias: bool  # Whether to add bias to the first convolution
        self.activation: ActivationDict = ActivationDict(self.activation)  # Activation


class FastGANEncoder(torch.nn.Module):
    """
    FastGAN discriminator, without the autoencoding trick.

    Changed the final layer with an additional layer to obtain a scalar output. Adapted the architecture for lower-
    dimensional images.

    Can optionnally take as input an additional tensor with is preprocessed by an MLP, then used to modulate
    convolutional layers outputs.
    """

    def __init__(self, in_channels: int, in_size: int, out_dim: int, config: FastGANEncoderDict,
                 additional_input: int | None = None):
        super().__init__()
        self.ndf = config.width
        self.im_size = in_size
        act = config.activation
        bias = not config.batch_norm
        batch_norm = config.batch_norm
        spectral_norm = config.spectral_norm

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

        nfc_multi = {4: 16, 8: 16, 16: 8, 32: 4, 64: 2, 128: 1, 256: 0.5, 512: 0.25, 1024: 0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v * self.ndf)

        if in_size == 1024:
            self.down_from_big = torch.nn.Sequential(make_conv_block(torch.nn.Conv2d(in_channels, nfc[1024], 4,
                                                                                     stride=2, padding=1,
                                                                                     bias=config.first_bias),
                                                                     act, spectral_norm=spectral_norm,
                                                                     additional_input=additional_input),
                                                     make_conv_block(torch.nn.Conv2d(nfc[1024], nfc[512], 4, stride=2,
                                                                                     padding=1,
                                                                                     bias=config.first_bias),
                                                                     act, batch_norm=batch_norm,
                                                                     spectral_norm=spectral_norm,
                                                                     additional_input=additional_input))
        elif in_size == 512:
            self.down_from_big = make_conv_block(torch.nn.Conv2d(in_channels, nfc[512], 4, stride=2, padding=1,
                                                                 bias=config.first_bias),
                                                 act, spectral_norm=spectral_norm, additional_input=additional_input)
        else:
            self.down_from_big = make_conv_block(torch.nn.Conv2d(in_channels, nfc[in_size * 2], 3, stride=1,
                                                                 padding=1, bias=config.first_bias),
                                                 act, spectral_norm=spectral_norm, additional_input=additional_input)

        if in_size >= 256:
            self.down_4 = DownBlockComp(nfc[512], nfc[256], act, bias, batch_norm, spectral_norm,
                                        additional_input=additional_input)
            self.se_2_16 = SEBlock(nfc[512], nfc[64], False, spectral_norm, additional_input=additional_input)
            self.se_4_32 = SEBlock(nfc[256], nfc[32], False, spectral_norm, additional_input=additional_input)
        if in_size >= 128:
            self.down_8 = DownBlockComp(nfc[256], nfc[128], act, bias, batch_norm, spectral_norm,
                                        additional_input=additional_input)
            self.se_8_64 = SEBlock(nfc[128], nfc[16], False, spectral_norm, additional_input=additional_input)
        if in_size >= 64:
            self.down_16 = DownBlockComp(nfc[128], nfc[64], act, bias, batch_norm, spectral_norm,
                                         additional_input=additional_input)
            self.se_16_128 = SEBlock(nfc[64], nfc[8], False, spectral_norm, additional_input=additional_input)

        self.down_32 = DownBlockComp(nfc[64], nfc[32], act, bias, batch_norm, spectral_norm,
                                     additional_input=additional_input)
        self.down_64 = DownBlockComp(nfc[32], nfc[16], act, bias, batch_norm, spectral_norm,
                                     additional_input=additional_input)
        self.down_128 = DownBlockComp(nfc[16], nfc[8], act, bias, batch_norm, spectral_norm,
                                      additional_input=additional_input)

        self.rf_big_1 = make_conv_block(torch.nn.Conv2d(nfc[8], nfc[4], 1, stride=1, padding=0, bias=bias), act,
                                        batch_norm=batch_norm, spectral_norm=spectral_norm,
                                        additional_input=additional_input)
        self.rf_big_2 = SequentialTuple(make_conv_block(torch.nn.Conv2d(nfc[4], out_dim, 4, stride=1, padding=0,
                                                                        bias=False),
                                                        None, spectral_norm=spectral_norm,
                                                        additional_input=additional_input),
                                        torch.nn.Flatten())

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        # Preprocessing of 2nd output
        if c is not None:
            c = self.c_mlp(c)
            c_args = [c]
        else:
            c_args = []

        feat = self.down_from_big(x, *c_args)

        if self.im_size >= 256:
            feat_2 = feat
            feat_4 = self.down_4(feat, *c_args)
        else:
            feat_2, feat_4 = None, None
        if self.im_size >= 128:
            feat_8 = self.down_8(feat_4 if feat_4 is not None else feat, *c_args)
        else:
            feat_8 = None

        if self.im_size >= 64:
            feat_16 = self.down_16(feat_8 if feat_8 is not None else feat, *c_args)
            if feat_2 is not None:
                feat_16 = self.se_2_16(feat_2, feat_16, *c_args)
        else:
            feat_16 = None

        feat_32 = self.down_32(feat_16 if feat_16 is not None else feat, *c_args)
        if feat_4 is not None:
            feat_32 = self.se_4_32(feat_4, feat_32, *c_args)

        feat_64 = self.down_64(feat_32, *c_args)
        if feat_8 is not None:
            feat_64 = self.se_8_64(feat_8, feat_64, *c_args)

        feat_last = self.down_128(feat_64, *c_args)
        if feat_16 is not None:
            feat_last = self.se_16_128(feat_16, feat_last, *c_args)

        return self.rf_big_2(self.rf_big_1(feat_last, *c_args), *c_args)
