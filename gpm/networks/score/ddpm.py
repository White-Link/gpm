# Copyright (C) 2023 Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac,
# Mickaël Chen, Alain Rakotomamonjy
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/


# Adapted Nvidia code from https://github.com/NVlabs/edm (file `training/networks`).

# Initially released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license and
# redistributed under the same license (see attached file).

# The original work included the following copyright notice.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# Changes:
#  - Removed unused module definitions.
#  - Adapted the code to the coding style of this program (typing, spacing, naming, etc.).
#  - Removed the custom backward for attention because of its incompatibility with `functorch`.
#  - Replaced the custom implementation of linear layers with Pytorch's one.


import enum
import math
import torch

from gpm.networks.utils import ScoreDict


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, bias: bool = True, up: bool = False,
                 down: bool = False, resample_filter: list[int] = [1, 1], fused_resample: bool = False):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        self.weight = torch.nn.Parameter(torch.rand([out_channels, in_channels, kernel, kernel])) if kernel else None
        self.bias = torch.nn.Parameter(torch.rand([out_channels])) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.resample_filter: torch.Tensor | None
        self.register_buffer('resample_filter', f if up or down else None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            assert f is not None
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                                                     groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            assert f is not None
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                assert f is not None
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                                                         groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                assert f is not None
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels,
                                               stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels: int, num_groups: int = 32, min_channels_per_group: int = 4, eps: float = 1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        assert self.num_groups > 0
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype),
                                           bias=self.bias.to(x.dtype), eps=self.eps)
        return x


class UNetBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, up: bool = False, down: bool = False,
                 attention: bool = False, num_heads: int | None = None, channels_per_head: int = 64,
                 dropout: float = 0, skip_scale: float = 1, eps: float = 1e-5, resample_filter: list[int] = [1, 1],
                 resample_proj: bool = False, adaptive_scale: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        if not attention:
            self.num_heads = 0
        else:
            self.num_heads = num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down,
                            resample_filter=resample_filter)
        self.affine = torch.nn.Linear(in_features=emb_channels,
                                      out_features=out_channels * (2 if adaptive_scale else 1))
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down,
                               resample_filter=resample_filter)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1)
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1)

        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None):
        orig = x
        x = self.conv0(self.silu(self.norm0(x)))

        if emb is not None:
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        else:
            params = torch.zeros_like(torch.cat([x, x], dim=1)) if self.adaptive_scale else torch.zeros_like(x)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = self.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = self.silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3,
                                                      -1).unbind(2)
            # w = AttentionOp.apply(q, k)
            w = torch.einsum('ncq,nck->nqk', q, k / math.sqrt(k.shape[1])).softmax(dim=2)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


class SongUNetBase(torch.nn.Module):
    def __init__(self, img_resolution: int, in_channels: int, out_channels: int, noise_channels: int,
                 model_channels: int = 128, channel_mult: list[int] = [1, 2, 2, 2], channel_mult_emb: int = 4,
                 num_blocks: int = 4, attn_resolutions: list[int] = [16], dropout: float = 0.10,
                 encoder_type: str = 'standard', decoder_type: str = 'standard', resample_filter: list[int] = [1, 1]):
        super().__init__()
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        emb_channels = model_channels * channel_mult_emb
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=math.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False
        )

        # Mapping.
        if noise_channels > 0:
            self.map_layer0 = torch.nn.Linear(in_features=noise_channels, out_features=emb_channels)
            self.map_layer1 = torch.nn.Linear(in_features=emb_channels, out_features=emb_channels)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True,
                                                          **block_kwargs)  # type: ignore
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0,
                                                               down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3,
                                                                   down=True, resample_filter=resample_filter,
                                                                   fused_resample=True)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)  # type: ignore
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                         **block_kwargs)  # type: ignore
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout,
                                                         **block_kwargs)  # type: ignore
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True,
                                                        **block_kwargs)  # type: ignore
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()  # type: ignore
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)  # type: ignore
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                             kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3)

        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor | None = None):
        # Mapping.
        emb = noise_labels
        if emb is not None:
            emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
            emb = self.silu(self.map_layer0(emb))
            emb = self.silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / math.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(self.silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


@enum.unique
class EncoderType(str, enum.Enum):
    STANDARD = 'standard'
    SKIP = 'skip'
    RESIDUAL = 'residual'


@enum.unique
class DecoderType(str, enum.Enum):
    STANDARD = 'standard'
    SKIP = 'skip'


class SongUNetDict(ScoreDict):
    """
    Parameter dictionary for the U-Net used in the EDM paper. Cf. the EDM code for documentation.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'model_channels', 'channel_mult', 'channel_mult_emb', 'resample_filter', 'num_blocks', 'dropout',
                'encoder_type', 'decoder_type'}.issubset(self.keys())
        self.model_channels: int
        self.channel_mult: list[int]
        self.channel_mult_emb: int
        self.resample_filter: list[int]
        self.num_blocks: int
        self.dropout: float
        self.encoder_type: EncoderType = EncoderType(self.encoder_type)
        self.decoder_type: DecoderType = DecoderType(self.decoder_type)


class SongUNet(SongUNetBase):
    """
    U-Net used in the EDM paper. Cf. the EDM code for documentation.
    """

    def __init__(self, img_resolution: int, in_channels: int, out_channels: int, noise_channels: int,
                 config: SongUNetDict):
        super().__init__(img_resolution, in_channels, out_channels, noise_channels,
                         model_channels=config.model_channels, channel_mult=config.channel_mult,
                         channel_mult_emb=config.channel_mult_emb, resample_filter=config.resample_filter,
                         num_blocks=config.num_blocks, dropout=config.dropout, encoder_type=config.encoder_type.value,
                         decoder_type=config.decoder_type.value)
