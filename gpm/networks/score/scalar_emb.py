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
#  - Created a class encompassing with positional and Fourier embeddings as inputs of another module.


import enum
import math
import torch


@enum.unique
class NoiseEncoding(str, enum.Enum):
    """
    Implemented types of time / noise level embeddings.
    """
    POSITIONAL = 'positional'
    FOURIER = 'fourier'


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels: int, scale: float = 16):
        super().__init__()
        self.freqs: torch.Tensor
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger(2 * math.pi * self.freqs)
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class NetworkScalarEmbedding(torch.nn.Module):
    """
    Wraps the 2nd scalar input of a network with either Fourier or positional embeddings.
    """

    def __init__(self, model: torch.nn.Module, embedding: NoiseEncoding, embedding_size: int):
        super().__init__()
        self.model = model

        assert embedding_size % 2 == 0
        if embedding is NoiseEncoding.POSITIONAL:
            self.embed = PositionalEmbedding(embedding_size, endpoint=True)
        elif embedding is NoiseEncoding.FOURIER:
            self.embed = FourierEmbedding(embedding_size)
        else:
            raise ValueError(f'Embedding `{embedding}` not implemented yet')

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        embedding = self.embed(s)
        return self.model(x, embedding)
