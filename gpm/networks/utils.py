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


import enum
import torch

from typing import Iterable

from gpm.networks.activations import ActivationDict
from gpm.networks.score.edm import NoiseEncoding
from gpm.utils.config import DotDict


Grad = tuple[torch.Tensor | None, ...]


@enum.unique
class Encoder(str, enum.Enum):
    """
    Implemented encoder architectures.
    """
    DCGAN = 'dcgan'
    RESNET = 'resnet'
    FASTGAN = 'fastgan'
    MLP = 'mlp'


@enum.unique
class Decoder(str, enum.Enum):
    """
    Implemented decoder architectures.
    """
    DCGAN = 'dcgan'
    RESNET = 'resnet'
    FASTGAN = 'fastgan'
    MLP = 'mlp'


@enum.unique
class Score(str, enum.Enum):
    """
    Implemented score architectures.
    """
    MLP = 'mlp'
    DDPM = 'ddpm'  # U-Net


class NetworkDict(DotDict):
    """
    Common parameter dictionary for encoders, decoders and score networks. Offers the option to choose a final
    activation.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        if 'final_activation' in self and self.final_activation is not None:
            self.final_activation: ActivationDict | None = ActivationDict(self.final_activation)
        else:
            self.final_activation = None


class EncoderDict(NetworkDict):
    """
    Common parameter dictionary for encoders. The architecture type should be specified, which will then trigger
    parameter processing for the required architecture.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'name' in self
        self.name: Encoder = Encoder(self.name)


class DecoderDict(NetworkDict):
    """
    Common parameter dictionary for decoders. The architecture type should be specified, which will then trigger
    parameter processing for the required architecture.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'name' in self
        self.name: Decoder = Decoder(self.name)


class ScoreDict(NetworkDict):
    """
    Common parameter dictionary for score networks. The architecture type should be specified, which will then trigger
    parameter processing for the required architecture.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'name' in self
        self.name: Score = Score(self.name)


class EDMDenoiserDict(ScoreDict):
    """
    Parameter for EDM-based score networks.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'sigma_min', 'sigma_max', 'sigma_data', 'embedding', 'embedding_size'}.issubset(self.keys())
        # Noise scheduling parameters, cf. the EDM paper
        self.sigma_min: float
        self.sigma_max: float
        self.sigma_data: float
        self.embedding: NoiseEncoding = NoiseEncoding(self.embedding)  # Time / noise level embedding type
        self.embedding_size: int  # Size of time / noise level embeddings


class SequentialTuple(torch.nn.Sequential):
    """
    Extension of PyTorch's `Sequential` for tuple inputs.
    """

    def forward(self, *inputs):
        first = True
        for module in self:
            if first:
                inputs = module(*inputs)
                first = False
            else:
                inputs = module(inputs)
        return inputs
