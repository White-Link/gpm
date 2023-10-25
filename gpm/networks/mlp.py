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


import math
import torch

from torch.nn.utils.parametrizations import spectral_norm
from typing import Sequence

from gpm.networks.activations import ActivationDict, activation_factory
from gpm.networks.utils import EncoderDict, DecoderDict, ScoreDict
from gpm.utils.config import DotDict


class MLPDict(DotDict):
    """
    Parameter dictionary for MLPs.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'depth', 'width', 'activation'}.issubset(self.keys())
        self.depth: int  # Number of hidden layers
        self.width: int  # Width of hidden layers
        self.activation: ActivationDict | list[ActivationDict]  # Activations of hidden layers
        if isinstance(self.activation, list):
            self.activation = list(map(ActivationDict, self.activation))
        else:
            self.activation = ActivationDict(self.activation)
        self.batch_norm: bool  # Whether to incorporate batch norm (default, False)
        if 'batch_norm' not in self:
            self.batch_norm = False
        self.spectral_norm: bool  # Whether to perform spectral norm (default, False)
        if 'spectral_norm' not in self:
            self.spectral_norm = False


class MLPEncoderDict(MLPDict, EncoderDict):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


class MLPDecoderDict(MLPDict, DecoderDict):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


class MLPScoreDict(MLPDict, ScoreDict):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)


class MLP(torch.nn.Module):
    """
    MLP architecture. Can optionnally take as input two tensors that are then concatenated.
    """

    def __init__(self, in_size: int | Sequence[int], out_size: int | Sequence[int], config: MLPDict,
                 compl_size: int | None = None):
        super().__init__()

        if config.depth >= 0:
            if isinstance(in_size, int):
                in_dim = in_size
            else:
                in_dim = math.prod(in_size)
            if compl_size is not None:
                in_dim += compl_size
            if isinstance(out_size, int):
                out_dim = out_size
            else:
                out_dim = math.prod(out_size)

            sn = spectral_norm if config.spectral_norm else lambda x: x

            if isinstance(config.activation, list):
                activation_fns = list(map(activation_factory, config.activation))
            else:
                activation_fns = [activation_factory(config.activation)] * config.depth
            hidden_width = config.width if config.depth > 0 else out_dim
            layers = [
                sn(torch.nn.Linear(in_dim, hidden_width)),
                *[
                    torch.nn.Sequential(*(
                        ([torch.nn.BatchNorm1d(config.width)] if config.batch_norm else []) + [
                            activation_fns[i],
                            sn(torch.nn.Linear(config.width,
                                               config.width if i < config.depth - 1 else out_dim))  # type:ignore
                        ]
                    ))
                    for i in range(config.depth)
                ],
            ]
            if not isinstance(out_size, int):
                layers.append(torch.nn.Unflatten(1, tuple(out_size)))

            self.mlp = torch.nn.Sequential(*layers)

        else:
            assert in_size == out_size
            self.mlp = torch.nn.Identity()

    def forward(self, x: torch.Tensor, compl_input: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        if compl_input is not None:
            x = torch.cat([x, compl_input], dim=1)
        return self.mlp(x)
