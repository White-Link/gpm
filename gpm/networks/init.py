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

from functools import partial
from typing import Sequence

from gpm.networks.score.ddpm import Conv2d, GroupNorm
from gpm.utils.config import DotDict


linear_layers = (
    torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d, torch.nn.Linear, Conv2d
)

normalization_layers = (
    torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.GroupNorm, torch.nn.LayerNorm,
    GroupNorm
)


@enum.unique
class Init(str, enum.Enum):
    """
    Implemented initializations.
    """
    DEFAULT = 'default'  # PyTorch's default layer initizalization
    NORMAL = 'normal'  # Normal with zero bias init
    NORMAL_BIAS = 'normal_bias'  # Normal with non-zero bias init
    XAVIER = 'xavier'
    KAIMING = 'kaiming'
    ORTHOGONAL = 'orthogonal'


class InitDict(DotDict):
    """
    Common parameter dictionary for initializations. The initialization type should be specified, which will then
    trigger parameter processing for the required init.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'name' in self
        self.name: Init = Init(self.name)


class InitGainDict(InitDict):
    """
    Common parameter dictionary for initializations with gain.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'gain' in self
        self.gain: float


class InitKaimingDict(InitGainDict):
    """
    Parameter dictionary for Kaiming initializations.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'activation' in self
        self.slope: float
        if 'slope' not in self:
            self.slope = 0
        self.activation: str


def init_module(config: InitDict, module: torch.nn.Module):
    """
    Initializes weights of a Module following the initialization choice.
    """

    if config.name is not Init.DEFAULT:
        config = InitGainDict(config)

        if isinstance(module, normalization_layers):
            if module.weight is not None:
                torch.nn.init.normal_(module.weight.data, mean=1, std=config.gain)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias.data, 0)

        if isinstance(module, linear_layers):
            if module.weight is not None:
                if config.name is Init.NORMAL or config.name is Init.NORMAL_BIAS:
                    torch.nn.init.normal_(module.weight.data, mean=0, std=config.gain)
                elif config.name is Init.XAVIER:
                    torch.nn.init.xavier_normal_(module.weight.data, gain=config.gain)
                elif config.name is Init.KAIMING:
                    config = InitKaimingDict(config)
                    torch.nn.init.kaiming_normal_(module.weight.data, a=config.slope, nonlinearity=config.activation)
                elif config.name is Init.ORTHOGONAL:
                    torch.nn.init.orthogonal_(module.weight.data, gain=config.gain)  # type: ignore
                else:
                    raise ValueError(f'Initialization method {config.name} not yet implemented')
            if hasattr(module, 'bias') and module.bias is not None:
                if config.name is not Init.NORMAL_BIAS:
                    torch.nn.init.constant_(module.bias.data, 0.0)
                else:
                    torch.nn.init.normal_(module.bias.data, mean=0, std=config.gain)


def init_network(modules: torch.nn.Module | Sequence[torch.nn.Module], configs: InitDict | Sequence[InitDict]):
    """
    Applies a given sequence of initializations to a given list of Modules.
    """

    if isinstance(modules, torch.nn.Module):
        modules = [modules]
    if isinstance(configs, InitDict):
        configs = [configs]
    for config in configs:
        init_function = partial(init_module, config)
        for module in modules:
            module.apply(init_function)
