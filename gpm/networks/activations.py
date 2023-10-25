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


import enum
import math
import torch

from gpm.utils.config import DotDict


@enum.unique
class Activation(str, enum.Enum):
    """
    Implemented activations.
    """
    IDENTITY = 'none'
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    SILU = 'silu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    SOFTPLUS = 'softplus'
    SIN = 'sin'
    RBF = 'rbf'  # Special type of sin activation


class Sin(torch.nn.Module):
    def __init__(self, A, w, phi):
        super().__init__()
        self.A = A
        self.w = w
        self.phi = phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.A * torch.sin(self.w * x + self.phi)

    def __repr__(self):
        return f'Sin(A={self.A}, w={self.w}, phi={self.phi})'


class ActivationDict(DotDict):
    """
    Common parameter dictionary for activations. The activation type should be specified, which will then trigger
    parameter processing for the required activation.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'name' in self
        assert self.name in [activation.value for activation in Activation]
        self.name: Activation = Activation(self.name)


class ReLUDict(ActivationDict):
    """
    Parameter dictionary for ReLU activations. Offers the option to compute them in-place.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.inplace: bool
        if 'inplace' not in self:
            self.inplace = True


class LeakyReLUDict(ReLUDict):
    """
    Parameter dictionary for the LeakyReLU activation.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.negative_splope: float  # Negative slope of the activation (default, 0.2)
        if 'negative_splope' not in self:
            self.negative_splope = 0.2


class SinDict(ActivationDict):
    """
    Parameter dictionary for the Sin activation.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'w' in self
        self.A: float
        if 'A' not in self:
            self.A = 1
        self.w: float
        self.phi: float
        if 'phi' not in self:
            self.phi = 0


class RBFDict(ActivationDict):
    """
    Parameter dictionary for the RBF activation.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'gamma' in self
        self.gamma: float


def activation_factory(activation: ActivationDict | None) -> torch.nn.Module:
    """
    Constructs an activation based on the provided parameters.
    """
    if activation is None:
        return torch.nn.Identity()
    name = activation.name
    if name is Activation.IDENTITY:
        return torch.nn.Identity()
    elif name is Activation.RELU:
        activation = ReLUDict(activation)
        return torch.nn.ReLU(inplace=activation.inplace)
    elif name is Activation.LEAKY_RELU:
        activation = LeakyReLUDict(activation)
        return torch.nn.LeakyReLU(activation.negative_splope, inplace=activation.inplace)
    elif name is Activation.SILU:
        activation = ReLUDict(activation)
        return torch.nn.SiLU(inplace=activation.inplace)
    elif name is Activation.SIGMOID:
        return torch.nn.Sigmoid()
    elif name is Activation.TANH:
        return torch.nn.Tanh()
    elif name is Activation.SOFTPLUS:
        return torch.nn.Softplus()
    elif name is Activation.SIN:
        activation = SinDict(activation)
        return Sin(activation.A, activation.w, activation.phi)
    elif name is Activation.RBF:
        activation = RBFDict(activation)
        return Sin(math.sqrt(2), math.sqrt(2 * activation.gamma), math.pi / 4)
    raise ValueError(f'Activation function `{name}` not yet implemented')
