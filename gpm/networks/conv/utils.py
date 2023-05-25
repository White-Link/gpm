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


import torch

from torch.nn.utils.parametrizations import spectral_norm as sn

from gpm.networks.activations import ActivationDict, activation_factory
from gpm.networks.utils import SequentialTuple


class Adaptive2DLayer(torch.nn.Module):
    """
    Modulates (channel-wise) the output of a convolutional layer by a 2nd flat output.
    """

    def __init__(self, layer: torch.nn.Module, nb_channels: int, spectral_norm: bool, additional_input: int):
        super().__init__()
        self.affine = torch.nn.Linear(additional_input, nb_channels * 2)
        if spectral_norm:
            self.affine = sn(self.affine)
        self.layer = layer

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.affine(c).unsqueeze(-1).unsqueeze(-1)
        scale, shift = params.chunk(chunks=2, dim=1)
        return torch.addcmul(shift, self.layer(x), scale + 1)


def make_conv_block(conv: torch.nn.Conv2d | torch.nn.ConvTranspose2d,
                    activation: ActivationDict | torch.nn.Module | None,
                    batch_norm: bool = False, spectral_norm: bool = False,
                    additional_input: int | None = None) -> torch.nn.Module:
    """
    Builds a convolutional block based on a convolution module by adding batch norm, spectral norm, an activation, and
    optionnally a channel-wise modulation of the output of the convolution by a 2nd flat output.
    """
    out_channels = conv.out_channels
    if spectral_norm:
        conv_sn = sn(conv)
    else:
        conv_sn = conv
    pre_act_modules: list[torch.nn.Module] = [conv_sn]
    if batch_norm:
        pre_act_modules.append(torch.nn.BatchNorm2d(out_channels))
    act_modules = []
    if activation is not None:
        if isinstance(activation, torch.nn.Module):
            act_modules.append(activation)
        else:
            act_modules.append(activation_factory(activation))
    if additional_input is None or additional_input == 0:
        return torch.nn.Sequential(*(pre_act_modules + act_modules))
    else:
        adaptive_layer = Adaptive2DLayer(torch.nn.Sequential(*pre_act_modules), out_channels,
                                         spectral_norm, additional_input)
        return SequentialTuple(adaptive_layer, *act_modules)
