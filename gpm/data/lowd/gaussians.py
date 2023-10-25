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

from torch import distributions as D

from gpm.data.base import Split
from gpm.data.lowd.base import BaseSampleLowDDataset, BaseSampleLowDParamsDict


class GaussiansParamsDict(BaseSampleLowDParamsDict):
    """
    Parameter dictionary for the Gaussians dataset.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'nb' in self and 'radius' in self and 'std' in self
        self.nb: int  # Number of modes.
        self.radius: float  # Radius of the circle containing each mode center.
        self.std: float  # Standard deviation of each Gaussian mode.


class GaussiansDataset(BaseSampleLowDDataset):
    """
    Gaussians dataset: Gaussian modes evenly distributed on a centered circle.
    """

    def __init__(self, data_path: str, split: Split, config: GaussiansParamsDict):
        super().__init__(data_path, split, config)
        self.nb = config.nb
        self.radius = config.radius
        self.std = config.std

        # Sets up each mode Gaussain distributon (by computing their centers)
        n = self.nb
        delta_theta = 2 * math.pi / n
        centers_x = torch.cos(delta_theta * torch.arange(n))
        centers_y = torch.sqrt(1 - centers_x ** 2) * torch.sign(torch.arange(n) - n / 2)
        centers = torch.stack([centers_x, centers_y], dim=1)
        component_distribution = D.Independent(D.Normal(self.radius * centers, self.std), 1)

        # The final distribution is a mixture of the mode
        mixture_distribution = D.Categorical(torch.ones(n))
        self.distribution = D.MixtureSameFamily(mixture_distribution, component_distribution)

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    @property
    def dim(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return f'{self.nb} Gaussians'
