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

from abc import abstractmethod

from gpm.data.base import BaseDataset, Split
from gpm.utils.config import DotDict
from gpm.utils.types import Batch


class BaseLowDDataset(BaseDataset):
    """
    Asbtract class for dataset on low-dimensional data (lying in a non-structured data space).
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    def __str__(self) -> str:
        return f'{self.name}: {len(self)} points of dim {self.dim}'


class BaseSampleLowDParamsDict(DotDict):
    """
    Asbtract parameter dictionary for `BaseSampleLowDDataset`.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'size' in self
        self.size: int  # Number of sampled elements


class BaseSampleLowDDataset(BaseLowDDataset):
    """
    Asbtract class for dataset on low-dimensional data which are sampled from a given distribution.
    """
    def __init__(self, data_path: str, split: Split, config: BaseSampleLowDParamsDict):
        super().__init__(data_path, split)
        self.size = config.size

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Asbtract method used to sample elements from the dataset.
        """
        pass

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Batch:
        return Batch(x=self.sample())
