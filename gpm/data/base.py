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

from abc import abstractmethod
from dataclasses import dataclass
from torch.utils.data import Dataset

from gpm.utils.types import Batch


@enum.unique
class Split(str, enum.Enum):
    """
    Data split indicator (train, val, test).
    """
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class BaseDataset(Dataset):
    """
    Abstract class that all implemented datasets should inherit.
    """

    def __init__(self, data_path: str, split: Split):
        super().__init__()
        self.data_path = data_path
        self.split = split

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Batch:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class BatchClass(Batch):
    """
    Batches with additional class information stored in the `y` attribute.
    """
    y: torch.Tensor


def split_train_val(dataset_len: int, ratio_val: float, rng: torch.Generator) -> tuple[list[int], list[int]]:
    """
    Splits dataset indices into a training and validation partition using the given RNG seed.

    Useful for datasets which have no built-in validation set.
    """
    assert 0 < ratio_val < 1
    assert 0 < int((1 - ratio_val) * dataset_len) < dataset_len
    permutation = torch.randperm(dataset_len, generator=rng).tolist()
    return permutation[:int((1 - ratio_val) * dataset_len)], permutation[int((1 - ratio_val) * dataset_len):]
