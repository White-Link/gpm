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
import torchvision

from torchvision.transforms import CenterCrop, Compose, Normalize, ToTensor

from gpm.data.base import BatchClass, Split, split_train_val
from gpm.data.image.base import BaseImageDataset
from gpm.utils.config import DotDict


class MNISTParamsDict(DotDict):
    """
    Parameter dictionary for the MNIST dataset.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'ratio_val' in self
        self.ratio_val: float  # Proportion of the training dataset which will be kept for validation only.


class MNIST(BaseImageDataset):
    """
    MNIST dataset, expanded to 32x32 by adding four layers of black pixels around the original images.

    The validation dataset is chosen by randomly removing elements of the original training dataset. The random choice
    is performed using the given seed to be shared with the constructor of the training dataset.
    """

    def __init__(self, data_path: str, split: Split, ratio_val: float | None = None, seed: int | None = None):
        super().__init__(data_path, split)
        transform_list = [CenterCrop(32), ToTensor(), Normalize(0.5, 0.5)]
        self.dataset = torchvision.datasets.MNIST(self.data_path, train=self.split is not Split.TEST,
                                                  transform=Compose(transform_list), download=True)
        # Split training dataset in training and validation sets
        if self.split is not Split.TEST:
            assert seed is not None and ratio_val is not None
            rng = torch.Generator()
            rng.manual_seed(seed)
            indices_train, indices_val = split_train_val(len(self.dataset), ratio_val, rng)
            if self.split is Split.TRAIN:
                self.indices = indices_train
            else:
                self.indices = indices_val
        else:
            self.indices = range(len(self.dataset))

    @property
    def name(self) -> str:
        train = 'train' if self.split is not Split.TEST else 'test'
        return f'MNIST ({train})'

    @property
    def channels(self) -> int:
        return 1

    @property
    def image_size(self) -> int:
        return 32

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> BatchClass:
        datapoint = self.dataset[self.indices[index]]
        return BatchClass(x=datapoint[0], y=datapoint[1])
