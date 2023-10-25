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


import torchvision

from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

from gpm.data.base import BatchClass, Split
from gpm.data.image.base import BaseImageDataset
from gpm.utils.config import DotDict


class CelebAParamsDict(DotDict):
    """
    Parameter dictionary for the CelebA dataset.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'image_size' in self
        self.image_size: int  # Size to which CelebA images should be resized.


class CelebA(BaseImageDataset):
    """
    CelebA dataset. Each CelebA original image is center-cropped and resized to the given image size.

    The attached class information correspond to image attribute values.
    """

    def __init__(self, data_path: str, split: Split, image_size: int):
        super().__init__(data_path, split)
        assert 0 < image_size <= 178
        self._image_size = image_size
        transform = Compose([Resize(self._image_size, interpolation=InterpolationMode.LANCZOS), CenterCrop(image_size),
                             ToTensor(), Normalize(0.5, 0.5)])
        celeba_split = self.split.value if self.split is not Split.VAL else 'valid'
        self.dataset = torchvision.datasets.CelebA(self.data_path, split=celeba_split, target_type='attr',
                                                   transform=transform, download=True)

    @property
    def name(self) -> str:
        return f'CelebA ({self.split.value})'

    @property
    def channels(self) -> int:
        return 3

    @property
    def image_size(self) -> int:
        return self._image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> BatchClass:
        datapoint = self.dataset[index]
        return BatchClass(x=datapoint[0], y=datapoint[1])
