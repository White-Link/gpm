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


from typing import Any, Callable

from gpm.data.base import BaseDataset, Split
from gpm.data.collate import default_collate_dataclass
from gpm.data.datasets import Dataset
from gpm.data.image.celeba import CelebA, CelebAParamsDict
from gpm.data.image.mnist import MNIST, MNISTParamsDict
from gpm.data.lowd.gaussians import GaussiansDataset, GaussiansParamsDict
from gpm.utils.config import ModelDict


def dataset_factory_train_val(opt: ModelDict, seed: int) -> tuple[BaseDataset, tuple[BaseDataset, ...]]:
    """
    Returns the training set and the validation sets for the chosen dataset.

    The input seed serves as a common RNG initialization to split training dataset into training and validation sets
    for datasets like MNIST which do not come with their precomputed validation set.
    """
    val_sets: list[BaseDataset] = []
    if opt.dataset is Dataset.MNIST:
        opt.dataset_params = MNISTParamsDict(opt.dataset_params)
        train_set = MNIST(opt.data_path, Split.TRAIN, opt.dataset_params.ratio_val, seed=seed)
        for eval_config in opt.eval:
            val_params = eval_config.dataset_params
            if val_params is None:
                val_params = opt.dataset_params
            else:
                val_params = MNISTParamsDict(val_params)
                assert val_params.ratio_val == opt.dataset_params.ratio_val
            val_sets.append(MNIST(opt.data_path, Split.VAL, opt.dataset_params.ratio_val, seed=seed))
    elif opt.dataset is Dataset.CELEBA:
        opt.dataset_params = CelebAParamsDict(opt.dataset_params)
        train_set = CelebA(opt.data_path, Split.TRAIN, opt.dataset_params.image_size)
        for eval_config in opt.eval:
            val_params = eval_config.dataset_params
            if val_params is None:
                val_params = opt.dataset_params
            else:
                val_params = CelebAParamsDict(val_params)
            val_sets.append(CelebA(opt.data_path, Split.VAL, val_params.image_size))
    elif opt.dataset is Dataset.GAUSSIANS:
        opt.dataset_params = GaussiansParamsDict(opt.dataset_params)
        train_set = GaussiansDataset(opt.data_path, Split.TRAIN, opt.dataset_params)
        for eval_config in opt.eval:
            val_params = eval_config.dataset_params
            if val_params is None:
                val_params = opt.dataset_params
            else:
                val_params = GaussiansParamsDict(val_params)
            val_sets.append(GaussiansDataset(opt.data_path, Split.VAL, val_params))
    else:
        raise ValueError(f'No dataset named `{opt.dataset}`')
    return train_set, tuple(val_sets)


def dataset_factory_test(opt: ModelDict) -> tuple[BaseDataset, ...]:
    """
    Returns the testing sets for the chosen dataset.
    """
    test_sets: list[BaseDataset] = []
    for test_config in opt.test:
        test_params = test_config.dataset_params
        if opt.dataset is Dataset.MNIST:
            opt.dataset_params = MNISTParamsDict(opt.dataset_params)
            test_params = MNISTParamsDict(test_params) if test_params is not None else opt.dataset_params
            test_sets.append(MNIST(opt.data_path, Split.TEST))
        elif opt.dataset is Dataset.CELEBA:
            opt.dataset_params = CelebAParamsDict(opt.dataset_params)
            test_params = CelebAParamsDict(test_params) if test_params is not None else opt.dataset_params
            test_sets.append(CelebA(opt.data_path, Split.TEST, test_params.image_size))
        elif opt.dataset is Dataset.GAUSSIANS:
            opt.dataset_params = GaussiansParamsDict(opt.dataset_params)
            test_params = GaussiansParamsDict(test_params) if test_params is not None else opt.dataset_params
            test_sets.append(GaussiansDataset(opt.data_path, Split.TEST, test_params))
        else:
            raise ValueError(f'No dataset named `{opt.dataset}`')
    return tuple(test_sets)


def collate_fn_factory(dataset: BaseDataset) -> Callable[[list[Any]], Any]:
    """
    Returns the DataLoader's collate function.
    """
    return default_collate_dataclass
