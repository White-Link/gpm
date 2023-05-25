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


import yaml

from typing import Any, Callable

from gpm.data.datasets import Dataset
from gpm.eval.tests import Test, quanti_test
from gpm.models.models import Model
from gpm.utils import args


class DotDict(dict):
    """
    Dot notation access to nested dictionary attributes.
    """
    def __init__(self, *kargs, replace_nones: bool = False, **kwargs):
        super().__init__(*kargs, **kwargs)
        for k, v in self.items():
            if replace_nones and v is None:
                self[k] = DotDict()
            if isinstance(v, dict):
                self[k] = DotDict(v, replace_nones=replace_nones)
            if isinstance(v, list):
                self[k] = [DotDict(w, replace_nones=replace_nones) if isinstance(w, dict) else w for w in v]

    __getattr__ = dict.get
    __setattr__: Callable[[dict[Any, Any], Any, Any], None] = dict.__setitem__
    __delattr__: Callable[[dict[Any, Any], Any], None] = dict.__delitem__


class ArgDict(DotDict):
    """
    Specification of a DotDict containing command-line arguments of the program.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.seed: int | None = self[args.seed_arg]
        self.data_path: str = self[args.data_arg]
        self.configs: list[str] = self[args.configs_arg]
        self.load: bool = self[args.load_arg]
        self.load_best: bool = self[args.load_best_arg]
        self.only_test: bool = self[args.only_test_arg]
        self.save_path: str = self[args.save_arg]
        self.save_name: str = self[args.name_arg]
        self.erase: bool = self[args.erase_arg]
        self.no_chkpt: bool = self[args.no_chkpt_arg]
        self.train_log_freq: int = self[args.train_log_freq_arg]
        self.log_save_freq: int = self[args.log_save_freq_arg]
        self.amp: bool = self[args.amp_arg]
        self.deterministic: bool = self[args.deterministic_arg]
        self.no_benchmark: bool = self[args.no_benchmark_arg]
        self.device: list[int] | None = self[args.device_arg]
        self.nb_workers: int = self[args.workers_arg]
        self.prefetch_factor: int | None = None if self[args.prefetch_arg] == 0 else self[args.prefetch_arg]


class ModelDict(ArgDict):
    """
    ArgDict containing the information of the considered task and model.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'model', 'model_params', 'dataset', 'dataset_params', 'optim', 'eval', 'eval_freq', 'save_best',
                'test'}.issubset(self.keys())

        assert self.model in [model.value for model in Model]
        assert self.dataset in [dataset.value for dataset in Dataset]
        self.model: Model = Model(self.model)
        self.model_params: DotDict
        self.dataset: Dataset = Dataset(self.dataset)
        self.dataset_params: DotDict
        self.optim: OptimDict = OptimDict(self.optim)
        self.eval_freq: int
        self.save_best: bool
        self.n_gpu: int = 0
        self.info: DotDict = DotDict()

        self.eval: tuple[TestDict, ...] = tuple(TestDict(eval_config) for eval_config in self.eval)
        self.test: tuple[TestDict, ...] = tuple(TestDict(test_config) for test_config in self.test)
        assert len(self.eval) > 0 and len(self.test) > 0
        assert len(self.eval) == len({eval_config.config_name for eval_config in self.eval})
        assert len(self.test) == len({test_config.config_name for test_config in self.test})
        assert all(eval_config.metric is None for eval_config in self.eval[1:])

        assert self.eval_freq > 0


class OptimDict(DotDict):
    """
    Specification of a DotDict containing optimization options.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'batch_size', 'nb_steps', 'optimizers', 'schedulers'}.issubset(self.keys())
        self.batch_size: int
        self.nb_steps: int
        self.optimizers: DotDict
        self.schedulers: DotDict
        assert set(self.schedulers.keys()).issubset(self.optimizers.keys())


class ObjectDict(DotDict):
    """
    Specification of a DotDict containing an object name (e.g., a PyTorch optimizer) and its corresponding parameters.
    """
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'name' in self and 'settings' in self
        self.name: str
        self.settings: DotDict


class TestDict(DotDict):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'config_name', 'nb_steps', 'batch_size', 'nb_quali', 'tests'}.issubset(self.keys())
        assert all(test in [t.value for t in Test] for test in self.tests)
        self.config_name: str
        self.dataset_params: DotDict | None
        if 'dataset_params' not in self:
            self.dataset_params = None
        self.test_params: DotDict
        if 'test_params' not in self:
            self.test_params = DotDict()
        self.nb_steps: int
        self.batch_size: int
        self.nb_quali: int
        self.tests: list[Test] = list(set(map(Test, self.tests)))  # type: ignore
        if 'metric' in self:
            assert self.metric in [t.value for t in Test]
            self.metric: Test | None = Test(self.metric)
            assert self.metric in quanti_test and self.metric in self.tests
        else:
            self.metric = None


def load_yaml(path: str) -> DotDict:
    """
    Loads a YAML file as a DotDict.
    """
    with open(path, 'r') as f:
        opt = yaml.safe_load(f)
    return DotDict(opt, replace_nones=True)
