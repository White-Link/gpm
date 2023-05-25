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


import random
import torch

from abc import abstractmethod
from torch.utils.data import DataLoader
from typing import Any, Iterator, Sequence

from gpm.utils.config import ModelDict, ObjectDict, TestDict
from gpm.utils.optimizer import build_optimizer
from gpm.utils.types import Batch, ForwardFn, Log, Optimizers, Scalers, Schedulers, Scaler, Scheduler


class BaseModel(torch.nn.Module):
    """
    Abstract Module class that all models should inherit. Any model should be embedded in such a single Module. Its
    specification is used in the training loop.
    """

    def __init__(self):
        super().__init__()
        self.optimizers: Optimizers = {}

    @abstractmethod
    def optimizers_names(self, opt: ModelDict) -> set[str]:
        """
        Returns the set of optimizer names required to be defined in the configuration files. Only one is needed in
        many cases, but GANs and Score GANs need two (one for the generator and one for the discriminator / score).
        """
        pass

    @abstractmethod
    def optimized_params(self, optimizer_name: str, opt: ModelDict) -> Iterator[torch.nn.parameter.Parameter]:
        """
        Returns the set of optimized parameters for each optimizer name.
        """
        pass

    def build_optimizers(self, opt: ModelDict) -> Optimizers:
        """
        Creates the optimizers based on the configuration file.
        """
        assert set(opt.optim.optimizers.keys()) == self.optimizers_names(opt)
        optimizers = {}
        for optimizer_name, optimizer_config in opt.optim.optimizers.items():
            optimizers[optimizer_name] = build_optimizer(self.optimized_params(optimizer_name, optimizer_config),
                                                         ObjectDict(opt.optim.optimizers[optimizer_name]))
        return optimizers

    def set_optimizers(self, optimizers: Optimizers):
        self.optimizers = optimizers

    @classmethod
    def extract_optimizer(cls, optimizer_name: str, optimizers: Optimizers, scalers: Scalers,
                          schedulers: Schedulers) -> tuple[torch.optim.Optimizer, Scaler, Scheduler]:
        """
        Utility function for training iterations that returns the optimizer and associated scaler (for AMP) and
        scheduler (for LR tuning).
        """
        optimizer = optimizers[optimizer_name]
        scaler, scheduler = None, None
        if scalers is not None:
            scaler = scalers[optimizer_name]
        if optimizer_name in schedulers:
            scheduler = schedulers[optimizer_name]
        return optimizer, scaler, scheduler

    @abstractmethod
    def forward(self, batch: Batch, op: Any) -> Sequence[torch.Tensor]:
        """
        Must be called once per backward call. Typically computes the loss ber batch element. `op` helps it know
        which training step to perform (e.g. for GANs, either the discriminator or the generator step).
        """
        pass

    @classmethod
    @abstractmethod
    def training_step(cls, step: int, batch: Batch, forward_fn: ForwardFn, optimizers: Optimizers, scalers: Scalers,
                      schedulers: Schedulers, device: torch.device, shared_rng: random.Random,
                      opt: ModelDict) -> tuple[float, Log, list[str]]:
        """
        Corresponds to a training iteration. Should take care of zero_grad() and optimizer update. Must call one
        forward from the model (provided by `forward_fn`) per backward call, to respect DDP requirements.
        """
        pass

    def post_training_step(self, step: int, shared_rng: random.Random):
        """
        If needed, adjust some parameters after the gradient update.
        """
        return

    @abstractmethod
    def evaluation_step(self, batch: Batch, device: torch.device, opt: ModelDict,
                        config: TestDict) -> Sequence[torch.Tensor]:
        """
        Computes evaluation statistics (loss, generated samples, some additional information) on a minibatch.
        """
        pass

    @abstractmethod
    def evaluation_logs(self, eval_results: Sequence[torch.Tensor], device: torch.device, opt: ModelDict,
                        config: TestDict) -> tuple[float, Log]:
        """
        Takes evaluation statistics from `evaluation_step` on the whole validation/testing set, and computes different
        metrics and prints of relevant information as specified in the test configuration.
        """
        pass

    @classmethod
    def combine_logs(cls, results_step: Sequence[torch.Tensor], results: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Compiles a minibatch of `evaluation_step` with previously compiled results.
        """
        return [torch.cat([prev_result, result]) for prev_result, result in zip(results, results_step)]

    def evaluation(self, loader: DataLoader, device: torch.device, opt: ModelDict,
                   config: TestDict) -> tuple[float, Log]:
        """
        Compiles all minibatches of `evaluation_step` and launches the computation of different metrics and printing of
        relevant information as specified in the test configuration.
        """
        results = None
        for step, batch in enumerate(loader):
            if step >= config.nb_steps:
                break
            results_step = self.evaluation_step(batch, device, opt, config)
            results_step = [tensor.cpu() for tensor in results_step]
            if results is None:
                results = results_step
            else:
                results = self.combine_logs(results_step, results)
        assert results is not None
        return self.evaluation_logs(results, device, opt, config)
