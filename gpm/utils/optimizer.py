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
import typing

from torch import optim
from torch.cuda.amp.autocast_mode import autocast
from typing import Iterator

from gpm.utils.config import ModelDict, ObjectDict
from gpm.utils.types import Scaler, Scheduler


def build_optimizer(parameters: Iterator[torch.nn.parameter.Parameter], config: ObjectDict) -> optim.Optimizer:
    """
    Builds an optimizer on the given parameters from its PyTorch name and parameters.
    """
    try:
        optimizer = getattr(optim, config.name)
        return optimizer(parameters, **config.settings)
    except Exception as e:
        raise AssertionError(f'Configuration for optimizer {config.name} incompatible with PyTorch:\n{e}')


def build_scheduler(optimizer: optim.Optimizer, config: ObjectDict) -> optim.lr_scheduler._LRScheduler:
    """
    Builds a scheduler on the given optimize from its PyTorch name and parameters.
    """
    try:
        return getattr(optim.lr_scheduler, config.name)(optimizer, **config.settings)
    except Exception as e:
        raise AssertionError(f'Configuration for scheduler {config.name} incompatible with PyTorch:\n{e}')


def optimizer_update(loss: torch.Tensor, optimizer: optim.Optimizer, scaler: Scaler, scheduler: Scheduler,
                     opt: ModelDict):
    """
    Updates the optimizer state from a given loss. Performs the backward operation on the loss.
    """
    if opt.amp:
        assert scaler is not None
        with autocast(enabled=False):
            scaled_loss = scaler.scale(loss)
            assert isinstance(scaled_loss, torch.Tensor)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
    else:
        loss.backward()
        optimizer.step()
    # TODO: improve handling of schedulers (epochs vs. iterations schedulers)
    if scheduler is not None:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_cast: optim.lr_scheduler.ReduceLROnPlateau = typing.cast(optim.lr_scheduler.ReduceLROnPlateau,
                                                                               scheduler)
            scheduler_cast.step(loss)
        else:
            scheduler.step()
