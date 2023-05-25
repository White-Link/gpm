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

from dataclasses import dataclass
from PIL import Image
from random import Random
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Callable, Sequence

from gpm.utils.config import ModelDict


@dataclass
class Batch:
    """
    Batches are dataclasses containing at least a tensor at field `x`.
    """
    x: torch.Tensor


Log = dict[str, dict | (int | float) | list[int | float] | str | Image.Image | Sequence[Image.Image] | torch.Tensor]

Optimizers = dict[str, torch.optim.Optimizer]
Scalers = dict[str, GradScaler] | None
Schedulers = dict[str, _LRScheduler]
Scaler = GradScaler | None
Scheduler = _LRScheduler | None

ForwardFn = Callable[[Batch | torch.Tensor, Any], Sequence[torch.Tensor]]
TrainingStep = Callable[[int, Batch, ForwardFn, Optimizers, Scalers, Schedulers, torch.device, Random, ModelDict],
                        tuple[float, Log, list[str]]]
