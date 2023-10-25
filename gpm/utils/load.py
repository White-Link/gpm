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


import pickle
import random
import torch

from os.path import join

from gpm.models.base import BaseModel
from gpm.utils.config import ModelDict
from gpm.utils.types import Optimizers, Scalers, Schedulers


def load(opt: ModelDict, model: BaseModel, optimizers: Optimizers, schedulers: Schedulers, scalers: Scalers,
         shared_rng: random.Random) -> int:
    """
    Loads a checkpoint.
    """
    exp_dir = join(opt.save_path, opt.save_name)
    chkpt_dir = join(exp_dir, 'chkpt')
    best_suffix = '_best' if opt.load_best else ''

    with open(join(chkpt_dir, f'step{best_suffix}.txt'), 'r') as file:
        step = int(file.readline())

    model.load_state_dict(torch.load(join(chkpt_dir, f'model{best_suffix}.pt'), map_location='cpu'))
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(torch.load(join(chkpt_dir, f'{name}_optimizer{best_suffix}.pt'), map_location='cpu'))
    if scalers is not None:
        for name, scaler in scalers.items():
            scaler.load_state_dict(torch.load(join(chkpt_dir, f'{name}_scaler{best_suffix}.pt'), map_location='cpu'))
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(torch.load(join(chkpt_dir, f'{name}_scheduler{best_suffix}.pt'), map_location='cpu'))
    with open(join(chkpt_dir, f'shared_rng{best_suffix}.pickle'), 'rb') as file:
        shared_rng_state = pickle.load(file)
        shared_rng.setstate(shared_rng_state)

    return step
