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


import copy
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
import torch
import warnings

from collections import OrderedDict
from collections.abc import MutableMapping, Sequence
from os.path import join
from PIL import Image
from typing import Any

from gpm.utils.config import ModelDict
from gpm.utils.types import Log, Optimizers, Scalers, Schedulers


# Always show custom runtime warnings
warnings.filterwarnings('always', category=RuntimeWarning)


def dict_to_cpu(dictionary: MutableMapping) -> dict:
    cpu_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, dict):
            cpu_dict[k] = dict_to_cpu(v)
        else:
            if isinstance(v, torch.Tensor):
                cpu_dict[k] = v.cpu()
            else:
                cpu_dict[k] = v
    return cpu_dict


def last_entry(dictionary: MutableMapping) -> dict:
    last_entry_dict = OrderedDict()
    for k, v in dictionary.items():
        if isinstance(v, Sequence):
            last_entry_dict[k] = v[-1]
        elif isinstance(v, MutableMapping):
            last_entry_dict[k] = last_entry(v)
        else:
            last_entry_dict[k] = v
    return last_entry_dict


def save_pt(pt_object, path: str) -> bool:
    try:
        torch.save(pt_object, path)
        success = True
    except Exception as e:
        warnings.warn(f'Could not save PyTorch object at `{path}`, encountered error:\n{e}', category=RuntimeWarning)
        success = False
    return success


def save_pickle(any_object: Any, path: str) -> bool:
    try:
        with open(path, 'wb') as file:
            pickle.dump(any_object, file, pickle.HIGHEST_PROTOCOL)
        success = True
    except Exception as e:
        warnings.warn(f'Could not pickle save object at `{path}`, encountered error:\n{e}', category=RuntimeWarning)
        success = False
    return success


def save_json(data_dict: MutableMapping, path: str) -> bool:
    ordered = isinstance(data_dict, OrderedDict)
    try:
        with open(path, 'w') as f:
            json.dump(data_dict, f, sort_keys=not ordered, indent=4)
        success = True
    except Exception as e:
        warnings.warn(f'Could not save JSON object at `{path}`, encountered error:\n{e}', category=RuntimeWarning)
        success = False
    return success


def save_src(path: str) -> bool:
    try:
        shutil.make_archive(path, 'zip')
        success = True
    except Exception as e:
        warnings.warn(f'Could not save source at `{path}`, encountered error:\n{e}', category=RuntimeWarning)
        success = False
    return success


class Logger(object):
    """
    Logger object: stores all logged variables (numbers, images, tensors, etc.), saves them with the model in the
    experiment folder. When requested, also tracks and saves the best model. Additionnally saves source code and
    parameters.
    """

    def __init__(self, opt: ModelDict, save_model: bool, save_best: bool):
        super().__init__()

        # Creation of experiment folder
        self.exp_dir = join(opt.save_path, opt.save_name)
        self.chkpt_dir = join(self.exp_dir, 'chkpt')
        print(f'Experiment directory: `{self.exp_dir}`.')
        if os.path.isdir(self.exp_dir) and not opt.load:
            if not opt.erase and input(f'Experiment directory `{self.exp_dir}` already exists. Remove? (y|n) ') != 'y':
                sys.exit()
            shutil.rmtree(self.exp_dir)
        if not os.path.isdir(self.exp_dir):
            if opt.load:
                raise FileExistsError(f'Experiment directory `{self.exp_dir}` does not exist.')
            else:
                os.makedirs(self.chkpt_dir)
                print('Experiment directory created.')

        # Attributes and parameters
        self.logs = OrderedDict()
        self.save_model = save_model
        self.save_best = save_best
        self.best_results = None
        if self.save_best:
            self.best_score = -math.inf
            self.best_step = None
            self.best_model = None
            self.best_optimizers = None
            self.best_scalers = None
            self.best_schedulers = None
            self.best_shared_rng = None
        self.nb_checkpoints = 0
        self.log_save_freq = opt.log_save_freq
        assert self.log_save_freq > 0
        self.path_buffer: list[str] = []
        self.data_buffer: list[str | Image.Image | Sequence[Image.Image] | torch.Tensor] = []
        self.nb_tries = 5

    def initialize(self, opt: ModelDict):
        """
        Save configuration and code.
        """
        opt.info.running = True
        for config_file in opt.configs:
            shutil.copy(config_file, self.exp_dir)
        save_json(opt, join(self.exp_dir, 'config.json'))
        save_src(join(self.exp_dir, 'source'))

    def log(self, step: int, key: str,
            val: Log | (int | float) | list[int | float] | str | Image.Image | Sequence[Image.Image] | torch.Tensor,
            previous_keys: list[str] = [], log_dict: OrderedDict | None = None):
        """
        Logs a value under a given key.
        """
        log_dict = self.logs if log_dict is None else log_dict
        if isinstance(val, dict):
            if key not in log_dict:
                log_dict[key] = OrderedDict()
            for k, v in val.items():
                self.log(step, k, v, previous_keys=previous_keys + [key], log_dict=log_dict[key])
        else:
            if key not in log_dict:
                log_dict[key] = []
            if isinstance(val, (int, float)):
                log_dict[key].append(val)
            elif isinstance(val, str):
                data_path = join(self.exp_dir, *previous_keys, f'{key}_{step:08}.txt')
                log_dict[key].append(data_path)
                self.path_buffer.append(data_path)
                self.data_buffer.append(val)
            elif isinstance(val, Image.Image) or (isinstance(val, Sequence) and len(val) > 0
                                                  and isinstance(val[0], Image.Image)):
                data_path = join(self.exp_dir, *previous_keys, f'{key}_{step:08}.png')
                log_dict[key].append(data_path)
                self.path_buffer.append(data_path)
                self.data_buffer.append(val)  # type: ignore
            elif isinstance(val, torch.Tensor):
                data_path = join(self.exp_dir, *previous_keys, f'{key}_{step:08}.pt')
                log_dict[key].append(data_path)
                self.path_buffer.append(data_path)
                self.data_buffer.append(val)  # type: ignore
            elif isinstance(val, list):
                for v in val:
                    assert isinstance(val, (int, float)), f'Trying to log {key}, but it is not a number.'
                log_dict[key].append(val)
            else:
                raise TypeError(f'Failed to log `{key}`. Logging `{type(val)}` is not supported.')

    def _dump(self, step: int, model: torch.nn.Module | None, optimizers: Optimizers | None, scalers: Scalers,
              schedulers: Schedulers | None, shared_rng: random.Random | None) -> bool:
        """
        Saves logged variables in the buffer.
        """
        success = True

        with open(join(self.chkpt_dir, 'step.txt'), 'w') as f:
            f.write(str(step))
        if self.save_best and self.best_step is not None:
            with open(join(self.chkpt_dir, 'step_best.txt'), 'w') as f:
                f.write(str(self.best_step))

        # Save PyTorch objects
        if (
            self.save_model and model is not None and optimizers is not None and schedulers is not None
            and shared_rng is not None
        ):
            # Save models
            success = success and save_pt(model.state_dict(), join(self.chkpt_dir, 'model.pt'))
            if self.save_best and self.best_model is not None:
                success = success and save_pt(self.best_model, join(self.chkpt_dir, 'model_best.pt'))
            # Save optimizers
            for name in optimizers.keys():
                success = success and save_pt(optimizers[name].state_dict(),
                                              join(self.chkpt_dir, f'{name}_optimizer.pt'))
                if self.save_best and self.best_optimizers is not None:
                    success = success and save_pt(self.best_optimizers[name],
                                                  join(self.chkpt_dir, f'{name}_optimizer_best.pt'))
            # Save scalers
            if scalers is not None:
                for name in scalers.keys():
                    success = success and save_pt(scalers[name].state_dict(),
                                                  join(self.chkpt_dir, f'{name}_scaler.pt'))
                    if self.save_best and self.best_scalers is not None:
                        success = success and save_pt(self.best_scalers[name],
                                                      join(self.chkpt_dir, f'{name}_scaler_best.pt'))
            # Save schedulers
            for name in schedulers.keys():
                success = success and save_pt(schedulers[name], join(self.chkpt_dir, f'{name}_scheduler.pt'))
                if self.save_best and self.best_schedulers is not None:
                    success = success and save_pt(self.best_schedulers[name],
                                                  join(self.chkpt_dir, f'{name}_scheduler_best.pt'))

            # Save shared RNG
            success = success and save_pickle(shared_rng.getstate(), join(self.chkpt_dir, 'shared_rng.pickle'))
            if self.save_best and self.best_shared_rng is not None:
                success = success and save_pickle(self.best_shared_rng.getstate(),
                                                  join(self.chkpt_dir, 'shared_rng_best.pickle'))

        # Write logs on JSON file
        for k, v in self.logs.items():
            success = success and save_json(v, join(self.exp_dir, f'logs.{k}.json'))
        # Write results (last entry)
        results = last_entry(self.logs)
        results['checkpoint'] = self.nb_checkpoints
        success = success and save_json(results, join(self.exp_dir, 'results.json'))
        if self.save_best and self.best_results is not None:
            success = success and save_json(self.best_results, join(self.exp_dir, 'results_best.json'))

        # Empty data buffer
        for path, data in zip(self.path_buffer, self.data_buffer):
            try:
                if not os.path.isdir(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                if isinstance(data, str):
                    with open(path, 'w') as f:
                        f.write(data)
                elif isinstance(data, Image.Image):
                    data.save(path)
                elif isinstance(data, Sequence) and len(data) > 0 and isinstance(data[0], Image.Image):
                    data[0].save(path, save_all=True, append_images=data[1:], duration=int(4 / len(data) * 1000),
                                 loop=0)
                elif isinstance(data, torch.Tensor):
                    torch.save(data, path)
                else:
                    raise ValueError('Unrecognized log type.')
            except Exception as e:
                warnings.warn(f'Could not save data object at `{path}`, encountered error:\n{e}',
                              category=RuntimeWarning)
                success = False
        self.path_buffer, self.data_buffer = [], []

        return success

    def checkpoint(self, step: int, model: torch.nn.Module, optimizers: Optimizers, scalers: Scalers,
                   schedulers: Schedulers, shared_rng: random.Random, score: float):
        """
        Saves a checkpoint and, depending on the iteration, the logged variables.
        """
        self.nb_checkpoints += 1
        if self.save_best:
            is_best = True if self.nb_checkpoints == 1 else score > self.best_score
            if is_best:
                self.best_score = score
                self.best_checkpoint = self.nb_checkpoints
                # Save best model
                self.best_step = step
                self.best_model = dict_to_cpu(copy.deepcopy(model.state_dict()))
                self.best_optimizers = {
                    name: dict_to_cpu(copy.deepcopy(optim.state_dict())) for name, optim in optimizers.items()
                }
                if scalers is not None:
                    self.best_scalers = {
                        name: dict_to_cpu(copy.deepcopy(scaler.state_dict())) for name, scaler in scalers.items()
                    }
                self.best_schedulers = {
                    name: copy.deepcopy(scheduler) for name, scheduler in schedulers.items()
                }
                self.best_shared_rng = copy.deepcopy(shared_rng)
                # Write best results (last entry)
                self.best_results = last_entry(self.logs)
                self.best_results['checkpoint'] = self.best_checkpoint
                self.best_results['score'] = self.best_score
        if self.nb_checkpoints % self.log_save_freq == 0:
            self._dump(step, model, optimizers, scalers, schedulers, shared_rng)

    def terminate(self, step: int, model: torch.nn.Module | None, optimizers: Optimizers | None, scalers: Scalers,
                  schedulers: Schedulers | None, shared_rng: random.Random | None, status_code: int):
        """
        Saves final checkpoint and logs.
        """
        self.nb_checkpoints += 1
        success = False
        i = 0
        pause = 300
        while not success:
            i += 1
            if i > 1:
                if i <= self.nb_tries:
                    print('Failed to save final checkpoint, retrying...')
                    time.sleep(pause)
                    pause *= 2
                else:
                    input('Failed to save final checkpoint. Press any key to retry.')
            try:
                config_path = join(self.exp_dir, 'config.json')
                with open(config_path, 'r') as info:
                    opt = json.load(info)
                opt['running'] = False
                opt['status_code'] = status_code
                success = save_json(opt, config_path)
            except Exception:
                success = False
            if not success:
                continue
            success = self._dump(step, model, optimizers, scalers, schedulers, shared_rng)
