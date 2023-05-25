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


import math
import os
import random
import sys
import time
import torch
import traceback

import numpy as np
import torch.backends.cudnn as cudnn

from torch import distributed
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from gpm.data.base import BaseDataset
from gpm.data.factory import collate_fn_factory, dataset_factory_test, dataset_factory_train_val
from gpm.models.base import BaseModel
from gpm.models.factory import model_factory
from gpm.utils.args import create_args
from gpm.utils.config import ArgDict, ModelDict, load_yaml
from gpm.utils.load import load
from gpm.utils.logger import Logger
from gpm.utils.optimizer import build_scheduler
from gpm.utils.types import ForwardFn, Log, Optimizers, Scalers, Schedulers, TrainingStep


# Local rank in distributed training
try:
    local_rank = int(os.environ['LOCAL_RANK'])
except Exception:
    local_rank = 0


def init_compute_env(opt: ModelDict) -> tuple[torch.device, random.Random, Logger | None]:
    """
    Initializes the computing environment based on the configuration file:
     - sets up deterministic execution if required;
     - choses the device and initialzes the distributed environment in the case of multi-GPU training;
     - fixes the seeds and creates a shared random state accross all process for multi-GPU training;
     - creates the logger instance.
    """
    opt.info.hostname = os.uname()[1]

    # Set deterministic PyTorch and Python behavior if required
    torch.use_deterministic_algorithms(opt.deterministic)
    if opt.deterministic:
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'

    # For 'is_grads_batched' option of torch.autograd.grad
    torch._C._debug_only_display_vmap_fallback_warnings(True)  # type:ignore

    # Devices and distributed learning
    if opt.device is None:
        device = torch.device('cpu')
    else:
        assert len(opt.device) == len(set(opt.device))
        opt.n_gpu = len(opt.device)
        device = torch.device(f'cuda:{opt.device[local_rank]}')
        assert cudnn.is_available()
        cudnn.benchmark = not opt.no_benchmark
        if opt.n_gpu > 1 or local_rank > 0:
            distributed.init_process_group(distributed.Backend.NCCL, world_size=opt.n_gpu)
            # Seeds must be equal between processes
            assert opt.seed is not None
            # The world size should correspond to the number of GPUs
            assert opt.n_gpu == distributed.get_world_size()
            # In distributed training, divide batch size by the number of GPUs
            assert opt.optim.batch_size % opt.n_gpu == 0
            opt.optim.batch_size = opt.optim.batch_size // opt.n_gpu

    # Seed
    if opt.seed is None:
        opt.seed = torch.seed()
    random.seed((opt.seed + local_rank) % (2**32 - 1))
    np.random.seed((opt.seed + local_rank) % (2**32 - 1))
    torch.manual_seed((opt.seed + local_rank) % (2**32 - 1))
    # Shared RNG state, needed because seeds differ between training processes
    if opt.n_gpu > 1:
        shared_rng = random.Random((opt.seed + distributed.get_world_size()) % (2**32 - 1))
    else:
        shared_rng = random.Random(opt.seed)

    print(f"Learning on {opt.n_gpu} GPU(s) (seed: {opt.seed})")

    # Logger
    if local_rank == 0:
        logger = Logger(opt, not opt.no_chkpt, opt.save_best)
    else:
        logger = None
    if opt.n_gpu > 1 or local_rank > 0:
        distributed.barrier()

    return device, shared_rng, logger


def load_data(opt: ModelDict) -> tuple[BaseDataset, DataLoader, tuple[DataLoader] | None, tuple[DataLoader] | None,
                                       DistributedSampler | None]:
    """
    Loads the training, validation and test data, creates their dataloaders and sets up a specific training data
    sampler for multi-GPU training. Returns the training dataset, respective dataloaders, and training sampler.

    There are as many validation and test dataloaders as there as validation and test configurations in the
    configuration file.
    """
    print('Loading data...')
    assert opt.seed is not None
    train_set, val_sets = dataset_factory_train_val(opt, opt.seed)
    collate_fn = collate_fn_factory(train_set)

    # Training dataloader
    train_sampler = None
    loader_shuffle = True
    if opt.n_gpu > 1:
        train_sampler = DistributedSampler(train_set)
        loader_shuffle = False
    assert 0 < opt.optim.batch_size <= len(train_set) and opt.nb_workers >= 0
    train_loader = DataLoader(train_set, batch_size=opt.optim.batch_size, shuffle=loader_shuffle,
                              sampler=train_sampler, num_workers=opt.nb_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True, prefetch_factor=opt.prefetch_factor,  # type: ignore
                              persistent_workers=opt.nb_workers > 0)
    print(f'{len(train_loader)} iterations per epoch.')

    # Evaluation only on master
    val_loaders = None
    if local_rank == 0:
        val_loaders = []
        for eval_config, val_set in zip(opt.eval, val_sets):
            assert 0 < eval_config.batch_size <= len(val_set)
            val_loader = DataLoader(val_set, batch_size=eval_config.batch_size, shuffle=True,
                                    num_workers=opt.nb_workers, collate_fn=collate_fn, pin_memory=True,
                                    prefetch_factor=opt.prefetch_factor)  # type: ignore
            val_loaders.append(val_loader)
            assert eval_config.nb_steps <= len(val_loader)
            if eval_config.nb_steps <= 0:
                eval_config.nb_steps = len(val_loader)
            assert eval_config.batch_size * eval_config.nb_steps >= eval_config.nb_quali
        val_loaders = tuple(val_loaders)

    # Test only on master
    test_loaders = None
    if local_rank == 0:
        test_loaders = []
        test_sets = dataset_factory_test(opt)
        for test_config, test_set in zip(opt.test, test_sets):
            assert 0 < test_config.batch_size <= len(test_set)
            test_loader = DataLoader(test_set, batch_size=test_config.batch_size, num_workers=opt.nb_workers,
                                     collate_fn=collate_fn, pin_memory=True,
                                     prefetch_factor=opt.prefetch_factor)  # type: ignore
            test_loaders.append(test_loader)
            assert test_config.nb_steps <= len(test_loader)
            if test_config.nb_steps <= 0:
                test_config.nb_steps = len(test_loader)
            assert test_config.batch_size * test_config.nb_steps >= test_config.nb_quali
        test_loaders = tuple(test_loaders)

    return train_set, train_loader, val_loaders, test_loaders, train_sampler


def build_model(opt: ModelDict, dataset: BaseDataset) -> BaseModel:
    """
    Instantiates the model as a single PyTorch module.

    Makes batch normalization layers synchronous for multi-GPU training.
    """
    print('Building model...')
    model = model_factory(opt, dataset)
    opt.info.model_object = str(model)
    nb_parameters = sum(p.numel() for p in model.parameters())
    opt.info.nb_parameters = nb_parameters
    print(f'{nb_parameters} parameters')
    if opt.n_gpu > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        assert isinstance(model, BaseModel)
    return model


def build_optimizers(model: BaseModel, opt: ModelDict) -> tuple[Optimizers, Schedulers, Scalers]:
    """
    Instantiates the optimizers required by the model with options provided in the configuration file. Optionnally adds
    learning rate schedulers and gradient scalers (in the latter case, for mixed-precision training.)
    """
    optimizers = model.build_optimizers(opt)
    schedulers = {}
    scalers = {}
    for optimizer_name, optimizer in optimizers.items():
        if optimizer_name in opt.optim.schedulers:
            schedulers[optimizer_name] = build_scheduler(optimizer, opt.optim.schedulers[optimizer_name])
        if opt.amp:
            scalers[optimizer_name] = GradScaler()
    opt.optimizers_object = str(optimizers)
    opt.schedulers_object = str(schedulers)
    if len(scalers) == 0:
        scalers = None
    return optimizers, schedulers, scalers


def setup(opt: ModelDict) -> tuple[torch.device, random.Random, Logger | None, DataLoader, tuple[DataLoader] | None,
                                   tuple[DataLoader] | None, DistributedSampler | None, BaseModel, Optimizers,
                                   Schedulers, Scalers, TrainingStep, ForwardFn, int]:
    """
    Instantiates all needed objects for training. Returns, among other things, the forward method of the model.

    Handles `DistributedDataParallel` for multi-GPU training and model loading if requested.
    """
    # Initialize environment
    device, shared_rng, logger = init_compute_env(opt)
    # Data loading
    train_set, train_loader, val_loaders, test_loaders, train_sampler = load_data(opt)
    # Model
    model = build_model(opt, train_set)
    # Optimizer and related objects
    optimizers, schedulers, scalers = build_optimizers(model, opt)
    # Multi-GPU and model setup
    model.to(device)
    if opt.load:
        step = load(opt, model, optimizers, schedulers, scalers, shared_rng)
    else:
        step = 0
    model.set_optimizers(optimizers)
    training_step = model.training_step
    if opt.n_gpu > 1:
        forward_fn = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        forward_fn = model
    return (device, shared_rng, logger, train_loader, val_loaders, test_loaders, train_sampler, model, optimizers,
            schedulers, scalers, training_step, forward_fn, step)


def evaluate(opt: ModelDict, device: torch.device, val_loaders: tuple[DataLoader], model: BaseModel,
             step: int, test: bool) -> tuple[float, Log]:
    """
    Performs model evaluation on the provided dataloaders and returns corresponding logs. Retains the score of the last
    evaluation.
    """
    score = None
    logs_val = {}
    for eval_config, val_loader in zip(opt.eval if not test else opt.test, val_loaders):
        val_score, log_val = model.evaluation(val_loader, device, opt, eval_config)
        if score is None:
            score = val_score
        logs_val[eval_config.config_name] = log_val
    logs_val['step'] = step
    assert score is not None
    return score, logs_val


def training_loop(opt: ModelDict, device: torch.device, shared_rng: random.Random, logger: Logger | None,
                  train_loader: DataLoader, val_loaders: tuple[DataLoader] | None,
                  test_loaders: tuple[DataLoader] | None, train_sampler: DistributedSampler | None, model: BaseModel,
                  optimizers: Optimizers, schedulers: Schedulers, scalers: Scalers, training_step: TrainingStep,
                  forward_fn: ForwardFn, step: int) -> int:
    """
    Trains, periodically validates, and finally tests the model.
    """
    status_code = 0

    # Setup
    print('Training...')
    assert opt.optim.nb_steps > 0
    assert opt.train_log_freq > 0
    if logger is not None:
        logger.initialize(opt)
        pb = tqdm(initial=step, total=opt.optim.nb_steps, ncols=0)
    else:
        pb = None
    log_train = None
    score = None
    finished = opt.only_test
    start = time.time()

    # Main loop
    try:
        while not finished:
            if train_sampler is not None:
                train_sampler.set_epoch(step)

            # ---------- TRAIN ----------
            for batch in train_loader:
                # Stop training when needed
                if step >= opt.optim.nb_steps:
                    finished = True
                    break
                step += 1
                opt.info.step = step

                # Step
                with autocast(enabled=opt.amp):
                    loss, new_log_train, pb_vars = training_step(step, batch, forward_fn, optimizers, scalers,
                                                                 schedulers, device, shared_rng, opt)
                    model.post_training_step(step, shared_rng)
                    if log_train is None:
                        log_train = new_log_train
                    else:
                        log_train.update(new_log_train)
                if not math.isfinite(loss):
                    status_code = 34
                    finished = True
                    break

                # Log train
                if step % opt.train_log_freq == 0 and logger is not None:
                    # log train
                    log_train['step'] = step
                    log_train['time'] = time.time() - start
                    logger.log(step, 'train', log_train)

                # Progress bar
                if pb is not None:
                    pb.set_postfix(**{v: log_train[v] for v in pb_vars if v in log_train}, val_score=score,
                                   refresh=False)
                    pb.update()

                # Checkpoint
                if step % opt.eval_freq == 0:
                    break

            # ---------- EVAL ----------
            # Only on one GPU
            if step % opt.eval_freq == 0:
                if val_loaders is not None and logger is not None:
                    assert local_rank == 0
                    model.eval()
                    with torch.inference_mode():
                        score, logs_val = evaluate(opt, device, val_loaders, model, step, False)
                        logger.log(step, 'eval', logs_val)
                    logger.checkpoint(step, model, optimizers, scalers, schedulers, shared_rng, score)
                    model.train()
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        status_code = 130

    # Final tests and logging, only on one GPU
    if local_rank == 0:
        assert pb is not None and logger is not None
        pb.close()
        model.eval()
        if status_code == 34:
            print('Numerical error. Saving logs to disk...')
            logger.terminate(step, None, None, None, None, None, status_code)
        else:
            print('Evaluating and saving to disk...')
            with torch.inference_mode():
                assert val_loaders is not None and test_loaders is not None
                score, logs_val = evaluate(opt, device, val_loaders, model, step, False)
                _, logs_test = evaluate(opt, device, test_loaders, model, step, True)
            if log_train is not None:
                logger.log(step, 'train', log_train)
            logger.log(step, 'eval', logs_val)
            logger.log(step, 'test', logs_test)
            logger.checkpoint(step, model, optimizers, scalers, schedulers, shared_rng, score)
            logger.terminate(step, model, optimizers, scalers, schedulers, shared_rng, status_code)
    print('Done')
    return status_code


def main(opt: ModelDict) -> int:
    """
    Sets up all needed objects for training, and launches model training.
    """
    # Setup
    try:
        (device, shared_rng, logger, train_loader, val_loaders, test_loaders, train_sampler, model, optimizers,
         schedulers, scalers, training_step, forward_fn, step) = setup(opt)
    except AssertionError:
        # https://stackoverflow.com/questions/11587223/how-to-handle-assertionerror-in-python-and-find-out-which-line-or-statement-it-o
        _, _, tb = sys.exc_info()
        tb_info = traceback.extract_tb(tb)
        filename, line, _, text = tb_info[-1]
        print('The configutation file is not valid:')
        print(f'File `{filename}`, line {line}')
        print(text)
        print('Aborting experiment')
        status_code = 22
        return status_code

    # Training
    return training_loop(opt, device, shared_rng, logger, train_loader, val_loaders, test_loaders, train_sampler,
                         model, optimizers, schedulers, scalers, training_step, forward_fn, step)


if __name__ == '__main__':
    # Arguments
    p = create_args()
    opt = ArgDict(vars(p.parse_args()))

    # Disable output for all processes but one for multi-GPU training
    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")

    # Configuration
    for config_file in opt.configs:
        config = load_yaml(config_file)
        opt.update(config)
    opt = ModelDict(opt)

    # Main
    main(opt)
