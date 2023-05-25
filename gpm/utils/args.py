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


import argparse


seed_arg = 'seed'
data_arg = 'data_path'
configs_arg = 'configs'
load_arg = 'load'
load_best_arg = 'load_best'
only_test_arg = 'only_test'
save_arg = 'save_path'
name_arg = 'save_name'
erase_arg = 'erase'
train_log_freq_arg = 'train_log_freq'
log_save_freq_arg = 'log_save_freq'
no_chkpt_arg = 'no_chkpt'
amp_arg = 'amp'
deterministic_arg = 'deterministic'
no_benchmark_arg = 'no_benchmark'
device_arg = 'device'
workers_arg = 'nb_workers'
prefetch_arg = 'prefetch_factor'


def create_args(seed: bool = True) -> argparse.ArgumentParser:
    """
    Creates and returns the argument parser of the training program.
    """
    p = argparse.ArgumentParser(
        prog='GAN Gradient Flows',
        description='GAN Gradient Flows',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    config_p = p.add_argument_group(title='Configuration of the experiment',
                                    description='Seed, loading, data and configuration files.')
    if seed:
        config_p.add_argument(f'--{seed_arg}', type=int, metavar='SEED', default=None,
                              help='Manual seed. If None, it is chosen randomly.')
    config_p.add_argument(f'--{data_arg}', type=str, metavar='PATH', required=True,
                          help='Path where the data files are located.')
    config_p.add_argument(f'--{configs_arg}', type=str, metavar='FILE', required=True, nargs='+',
                          help='Configuration files. Duplicate parameters take the values from the rightmost file.')
    config_p.add_argument(f'--{load_arg}', action='store_true',
                          help='Whether to start the experiment from a loaded checkpoint.')
    config_p.add_argument(f'--{load_best_arg}', action='store_true',
                          help='Whether to load the best model.')
    config_p.add_argument(f'--{only_test_arg}', action='store_true',
                          help='Whether to disable training and simply test the model.')

    logs_p = p.add_argument_group(title='Logs',
                                  description='Logging options.')
    logs_p.add_argument(f'--{save_arg}', type=str, metavar='PATH', required=True,
                        help='Path where the experiment directory should be created.')
    logs_p.add_argument(f'--{name_arg}', type=str, metavar='FOLDER', required=True,
                        help='Experiment directory name.')
    logs_p.add_argument(f'--{erase_arg}', action='store_true',
                        help='Whether to automatically erase the previous experiment directory with the same name.')
    logs_p.add_argument(f'--{no_chkpt_arg}', action='store_true',
                        help='Whether to disable checkpoint saving.')
    logs_p.add_argument(f'--{train_log_freq_arg}', type=int, metavar='STEPS', default=1000,
                        help='Number of model training steps between each training log.')
    logs_p.add_argument(f'--{log_save_freq_arg}', type=int, metavar='STEPS', default=2,
                        help='Number of model evaluation steps between each save of logs and models.')

    devices_p = p.add_argument_group(title='Devices',
                                     description='Options for optimization on devices.')
    devices_p.add_argument(f'--{amp_arg}', action='store_true',
                           help='Whether to use PyTorch\'s integrated mixed-precision training.')
    devices_p.add_argument(f'--{deterministic_arg}', action='store_true',
                           help='Whether PyTorch operations must use deterministic algorithms.')
    devices_p.add_argument(f'--{no_benchmark_arg}', action='store_true',
                           help='Whether PyTorch should not benchmark cuDNN alorithms.')
    devices_p.add_argument(f'--{device_arg}', type=int, metavar='DEVICE', default=None, nargs='+',
                           help='If not None, indicates the list of GPU indices to use with CUDA.')
    devices_p.add_argument(f'--{workers_arg}', type=int, metavar='NB', default=4,
                           help='Number of child processes for data loading.')
    devices_p.add_argument(f'--{prefetch_arg}', type=int, metavar='NB', default=2,
                           help='Number of samples loaded in advance by each worker.')

    return p
