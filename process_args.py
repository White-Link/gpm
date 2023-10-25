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


import sys
import random

from gpm.utils.args import create_args


if __name__ == '__main__':
    args = " ".join(sys.argv[1:])

    # Arguments
    p = create_args(seed=False)
    parsed_args = p.parse_args()

    # Seed
    seed = random.getrandbits(32)

    # Launch utility
    if parsed_args.device is not None and len(parsed_args.device) > 1:
        launch = f'OMP_NUM_THREADS={max(parsed_args.nb_workers, 1)} ' \
                 f'torchrun --standalone --nnodes 1 --nproc_per_node {len(parsed_args.device)}'
    else:
        launch = 'python'

    # Bash command
    command = f'{launch} -m gpm.train {args} --seed {seed}'

    print(command)
