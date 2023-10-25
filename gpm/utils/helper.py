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


def fill_as(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Adds as many dimensions to the first tensor as it needs to reach the number of dimensions of the second tensor.
    Useful for broadcasting.
    """
    return x.view((*x.size(), *[1] * (len(target.size()) - len(x.size()))))
