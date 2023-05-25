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


import dataclasses

from torch.utils.data import default_collate
from typing import Any


def default_collate_dataclass(batch: list[Any]) -> Any:
    """
    Default PyTorch collate function extended to dataclasses.
    """
    elem = batch[0]
    if dataclasses.is_dataclass(batch[0]):
        elem_type = type(elem)
        return elem_type(*default_collate([samples for samples in map(dataclasses.astuple, batch)]))
    else:
        return default_collate(batch)
