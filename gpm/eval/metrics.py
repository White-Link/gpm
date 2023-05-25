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

from torchmetrics.image.fid import FrechetInceptionDistance


def compute_fid(x: torch.Tensor, x_gen: torch.Tensor, batch_size: int, device: torch.device) -> float:
    """
    Comptes the FID between two sets of samples with the given batch size.
    """
    assert len(x.size()) == 4 and x.size(1) in [1, 3]
    if x.size(1) == 1:
        # Deal with grayscale images
        x = torch.cat([x] * 3, dim=1)
        x_gen = torch.cat([x_gen] * 3, dim=1)
    # Scales to [0, 1] and clamps values to comply with torchmetrics
    x = torch.clamp((1 + x) / 2, 0, 1)
    x_gen = torch.clamp((1 + x_gen) / 2, 0, 1)
    # Compute statistics by batch and then compute the FID
    fid = FrechetInceptionDistance(normalize=True, sync_on_compute=False).to(device)
    for x_batch, x_gen_batch in zip(x.split(batch_size), x_gen.split(batch_size)):
        fid.update(x_gen_batch.to(device), real=False)
        fid.update(x_batch.to(device), real=True)
    fid_value = fid.compute()
    return fid_value.item()
