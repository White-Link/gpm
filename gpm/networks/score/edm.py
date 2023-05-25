# Copyright (c) 2023 Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickaël Chen, Alain Rakotomamonjy
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/


# Adapted Nvidia code from https://github.com/NVlabs/edm (file `training/networks`).

# Initially released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license and
# redistributed under the same license (see attached file).

# The original work included the following copyright notice.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# Changes:
#  - Removed unused module definitions.
#  - Adapted the code to the coding style of this program (typing, spacing, naming, etc.).


import torch

from gpm.networks.score.scalar_emb import NetworkScalarEmbedding, NoiseEncoding
from gpm.utils.helper import fill_as


class EDMDenoiser(NetworkScalarEmbedding):
    """
    EDM preconditioning of a U-Net. Cf. the EDM code and paper.
    """

    def __init__(self, sigma_min: float, sigma_max: float, sigma_data: float, model: torch.nn.Module,
                 embedding: NoiseEncoding, embedding_size: int):
        super().__init__(model, embedding, embedding_size)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        c_in, c_skip, c_out = fill_as(c_in, x), fill_as(c_skip, x), fill_as(c_out, x)
        F_x = super().forward(c_in * x, c_noise)
        D_x = c_skip * x + c_out * F_x
        return D_x
