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

from gpm.models.discr_flow.base import DiscrFlow
from gpm.models.gan.losses import (hinge_discr_loss, ipm_gan_discr_loss, hinge_gan_gen_loss, ipm_gan_gen_loss,
                                   vanilla_gan_discr_loss, vanilla_gan_gen_loss)


class VanillaDiscrFlow(DiscrFlow):
    def discr_loss(self, fake: torch.Tensor, real: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return vanilla_gan_discr_loss(fake, real, self.discriminator_t(t))

    def gen_loss(self, fake: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return vanilla_gan_gen_loss(fake, self.discriminator_t(t))


class IPMDiscrFlow(DiscrFlow):
    def discr_loss(self, fake: torch.Tensor, real: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return ipm_gan_discr_loss(fake, real, self.discriminator_t(t))

    def gen_loss(self, fake: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ipm_gan_gen_loss(fake, self.discriminator_t(t))


class HingeDiscrFlow(DiscrFlow):
    def discr_loss(self, fake: torch.Tensor, real: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return hinge_discr_loss(fake, real, self.discriminator_t(t))

    def gen_loss(self, fake: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return hinge_gan_gen_loss(fake, self.discriminator_t(t))
