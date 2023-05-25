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


import enum
import torch

import functorch as ft
import torch.nn.functional as F

from typing import Callable

from gpm.utils.helper import fill_as


DiscrType = torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]


@enum.unique
class GradPenalty(str, enum.Enum):
    """
    Specifies on which data the gradient penalty is computed.
    """
    INTERPOLATION = 'interpolation'  # On an interpolation between fake and real data
    REAL = 'real'  # On real data
    FAKE = 'fake'  # On fake data


def vanilla_gan_discr_loss(fake: torch.Tensor, real: torch.Tensor,
                           discriminator: DiscrType) -> tuple[torch.Tensor, torch.Tensor]:
    discr_fake = discriminator(fake)
    discr_real = discriminator(real)
    labels_fake = torch.zeros(len(discr_fake), device=discr_fake.device)
    labels_real = torch.ones_like(labels_fake)
    loss_fake = F.binary_cross_entropy_with_logits(discr_fake, labels_fake, reduction='none')
    loss_real = -F.binary_cross_entropy_with_logits(discr_real, labels_real, reduction='none')
    return loss_fake, loss_real


def ipm_gan_discr_loss(fake: torch.Tensor, real: torch.Tensor,
                       discriminator: DiscrType) -> tuple[torch.Tensor, torch.Tensor]:
    discr_fake = discriminator(fake)
    discr_real = discriminator(real)
    return -discr_fake, -discr_real


def hinge_discr_loss(fake: torch.Tensor, real: torch.Tensor,
                     discriminator: DiscrType) -> tuple[torch.Tensor, torch.Tensor]:
    discr_fake = discriminator(fake)
    discr_real = discriminator(real)
    loss_fake = torch.clamp(1 + discr_fake, min=0)
    loss_real = torch.clamp(1 - discr_real, min=0)
    return loss_fake, -loss_real


def vanilla_gan_gen_loss(fake: torch.Tensor, discriminator: DiscrType) -> torch.Tensor:
    discr_fake = discriminator(fake)
    labels = torch.ones(len(discr_fake), device=discr_fake.device)
    return F.binary_cross_entropy_with_logits(discr_fake, labels, reduction='none')


def ipm_gan_gen_loss(fake: torch.Tensor, discriminator: DiscrType) -> torch.Tensor:
    return discriminator(fake)


def hinge_gan_gen_loss(fake: torch.Tensor, discriminator: DiscrType) -> torch.Tensor:
    return -discriminator(fake)


def grad_penalty(real: torch.Tensor, fake: torch.Tensor, discriminator: DiscrType, center: float, norm: float,
                 penalty_type: GradPenalty) -> torch.Tensor:
    if penalty_type is GradPenalty.INTERPOLATION:
        alpha = fill_as(torch.rand(len(real), device=real.device), real)
        grad_input = alpha * real + (1 - alpha) * fake
    elif penalty_type is GradPenalty.REAL:
        grad_input = real
    elif penalty_type is GradPenalty.FAKE:
        grad_input = fake
    else:
        raise ValueError(f'No gradient penalty named `{penalty_type}`')
    grad = ft.grad(lambda x: discriminator(x).sum())(grad_input)
    return (torch.linalg.norm(grad.flatten(start_dim=1), ord=norm, dim=1) - center) ** 2
