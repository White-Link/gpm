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


from __future__ import annotations

import enum
import random
import torch

import functorch as ft

from abc import abstractmethod
from contextlib import nullcontext
from typing import Iterator

from gpm.data.lowd.base import BaseLowDDataset
from gpm.data.image.base import BaseImageDataset
from gpm.eval.interpolate import Interpolation, interpolate
from gpm.eval.metrics import compute_fid
from gpm.eval.quali import gen_quali, inter_quali
from gpm.eval.tests import Test
from gpm.models.base import BaseModel
from gpm.models.gan.losses import GradPenalty, grad_penalty
from gpm.networks.activations import Activation
from gpm.networks.factory import encoder_factory, decoder_factory
from gpm.networks.init import InitDict, init_network
from gpm.networks.utils import EncoderDict, DecoderDict
from gpm.utils.config import DotDict, ModelDict, TestDict
from gpm.utils.helper import fill_as
from gpm.utils.optimizer import optimizer_update
from gpm.utils.types import Batch, ForwardFn, Log, Optimizers, Scalers, Schedulers


EvaluationTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@enum.unique
class GANModel(str, enum.Enum):
    """
    GAN models: this choice determines the generator and discriminator loss.
    """
    VANILLA = 'vanilla'
    IPM = 'ipm'
    HINGE = 'hinge'


class GANDict(DotDict):
    """
    Parameter dictionary for GANs.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'model', 'latent_dim', 'nb_discr_steps', 'discriminator', 'generator', 'init'}.issubset(self.keys())
        assert self.model in [model.value for model in GANModel]
        self.model: GANModel = GANModel(self.model)  # GAN loss: specificies discr. and gen. loss
        self.latent_dim: int  # Dimensionality of the generator's latent space
        self.nb_discr_steps: int  # Number of discr. optimization steps in-between generator updates
        self.discriminator: EncoderDict = EncoderDict(self.discriminator)  # Discriminator architecture
        self.generator: DecoderDict = DecoderDict(self.generator)  # Generator architecture
        self.init: InitDict = InitDict(self.init)  # Discr. and gen. initialization parameters

        self.grad_penalty: float  # Strength of gradient penalty (default, 0)
        self.grad_penalty_center: float  # Center of gradient penalty (default, 0)
        self.grad_penalty_norm: float  # Norm of gradient penalty (default, 2)
        # Choice of gradient penalty application (on fake, real, or interpolated data, default to the latter)
        self.grad_penalty_type: GradPenalty  # Strength of added noise at each generation step (default, 0)
        if 'grad_penalty' not in self:
            self.grad_penalty = 0
        if 'grad_penalty_center' not in self:
            self.grad_penalty_center = 1
        if 'grad_penalty_norm' not in self:
            self.grad_penalty_norm = 2
        if 'grad_penalty_type' not in self:
            self.grad_penalty_type = GradPenalty.INTERPOLATION
        else:
            self.grad_penalty_type = GradPenalty(self.grad_penalty_type)

        # Multiplier of the generator loss (default, 1)
        self.gen_loss_scale: float
        if 'gen_loss_scale' not in self:
            self.gen_loss_scale: float = 1


class GANInferenceDict(DotDict):
    """
    Inference parameter dictionary for GANs.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        # Resolution of pointwise generator loss for qualitative tests on 2D data, or number of interpolations
        self.resolution: int
        self.interpolation: Interpolation  # Interpolation method to use for qualitative interpolation tests
        if 'resolution' not in self:
            self.resolution = 200
        if 'interpolation' in self:
            self.interpolation = Interpolation(self.interpolation)
        else:
            self.interpolation = Interpolation.LINEAR


class GAN(BaseModel):
    """
    GAN model.
    """

    gen_optimizer_name = 'gen_optimizer'
    discr_optimizer_name = 'discr_optimizer'

    @enum.unique
    class Op(enum.Enum):
        """
        Operation indicator for the unique forward method.
        """
        DISCRIMINATOR = enum.auto()  # Discriminator optim.
        GENERATOR = enum.auto()  # Generator optim.

    def __init__(self, dataset: BaseLowDDataset | BaseImageDataset, config: GANDict):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.nb_discr_steps = config.nb_discr_steps
        self.grad_penalty = config.grad_penalty
        self.grad_penalty_center = config.grad_penalty_center
        self.grad_penalty_norm = config.grad_penalty_norm
        self.grad_penalty_type = config.grad_penalty_type
        self.init = config.init
        self.dataset = dataset

        if isinstance(dataset, BaseImageDataset):
            in_channels, in_size = dataset.channels, dataset.image_size
            data_size = in_channels, in_size, in_size
        else:
            data_size = dataset.dim,
        # Build generator
        self.generator = decoder_factory(dataset, config.latent_dim, data_size, config.generator)
        # Check that the discriminator does not end with an activation
        assert (
            config.discriminator.final_activation is None
            or config.discriminator.final_activation.name is Activation.IDENTITY
        )
        # Build discriminator
        self.discriminator = self.build_discriminator(data_size, config)
        # Initialize the generator and the discriminator
        init_network([self.discriminator, self.generator], config.init)

    def build_discriminator(self, data_size: tuple[int] | tuple[int, int, int], config: GANDict) -> torch.nn.Module:
        """
        Builds the discriminator.
        """
        return torch.nn.Sequential(encoder_factory(self.dataset, data_size, 1, config.discriminator),
                                   torch.nn.Flatten(start_dim=0))

    def optimizers_names(self, opt: ModelDict) -> set[str]:
        return {GAN.gen_optimizer_name, GAN.discr_optimizer_name}

    def optimized_params(self, optimizer_name: str, opt: ModelDict) -> Iterator[torch.nn.parameter.Parameter]:
        if optimizer_name == GAN.gen_optimizer_name:
            return self.generator.parameters()
        elif optimizer_name == GAN.discr_optimizer_name:
            return self.discriminator.parameters()
        else:
            raise ValueError(f'No optimizer named `{optimizer_name}`')

    def sample_latent(self, nb_gen: int, device: torch.device) -> torch.Tensor:
        """
        Samples a batch of latent states.
        """
        return torch.normal(0, 1, (nb_gen, self.latent_dim), device=device)

    def generate(self, nb_gen: int, device: torch.device, no_grad: bool = False) -> torch.Tensor:
        """
        Generates a given number of samples with the generator. Removes gradients if required.
        """
        with (torch.no_grad() if no_grad else nullcontext()):
            z = self.sample_latent(nb_gen, device)
            return self.generator(z)

    @abstractmethod
    def forward_discriminator(self, fake: torch.Tensor, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes fake and real discriminator losses. Abstract method that should be implemented for each specific GAN
        model.
        """
        pass

    def forward_generator(self, real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates fake data and returns them with their generator loss.
        """
        device = real.device
        fake = self.generate(len(real), device)
        return self.generator_loss(fake), fake

    @abstractmethod
    def generator_loss(self, fake: torch.Tensor) -> torch.Tensor:
        """
        Computes the pointwise generator loss on the given data.
        """
        pass

    def forward(self, x: torch.Tensor, op: GAN.Op) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor,
                                                                                                torch.Tensor,
                                                                                                torch.Tensor]:
        """
        Unique forward function that either computes the discriminator loss or the generator loss depending on its
        second input.
        """
        if op is GAN.Op.DISCRIMINATOR:
            device = x.device
            batch_size = len(x)
            fake = self.generate(batch_size, device, no_grad=True)
            # Fake/real discriminator loss + gradient penalty
            if self.grad_penalty > 0:
                grad_d = grad_penalty(x, fake, self.discriminator, self.grad_penalty_center, self.grad_penalty_norm,
                                      self.grad_penalty_type)
            else:
                grad_d = torch.zeros(len(x), device=x.device)
            return *self.forward_discriminator(fake, x), grad_d
        elif op is GAN.Op.GENERATOR:
            return self.forward_generator(x)
        else:
            raise ValueError(f'Operation `{op}` not yet defined')

    @classmethod
    def training_step(cls, step: int, batch: Batch, forward_fn: ForwardFn, optimizers: Optimizers, scalers: Scalers,
                      schedulers: Schedulers, device: torch.device, shared_rng: random.Random,
                      opt: ModelDict) -> tuple[float, Log, list[str]]:
        """
        A training step consists of a discriminator update, followed by a generator update if a sufficient number of
        discriminator steps were performed since the last generator step.
        """
        config = GANDict(opt.model_params)
        x = batch.x.to(device)
        log: Log = {}

        discr_loss = cls.discriminator_step(step, x, forward_fn, optimizers, scalers, schedulers, opt, config, log)
        cls.generator_step(step, x, forward_fn, optimizers, scalers, schedulers, opt, config, log)

        return float(discr_loss.item()), log, ['gen_loss', 'discr_loss', 'grad_norms_mean']

    @classmethod
    def discriminator_step(cls, step: int, x: torch.Tensor, forward_fn: ForwardFn, optimizers: Optimizers,
                           scalers: Scalers, schedulers: Schedulers, opt: ModelDict, config: GANDict,
                           log: Log) -> torch.Tensor:
        discr_optimizer, discr_scaler, discr_scheduler = cls.extract_optimizer(GAN.discr_optimizer_name, optimizers,
                                                                               scalers, schedulers)
        discr_optimizer.zero_grad()
        discr_loss_fake, discr_loss_real, grad_loss = forward_fn(x, GAN.Op.DISCRIMINATOR)
        # Assemble the discriminator losses
        discr_loss = discr_loss_fake.mean() - discr_loss_real.mean() + config.grad_penalty * grad_loss.mean()
        optimizer_update(discr_loss, discr_optimizer, discr_scaler, discr_scheduler, opt)
        log['discr_loss'] = discr_loss.item()
        return discr_loss

    @classmethod
    def generator_step(cls, step: int, x: torch.Tensor, forward_fn: ForwardFn, optimizers: Optimizers,
                       scalers: Scalers, schedulers: Schedulers, opt: ModelDict, config: GANDict, log: Log):
        """
        Performs a generator update only if a sufficient number of discriminator updates were performed since the last
        one.
        """
        is_generator_step = step % config.nb_discr_steps == 0
        if is_generator_step:
            gen_optimizer, gen_scaler, gen_scheduler = cls.extract_optimizer(GAN.gen_optimizer_name, optimizers,
                                                                             scalers, schedulers)
            gen_optimizer.zero_grad()
            gen_loss, _ = forward_fn(x, GAN.Op.GENERATOR)
            gen_loss = config.gen_loss_scale * gen_loss.mean()
            grad_norms_mean = optimizer_update(gen_loss, gen_optimizer, gen_scaler, gen_scheduler, opt)
            log['gen_loss'] = float(gen_loss.item())
            if grad_norms_mean is not None:
                log['grad_norms_mean'] = grad_norms_mean

    def evaluation_step(self, batch: Batch, device: torch.device, opt: ModelDict,
                        config: TestDict) -> EvaluationTuple:
        """
        Generates fake data and computes their corresponding loss and spatial gradients.
        """
        real = batch.x.to(device)
        gen_loss, fake = self.forward_generator(real)
        if len(real.size()) == 2:
            with torch.inference_mode(mode=False), torch.no_grad():
                grads = - ft.grad(lambda x: self.generator_loss(x).sum())(fake)
        else:
            grads = torch.zeros_like(real)
        if Test.INTER_QUALI in config.tests:
            # Interpolations in the prior
            gen_params = GANInferenceDict(config.test_params)
            z1 = self.sample_latent(len(real), device)
            z2 = self.sample_latent(len(real), device)
            interpolated_z = interpolate(z1, z2, gen_params.resolution, 1, gen_params.interpolation)
            interpolated_z = interpolated_z.flatten(end_dim=1)
            interpolations = self.generator(interpolated_z).view((len(real), gen_params.resolution) + real.size()[1:])
        else:
            interpolations = torch.zeros((len(real)))
        return real, fake, gen_loss, grads, interpolations

    def evaluation_logs(self, eval_results: EvaluationTuple, device: torch.device, opt: ModelDict,
                        config: TestDict) -> tuple[float, Log]:
        """
        If specified in the test configuration, computes the FID, creates sample images and/or saves the generated
        samples.
        """
        real, fake, gen_loss, grad, interpolations = eval_results
        gen_loss = gen_loss.mean().item()

        fid, gen_grid, inter_grid = None, None, None
        for test in config.tests:
            if test is Test.FID:
                assert isinstance(self.dataset, BaseImageDataset)
                fid = compute_fid(real, fake, config.batch_size, device)
            elif test is Test.GEN_QUALI:
                gen_params = GANInferenceDict(config.test_params)
                grad_mean = torch.linalg.norm(grad.flatten(start_dim=1), dim=1).mean().item()
                grad = grad / (grad_mean + 1e-6)
                norms = fill_as(torch.linalg.norm(grad.flatten(start_dim=1), dim=1), grad)
                grad = torch.where(norms < 1, grad / (norms + 1e-6), grad)
                gen_grid = gen_quali(real, fake, config.nb_quali, gradients=grad, scale_grad=3,
                                     loss=lambda grid: self.generator_loss(grid.to(device)).cpu(),
                                     resolution=gen_params.resolution)
            elif test is Test.INTER_QUALI:
                inter_grid = inter_quali(interpolations, config.nb_quali)
            elif test is not Test.GEN_SAVE:
                raise ValueError(f'Test `{test}` not implemented yet.')

        score = None
        logs: Log = {'gen_loss': gen_loss}
        if fid is not None:
            logs['fid'] = fid
            if Test.FID is config.metric:
                score = -fid
        if gen_grid is not None:
            logs['gen_quali'] = gen_grid
        if inter_grid is not None:
            logs['inter_quali'] = inter_grid
        if Test.GEN_SAVE in config.tests:
            logs['gen'] = fake
            if inter_grid is not None:
                logs['inter'] = interpolations

        return score if score is not None else -gen_loss, logs
