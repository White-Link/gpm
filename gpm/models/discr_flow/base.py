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


import math
import random
import torch

import functorch as ft
import numpy as np

from abc import abstractmethod
from typing import Any, Callable, Iterator, Sequence

from gpm.data.image.base import BaseImageDataset
from gpm.data.lowd.base import BaseLowDDataset
from gpm.eval.interpolate import Interpolation, interpolate
from gpm.eval.metrics import compute_fid
from gpm.eval.quali import gen_quali, inter_quali
from gpm.eval.tests import Test
from gpm.models.base import BaseModel
from gpm.models.gan.base import GANModel
from gpm.models.gan.losses import GradPenalty, grad_penalty
from gpm.networks.activations import Activation
from gpm.networks.factory import encoder_factory
from gpm.networks.init import InitDict, init_network
from gpm.networks.score.scalar_emb import NetworkScalarEmbedding, NoiseEncoding
from gpm.networks.utils import EncoderDict, SequentialTuple
from gpm.utils.config import DotDict, ModelDict, TestDict
from gpm.utils.helper import fill_as
from gpm.utils.optimizer import optimizer_update
from gpm.utils.types import Batch, ForwardFn, Log, Optimizers, Scalers, Schedulers


EvaluationTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class DiscrFlowDict(DotDict):
    """
    Parameter dictionary for Discriminator Flows.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'model', 'eta', 'nb_steps', 'discriminator', 'init', 'embedding',
                'embedding_size'}.issubset(self.keys())
        assert self.model in [model.value for model in GANModel]
        self.model: GANModel = GANModel(self.model)  # GAN loss: specificies discr. and gen. loss, cf. GAN model
        self.eta: float  # Gradient multiplier
        self.nb_steps: int  # Number of discretization steps during training
        self.discriminator: EncoderDict = EncoderDict(self.discriminator)  # Discriminator architecture
        self.init: InitDict = InitDict(self.init)  # Initialization of the discriminator
        self.embedding: NoiseEncoding = NoiseEncoding(self.embedding)  # Time embedding type
        self.embedding_size: int  # Size of time embeddings

        self.grad_penalty: float | list[float]  # Strength of gradient penalty (default, 0)
        self.grad_penalty_center: float  # Center of gradient penalty (default, 0)
        self.grad_penalty_norm: float  # Norm of gradient penalty (default, 2)
        # Choice of gradient penalty application (on fake, real, or interpolated data, default to the latter)
        self.grad_penalty_type: GradPenalty
        self.entropy_reg: float  # Strength of added noise at each generation step (default, 0)
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
        if 'entropy_reg' not in self:
            self.entropy_reg = 0

        # Mean and standard deviation of the initial distribution \pi (default, 0 and 1)
        self.p_mean: float
        self.p_std: float
        if 'p_mean' not in self:
            self.p_mean = 0
        if 'p_std' not in self:
            self.p_std = 1

        self.ignore_t: bool  # Whether discriminator should ignore its time input (default, False)
        if 'ignore_t' not in self:
            self.ignore_t = False
        self.time_scale_factor: float  # Multiplication factor of time at discriminator input (default, 1)
        if 'time_scale_factor' not in self:
            self.time_scale_factor = 1

        self.eval_for_gen: bool  # Sets the discriminator in eval mode for generation during training (default, False)
        if 'eval_for_gen' not in self:
            self.eval_for_gen = False


class DiscrFlowInferenceDict(DotDict):
    """
    Inference parameter dictionary for Discriminator Flows.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert 'nb_steps' in self
        self.nb_steps: int  # Number of gradient steps
        self.total_nb_steps: int  # Number of discretization steps (default, nb_steps)
        if 'total_nb_steps' not in self:
            self.total_nb_steps = self.nb_steps
        # Resolution of pointwise generator loss for qualitative tests on 2D data, or number of interpolations
        self.resolution: int
        if 'resolution' not in self:
            self.resolution = 200
        self.interpolation: Interpolation  # Interpolation method to use for qualitative interpolation tests
        if 'interpolation' in self:
            self.interpolation = Interpolation(self.interpolation)
        else:
            self.interpolation = Interpolation.LINEAR


class DiscrFlow(BaseModel):
    """
    Discriminator Flows model.
    """

    optimizer_name = 'optimizer'

    def __init__(self, dataset: BaseLowDDataset | BaseImageDataset, config: DiscrFlowDict):
        super().__init__()
        self.dataset = dataset
        self.eta = config.eta
        self.nb_steps = config.nb_steps
        self.grad_penalty = config.grad_penalty
        self.grad_penalty_center = config.grad_penalty_center
        self.grad_penalty_norm = config.grad_penalty_norm
        self.grad_penalty_type = config.grad_penalty_type
        self.entropy_reg = config.entropy_reg
        self.p_mean = config.p_mean
        self.p_std = config.p_std
        self.ignore_t = config.ignore_t
        self.time_scale_factor = config.time_scale_factor
        self.eval_for_gen = config.eval_for_gen

        # Check that the discriminator does not end with an activation
        assert (
            config.discriminator.final_activation is None
            or config.discriminator.final_activation.name is Activation.IDENTITY
        )

        # Build the discriminator
        if isinstance(dataset, BaseImageDataset):
            in_channels, in_size = dataset.channels, dataset.image_size
            data_size = in_channels, in_size, in_size
        else:
            data_size = dataset.dim,
        self.discriminator = self.build_discriminator(data_size, config)
        init_network(self.discriminator, config.init)

    def build_discriminator(self, data_size: tuple[int] | tuple[int, int, int],
                            config: DiscrFlowDict) -> torch.nn.Module:
        """
        Builds the discriminator.
        """
        network = SequentialTuple(encoder_factory(self.dataset, data_size, 1, config.discriminator,
                                                  additional_input=(1 - self.ignore_t) * config.embedding_size),
                                  torch.nn.Flatten(start_dim=0))
        # Add temporal embeddings if required
        if not self.ignore_t:
            network = NetworkScalarEmbedding(network, config.embedding, config.embedding_size)
        return network

    def optimizers_names(self, opt: ModelDict) -> set[str]:
        return {DiscrFlow.optimizer_name}

    def optimized_params(self, optimizer_name: str, opt: ModelDict) -> Iterator[torch.nn.parameter.Parameter]:
        if optimizer_name == DiscrFlow.optimizer_name:
            return self.discriminator.parameters()
        else:
            raise ValueError(f'No optimizer named `{optimizer_name}`')

    def t_transform(self, t: torch.Tensor) -> torch.Tensor | None:
        """
        Ignores time if required.
        """
        if not self.ignore_t:
            return t
        else:
            return None

    def discriminator_t(self, t: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Helper function for loss computations: partial discriminator application with the given input time.
        """
        return lambda x: self.discriminator(x, self.t_transform(t))

    @abstractmethod
    def discr_loss(self, fake: torch.Tensor, real: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes fake and real discriminator losses. Abstract method that should be implemented for each specific GAN
        model.
        """
        pass

    @abstractmethod
    def gen_loss(self, fake: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the pointwise generator loss on the given data and for the given time.
        """
        pass

    def discr_vector_field(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Gradients to apply to input particles in the generation process. Corresponds to the batchwise gradients of the
        pointwise generator loss applied on the input.
        """
        return -ft.grad(lambda x0: self.gen_loss(x0, t).sum())(x)

    def sample_latent_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Samples an initial particle of the same shape as the input.
        """
        return self.p_mean + self.p_std * torch.randn_like(x)

    @torch.inference_mode(mode=False)
    @torch.no_grad()
    def partial_generate_like(self, x: torch.Tensor, nb_steps: int, total_nb_steps: int | None = None,
                              z_0: torch.Tensor | None = None,
                              return_intermediate: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generation function. Applies the given number of discretization steps, provided the overall discretization
        level. Uses the input initial particles if provided, otherwise samples them.

        Returns the resulting synthesized data, the current gradients and times. Also returns the intermediate results
        if requested.
        """
        if total_nb_steps is None:
            total_nb_steps = self.nb_steps
        # Discretization times
        z = []
        grad = []
        z_t = self.sample_latent_like(x) if z_0 is None else z_0  # Initial samples from the prior \pi
        times = (torch.arange(nb_steps, device=x.device) / total_nb_steps).unsqueeze(-1).expand((-1, len(z_t)))
        grad_t = None
        # Generation loop
        for step, t in enumerate(times):
            grad_t = self.discr_vector_field(z_t, t)
            if return_intermediate:
                z.append(z_t)
                grad.append(grad_t)
            # Adds noise if requested
            if step < total_nb_steps - 1:
                diffusion = math.sqrt(self.entropy_reg / total_nb_steps) * torch.randn_like(z_t)
            else:
                diffusion = torch.zeros_like(z_t)
            # Applied the gradient
            z_t = z_t + self.eta * grad_t / total_nb_steps + diffusion
        current_t = (times[-1] + 1 / total_nb_steps) if len(times) > 0 else torch.zeros(len(z_t), device=z_t.device)
        if return_intermediate:
            z.append(z_t)
            return torch.stack(z, dim=1), torch.stack(grad, dim=1), torch.cat([times,
                                                                               current_t.unsqueeze(0)]).transpose(0, 1)
        else:
            return z_t, torch.zeros_like(z_t) if grad_t is None else grad_t, current_t

    def forward(self, x: torch.Tensor, op: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a batch of model samples and selects random generation times to train the time-dependent
        discriminator. Returns the fake/real losses of the latter as well as the gradient penalty.
        """
        is_train = self.training
        if self.eval_for_gen:
            self.discriminator.eval()
        # Generate all intermediate steps
        z, _, t = self.partial_generate_like(x, self.nb_steps - 1, return_intermediate=True)
        # Randomly select steps
        steps = torch.randint(self.nb_steps, (len(x),), device=x.device)
        batch_ids = torch.arange(len(x), device=x.device)
        z = z[batch_ids, steps]
        t = t[batch_ids, steps]
        # Compute the gradient penalty at the selected times
        if isinstance(self.grad_penalty, Sequence) or self.grad_penalty > 0:
            if isinstance(self.grad_penalty, Sequence):
                partition = (len(self.grad_penalty) * steps.numpy(force=True) / self.nb_steps).astype(int)
                grad_penalty_coef = np.array(self.grad_penalty)[partition]
                grad_penalty_coef = torch.tensor(grad_penalty_coef, device=x.device)
            else:
                grad_penalty_coef = self.grad_penalty
            grad_d = grad_penalty_coef * grad_penalty(x, z, self.discriminator_t(t), self.grad_penalty_center,
                                                      self.grad_penalty_norm, self.grad_penalty_type)
        else:
            grad_d = torch.zeros(len(x), device=x.device)
        if self.eval_for_gen:
            self.discriminator.train(mode=is_train)
        # Compute the discriminator loss at the selected times
        return *self.discr_loss(z, x, t), grad_d

    @classmethod
    def training_step(cls, step: int, batch: Batch, forward_fn: ForwardFn, optimizers: Optimizers, scalers: Scalers,
                      schedulers: Schedulers, device: torch.device, shared_rng: random.Random,
                      opt: ModelDict) -> tuple[float, Log, list[str]]:
        optimizer, scaler, scheduler = cls.extract_optimizer(DiscrFlow.optimizer_name, optimizers, scalers,
                                                             schedulers)
        optimizer.zero_grad()
        x = batch.x.to(device)
        discr_loss_fake, discr_loss_real, grad_loss = forward_fn(x, None)
        # Assemble the discriminator losses
        discr_loss = discr_loss_fake.mean() - discr_loss_real.mean() + grad_loss.mean()
        optimizer_update(discr_loss, optimizer, scaler, scheduler, opt)
        loss = discr_loss.item()
        return loss, {'discr_loss': loss}, ['discr_loss']

    def evaluation_step(self, batch: Batch, device: torch.device, opt: ModelDict, config: TestDict) -> EvaluationTuple:
        """
        Generates samples following the inference configuration.
        """
        x = batch.x.to(device)
        gen_params = DiscrFlowInferenceDict(config.test_params)
        x_gen, grad, t = self.partial_generate_like(x, gen_params.nb_steps, total_nb_steps=gen_params.total_nb_steps,
                                                    return_intermediate=Test.GEN_QUALI in config.tests)
        if Test.INTER_QUALI in config.tests:
            # Interpolations in the prior
            z1 = self.sample_latent_like(x)
            z2 = self.sample_latent_like(x)
            interpolated_z = interpolate((z1 - self.p_mean) / self.p_std, (z2 - self.p_mean) / self.p_std,
                                         gen_params.resolution, 1, gen_params.interpolation) * self.p_std + self.p_mean
            interpolated_z = interpolated_z.flatten(end_dim=1)
            interpolations = self.partial_generate_like(x, gen_params.nb_steps,
                                                        total_nb_steps=gen_params.total_nb_steps,
                                                        z_0=interpolated_z, return_intermediate=True)[0]
            interpolations = interpolations.view((len(x), gen_params.resolution) + interpolations.size()[1:])
        else:
            interpolations = torch.zeros((len(x)))
        return x, x_gen, grad, t, interpolations

    def evaluation_logs(self, eval_results: EvaluationTuple, device: torch.device, opt: ModelDict,
                        config: TestDict) -> tuple[float, Log]:
        """
        If specified in the test configuration, computes the FID, creates sample images and/or saves the generated
        samples.
        """
        x, x_gen, grad, t, interpolations = eval_results

        fid, gen_grid, grad_mean, inter_grid = None, None, None, None
        for test in config.tests:
            if test is Test.GEN_QUALI:
                gen_params = DiscrFlowInferenceDict(config.test_params)
                grad_mean = torch.linalg.norm(grad.flatten(start_dim=2), dim=2).mean().item()
                grad = grad / (grad_mean + 1e-6)
                norms = fill_as(torch.linalg.norm(grad.flatten(start_dim=2), dim=2), grad)
                grad = torch.where(norms < 1, grad / (norms + 1e-6), grad)
                gen_grid = [gen_quali(x, x_gen[:, i], config.nb_quali,
                                      gradients=grad[:, i] if i < grad.size(1) else None, scale_grad=5,
                                      loss=lambda grid: self.gen_loss(grid.to(device),
                                                                      t[:1, i].expand(len(grid)).to(device)).cpu(),
                                      resolution=gen_params.resolution) for i in range(x_gen.size(1))]
            elif test is Test.FID:
                assert isinstance(self.dataset, BaseImageDataset)
                fid = compute_fid(x, x_gen if Test.GEN_QUALI not in config.tests else x_gen[:, -1], config.batch_size,
                                  device)
            elif test is Test.INTER_QUALI:
                inter_grid = [inter_quali(interpolations[:, :, i],
                                          config.nb_quali) for i in range(interpolations.size(2))]
            elif test is not Test.GEN_SAVE:
                raise ValueError(f'Test `{test}` not implemented yet.')

        score = None
        logs: Log = {}
        if fid is not None:
            logs['fid'] = fid
            if Test.FID is config.metric:
                score = -fid
        if gen_grid is not None:
            logs['gen_quali'] = gen_grid
        if grad_mean is not None:
            logs['grad_mean'] = grad_mean
        if inter_grid is not None:
            logs['inter_quali'] = inter_grid
        if Test.GEN_SAVE in config.tests:
            logs['gen'] = x_gen
            if inter_grid is not None:
                logs['inter'] = interpolations

        return score if score is not None else 0., logs
