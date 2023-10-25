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

from collections import OrderedDict
from contextlib import nullcontext
from typing import Iterator
from gpm.data.image.base import BaseImageDataset

from gpm.data.lowd.base import BaseLowDDataset
from gpm.eval.interpolate import Interpolation, interpolate
from gpm.eval.metrics import compute_fid
from gpm.eval.quali import gen_quali, inter_quali
from gpm.eval.tests import Test
from gpm.models.base import BaseModel
from gpm.networks.factory import decoder_factory, edm_denoiser_factory
from gpm.networks.init import InitDict, init_network
from gpm.networks.utils import DecoderDict, EDMDenoiserDict
from gpm.utils.config import DotDict, ModelDict, TestDict
from gpm.utils.helper import fill_as
from gpm.utils.optimizer import optimizer_update
from gpm.utils.types import Batch, ForwardFn, Log, Optimizers, Scalers, Schedulers


EvaluationTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        torch.Tensor]


class ScoreGANDict(DotDict):
    """
    Parameter dictionary for Score GANs.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'sigma_min', 'sigma_max', 'rho', 'score_data', 'path_score_data', 'latent_dim', 'generator',
                'init_gen', 'score_gen', 'init_score', 'nb_score_steps'}.issubset(self.keys())
        # Architecture of the score network for the data distribution
        self.score_data: EDMDenoiserDict = EDMDenoiserDict(self.score_data)
        # Path to the saved score network for the data distribution, must comply with the provided parameters
        self.path_score_data: str
        # Architecture of the score network for the generated distribution
        self.score_gen: EDMDenoiserDict = EDMDenoiserDict(self.score_gen)
        self.init_score: InitDict = InitDict(self.init_score)  # Initialization of the score network
        self.latent_dim: int  # Dimensionality of the generator's latent space
        self.generator: DecoderDict = DecoderDict(self.generator)  # Generator architecture
        self.init_gen: InitDict = InitDict(self.init_gen)  # Generator initialization parameters
        # Number of score optimization steps in-between generator updates
        self.nb_score_steps: int
        # Noise sampling parameters, cf. the paper. Identical to the EDM inference noise scheduling.
        self.sigma_min: float
        self.sigma_max: float
        self.rho: float

        # Multiplier of the generator loss (default, 1)
        self.gen_loss_scale: float
        if 'gen_loss_scale' not in self:
            self.gen_loss_scale = 1

        # Multipliers of the data and generated score (default, 1)
        self.weight_data: float
        self.weight_gen: float
        if 'weight_data' not in self:
            self.weight_data = 1
        if 'weight_gen' not in self:
            self.weight_gen = 1


class ScoreGANInferenceDict(DotDict):
    """
    Inference parameter dictionary for Score GANs.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.sigma: float  # Standard deviation of noise to add to the data for vizualization purposes
        self.resolution: int  # Number of interpolations
        self.interpolation: Interpolation  # Interpolation method to use for qualitative interpolation tests
        if 'sigma' not in self:
            self.sigma = 0.5
        if 'resolution' not in self:
            self.resolution = 20
        if 'interpolation' in self:
            self.interpolation = Interpolation(self.interpolation)
        else:
            self.interpolation = Interpolation.LINEAR


class ScoreGAN(BaseModel):
    """
    Score GAN model.
    """

    gen_optimizer_name = 'gen_optimizer'
    score_gen_optimizer_name = 'score_optimizer'

    mse_loss = torch.nn.MSELoss(reduction='none')

    @enum.unique
    class Op(enum.Enum):
        """
        Operation indicator for the unique forward method.
        """
        SCORE = enum.auto()  # Score optim.
        GENERATOR = enum.auto()  # Generator optim.

    def __init__(self, dataset: BaseLowDDataset | BaseImageDataset, config: ScoreGANDict):
        super().__init__()
        self.sigma_min = 0.
        self.sigma_max = 0.
        self.rho = 0.
        self.p_mean = 0.
        self.p_std = 0.
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.rho = config.rho

        self.latent_dim = config.latent_dim
        self.gen_loss_scale = config.gen_loss_scale
        self.weight_data = config.weight_data
        self.weight_gen = config.weight_gen

        self.dataset = dataset
        if isinstance(dataset, BaseImageDataset):
            in_channels, in_size = dataset.channels, dataset.image_size
            data_size = in_channels, in_size, in_size
        else:
            data_size = dataset.dim,

        # Load the pretrained score of the data distribution
        self.denoiser_data = edm_denoiser_factory(dataset, data_size, config.score_data)
        checkpoint: OrderedDict = torch.load(config.path_score_data, map_location='cpu')
        updated_checkpoint = OrderedDict()
        for key, value in checkpoint.items():
            updated_checkpoint[key.replace('denoiser.', '')] = value
        self.denoiser_data.load_state_dict(updated_checkpoint)

        # Build generator
        self.generator = decoder_factory(dataset, config.latent_dim, data_size, config.generator)
        init_network(self.generator, config.init_gen)

        # Build the score network for the generated distribution
        self.denoiser_gen = edm_denoiser_factory(dataset, data_size, config.score_gen)
        init_network(self.denoiser_gen, config.init_score)

    def optimizers_names(self, opt: ModelDict) -> set[str]:
        optimizers = {ScoreGAN.gen_optimizer_name}
        optimizers.add(ScoreGAN.score_gen_optimizer_name)
        return optimizers

    def optimized_params(self, optimizer_name: str, opt: ModelDict) -> Iterator[torch.nn.parameter.Parameter]:
        if optimizer_name == ScoreGAN.gen_optimizer_name:
            return self.generator.parameters()
        elif optimizer_name == ScoreGAN.score_gen_optimizer_name:
            return self.denoiser_gen.parameters()
        else:
            raise ValueError(f'No optimizer named `{optimizer_name}`')

    def sample_noise_level(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Chooses a given number of noise levels at random in the EDM inference noise scheduling.
        """
        t = torch.rand(n, device=device)
        return (self.sigma_max ** (1 / self.rho)
                + t * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho

    def sample_latent(self, nb_gen: int, device: torch.device) -> torch.Tensor:
        """
        Samples a batch of latent states.
        """
        return torch.normal(0, 1, (nb_gen, self.latent_dim), device=device)

    def network_generate(self, nb_gen: int, device: torch.device, no_grad: bool = False) -> torch.Tensor:
        """
        Generates a given number of samples with the generator. Removes gradients if required.
        """
        with (torch.no_grad() if no_grad else nullcontext()):
            z = self.sample_latent(nb_gen, device)
            return self.generator(z)

    @classmethod
    def training_step(cls, step: int, batch: Batch, forward_fn: ForwardFn, optimizers: Optimizers, scalers: Scalers,
                      schedulers: Schedulers, device: torch.device, shared_rng: random.Random,
                      opt: ModelDict) -> tuple[float, Log, list[str]]:
        """
        A training step consists of a score update, followed by a generator update if a sufficient number of score
        steps were performed since the last generator step.
        """
        x = batch.x.to(device)
        log: Log = {}
        config = ScoreGANDict(opt.model_params)
        score_gen_loss = cls.score_gen_step(step, x, forward_fn, optimizers, scalers, schedulers, opt, config, log)
        cls.generator_step(step, x, forward_fn, optimizers, scalers, schedulers, opt, config, log)
        return float(score_gen_loss), log, ['score_gen_loss']

    @classmethod
    def score_gen_step(cls, step: int, x: torch.Tensor, forward_fn: ForwardFn, optimizers: Optimizers,
                       scalers: Scalers, schedulers: Schedulers, opt: ModelDict, config: ScoreGANDict,
                       log: Log) -> float:
        optimizer, scaler, scheduler = cls.extract_optimizer(ScoreGAN.score_gen_optimizer_name, optimizers,
                                                             scalers, schedulers)
        optimizer.zero_grad()
        loss, = forward_fn(x, (ScoreGAN.Op.SCORE, None))
        loss = torch.mean(loss)
        optimizer_update(loss, optimizer, scaler, scheduler, opt)
        loss = loss.item()
        log['score_gen_loss'] = loss
        return loss

    @classmethod
    def generator_step(cls, step: int, x: torch.Tensor, forward_fn: ForwardFn, optimizers: Optimizers,
                       scalers: Scalers, schedulers: Schedulers, opt: ModelDict, config: ScoreGANDict,
                       log: Log):
        """
        Performs a generator update only if a sufficient number of score updates were performed since the last
        one.
        """
        model_params = ScoreGANDict(opt.model_params)
        is_generator_step = (step - model_params.nb_score_steps) % model_params.nb_score_steps == 0
        if is_generator_step:
            gen_optimizer, gen_scaler, gen_scheduler = cls.extract_optimizer(ScoreGAN.gen_optimizer_name,
                                                                             optimizers, scalers, schedulers)
            gen_optimizer.zero_grad()
            gen_loss, = forward_fn(x, (ScoreGAN.Op.GENERATOR, None))
            gen_loss = model_params.gen_loss_scale * gen_loss.mean()
            optimizer_update(gen_loss, gen_optimizer, gen_scaler, gen_scheduler, opt)
            log['gen_loss'] = float(gen_loss.item())

    def forward_score_noise(self, real: torch.Tensor, sigma: torch.Tensor | None) -> torch.Tensor:
        """
        Denoising score matching loss.
        """
        x = self.network_generate(real.size(0), real.device, no_grad=True)
        if sigma is None:
            sigma = self.sample_noise_level(len(real), real.device)
        noise = torch.normal(0, 1, x.size(), device=x.device)
        perturbed_x = x + noise * fill_as(sigma, x)
        x_hat = self.denoiser_gen(perturbed_x, sigma)
        mse = ScoreGAN.mse_loss(x_hat, x)
        loss = torch.sum(mse, dim=list(range(1, len(x.size()))))
        return loss

    def forward_generator(self, real: torch.Tensor, sigma: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor,
                                                                                         torch.Tensor, torch.Tensor,
                                                                                         torch.Tensor]:
        """
        Trains the generator with the gradients provided by the score functions
        """
        x = self.network_generate(real.size(0), real.device)
        # Perturb the fake data
        if sigma is None:
            sigma = self.sample_noise_level(len(real), real.device)
        noise = torch.normal(0, 1, x.size(), device=x.device)
        std = fill_as(sigma, x)
        perturbed_x = x + noise * std
        # Compute the corresponding scores
        sigma_score_data = self.weight_data * (self.denoiser_data(perturbed_x, sigma) - perturbed_x) / std
        sigma_score_gen = (self.denoiser_gen(perturbed_x, sigma) - perturbed_x) / std
        diffusion = self.weight_gen * sigma_score_gen
        gradient = diffusion - sigma_score_data
        # Compute the adapted loss so that the generator follows the right gradients
        loss = torch.sum(self.gen_loss_scale * perturbed_x * gradient.detach(),
                         dim=list(range(1, len(perturbed_x.size())))).mean()
        return loss, x, perturbed_x, sigma_score_data, -diffusion

    def forward(self, x: torch.Tensor, op_sigma: tuple[ScoreGAN.Op, torch.Tensor | None]) -> tuple[torch.Tensor]:
        """
        Unique forward function that either computes the score loss or the generator loss depending on its second
        input.
        """
        op, sigma = op_sigma
        if op is ScoreGAN.Op.SCORE:
            return self.forward_score_noise(x, sigma),
        elif op is ScoreGAN.Op.GENERATOR:
            return self.forward_generator(x, sigma)[0],
        else:
            raise ValueError(f'Operation `{op}` not yet defined')

    def evaluation_step(self, batch: Batch, device: torch.device, opt: ModelDict, config: TestDict) -> EvaluationTuple:
        """
        Generates fake data and computes their corresponding gradients for a given noise level.
        """
        x = batch.x.to(device)
        gen_params = ScoreGANInferenceDict(config.test_params)
        sigma = gen_params.sigma
        noise = torch.normal(0, 1, x.size(), device=x.device)
        x_noise = x + noise * sigma
        _, x_gen, x_gen_noise, score_data, diffusion = self.forward_generator(x, torch.tensor([sigma] * len(x),
                                                                                              device=x.device))
        if Test.INTER_QUALI in config.tests:
            # Interpolations in the prior
            z1 = self.sample_latent(len(x), device)
            z2 = self.sample_latent(len(x), device)
            interpolated_z = interpolate(z1, z2, gen_params.resolution, 1, gen_params.interpolation)
            interpolated_z = interpolated_z.flatten(end_dim=1)
            interpolations = self.generator(interpolated_z).view((len(x), gen_params.resolution) + x.size()[1:])
        else:
            interpolations = torch.zeros((len(x)))
        return x, x_noise, x_gen, x_gen_noise, score_data, diffusion, interpolations

    def evaluation_logs(self, eval_results: EvaluationTuple, device: torch.device, opt: ModelDict,
                        config: TestDict) -> tuple[float, Log]:
        """
        If specified in the test configuration, computes the FID, creates sample images and/or saves the generated
        samples.
        """
        x, x_noise, x_gen, x_gen_noise, gradient1, gradient2, interpolations = eval_results

        fid, gen_grid, gen_grid_noise, inter_grid = None, None, None, None
        for test in config.tests:
            if test is Test.GEN_QUALI:
                grad_norm = torch.linalg.norm((gradient1 + gradient2).flatten(start_dim=1), dim=1).mean().item() / 2
                gradient1 /= (grad_norm + 1e-6)
                gradient2 /= (grad_norm + 1e-6)
                gen_grid_noise = gen_quali(x, x_gen_noise, config.nb_quali, gradients=[gradient1, gradient2],
                                           scale_grad=5)
                gen_grid = gen_quali(x, x_gen, config.nb_quali)
            elif test is Test.INTER_QUALI:
                inter_grid = inter_quali(interpolations, config.nb_quali)
            elif test is Test.FID:
                assert isinstance(self.dataset, BaseImageDataset)
                fid = compute_fid(x, x_gen, config.batch_size, device)
            elif test is not Test.GEN_SAVE:
                raise ValueError(f'Test `{test}` not implemented yet.')

        score = None
        logs: Log = {'loss': 0}
        if fid is not None:
            logs['fid'] = fid
            if Test.FID is config.metric:
                score = -fid
        if gen_grid is not None and gen_grid_noise is not None:
            logs['gen_quali'] = gen_grid
            logs['gen_noise'] = gen_grid_noise
        if inter_grid is not None:
            logs['inter_quali'] = inter_grid
        if Test.GEN_SAVE in config.tests:
            logs['gen'] = x_gen
            if inter_grid is not None:
                logs['inter'] = interpolations

        return score if score is not None else 0, logs
