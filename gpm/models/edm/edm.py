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
import math
import random
import torch

from torch import distributions as D
from typing import Any, Iterator

from gpm.data.image.base import BaseImageDataset
from gpm.data.lowd.base import BaseLowDDataset
from gpm.eval.interpolate import Interpolation, interpolate
from gpm.eval.metrics import compute_fid
from gpm.eval.quali import gen_quali, inter_quali
from gpm.eval.tests import Test
from gpm.models.base import BaseModel
from gpm.networks.factory import edm_denoiser_factory
from gpm.networks.init import InitDict, init_network
from gpm.networks.utils import EDMDenoiserDict
from gpm.utils.config import DotDict, ModelDict, TestDict
from gpm.utils.helper import fill_as
from gpm.utils.optimizer import optimizer_update
from gpm.utils.types import Batch, ForwardFn, Log, Optimizers, Scalers, Schedulers


EvaluationTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
SolverTuple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class EDMDict(DotDict):
    """
    Parameter dictionary for EDM.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'score_fn', 'init'}.issubset(self.keys())
        self.score_fn: EDMDenoiserDict = EDMDenoiserDict(self.score_fn)  # Architecture of the score network
        self.init: InitDict = InitDict(self.init)  # Initialization of the score network
        # LogNormal sampling parameters of the noise level during training (default to -1.2, 1.2)
        self.p_mean: float
        self.p_std: float
        if 'p_mean' not in self:
            self.p_mean = -1.2
        if 'p_std' not in self:
            self.p_std = 1.2


class EDMInferenceDict(DotDict):
    """
    Global inference parameter dictionary for EDM.
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'method', 'nb_steps', 'sigma_min', 'sigma_max', 'rho', 'sde'}.issubset(self.keys())
        self.method: EDM.Inference = EDM.Inference(self.method)  # Choice of sampler (Euler or Heun/EDM solver)
        self.nb_steps: int  # Number of steps in the solver
        self.sde: bool  # Whether to use stochastic or deterministic sampling
        # Noise scheduling parameters, cf. the EDM paper
        self.sigma_min: float
        self.sigma_max: float
        self.rho: float
        self.total_nb_steps: int  # Number of discretization steps (default, nb_steps, otherwise it will early stop)
        if 'total_nb_steps' not in self:
            self.total_nb_steps = self.nb_steps
        self.resolution: int  # Number of interpolations
        self.interpolation: Interpolation  # Interpolation method to use for qualitative interpolation tests
        if 'resolution' not in self:
            self.resolution = 10
        if 'interpolation' in self:
            self.interpolation = Interpolation(self.interpolation)
        else:
            self.interpolation = Interpolation.LINEAR


class EDMHeunInferenceDict(EDMInferenceDict):
    """
    Inference parameter dictionary for the second-order Heun sampler of EDM.

    Warning: defaults to deterministic sampling!
    """

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        # Noise application parameters, cf. the EDM paper
        self.S_churn: float
        self.S_min: float
        self.S_max: float
        self.S_noise: float
        if 'S_churn' not in self:
            self.S_churn = 0
        if 'S_min' not in self:
            self.S_min = 0
        if 'S_max' not in self:
            self.S_max = math.inf
        if 'S_noise' not in self:
            self.S_noise = 1


class EDM(BaseModel):
    """
    EDM model.
    """

    score_optimizer_name = 'score_optimizer'
    mse_loss = torch.nn.MSELoss(reduction='none')

    @enum.unique
    class Inference(str, enum.Enum):
        """
        Object representing the two possible samplers for EDM.
        """
        Euler = 'euler'
        EDM = 'edm'  # Heun

    def __init__(self, dataset: BaseLowDDataset | BaseImageDataset, config: EDMDict):
        super().__init__()
        self.dataset = dataset
        self.sigma_distr = D.LogNormal(config.p_mean, config.p_std)  # Noise level distribution during training
        if isinstance(dataset, BaseImageDataset):
            in_channels, in_size = dataset.channels, dataset.image_size
            data_size = in_channels, in_size, in_size
        else:
            data_size = dataset.dim,
        # Build the score network
        self.denoiser = edm_denoiser_factory(dataset, data_size, config.score_fn)
        init_network(self.denoiser, config.init)

    def optimizers_names(self, opt: ModelDict) -> set[str]:
        return {EDM.score_optimizer_name}

    def optimized_params(self, optimizer_name: str, opt: ModelDict) -> Iterator[torch.nn.parameter.Parameter]:
        if optimizer_name == EDM.score_optimizer_name:
            return self.denoiser.parameters()
        else:
            raise ValueError(f'No optimizer named `{optimizer_name}`')

    def sample_steps(self, nb_steps: int, sigma_min: float, sigma_max: float, rho: float,
                     device: torch.device) -> torch.Tensor:
        """
        Chooses discretization times for EDM inference, provided the number of steps and scheduling parameters.
        """
        step_indices = torch.arange(nb_steps, device=device)
        steps = (sigma_max ** (1 / rho) + step_indices / (nb_steps - 1)
                 * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        return torch.cat([steps, torch.zeros_like(steps[:1])])

    @classmethod
    def sample_latent_like(cls, x: torch.Tensor, sigma_max: float) -> torch.Tensor:
        """
        Samples an initial particle of the same shape as the input.
        """
        return torch.normal(0, sigma_max, x.size(), device=x.device)

    def sde_generate_like(self, x: torch.Tensor, nb_steps: int, sigma_min: float, sigma_max: float, rho: float,
                          sde: bool, total_nb_steps: int | None = None, z_0: torch.Tensor | None = None,
                          return_intermediate: bool = False) -> SolverTuple:
        """
        SDE sampler. Applies the given number of discretization steps, provided the overall discretization level. Uses
        the input initial particles if provided, otherwise samples them. Returns both the generated output and the
        gradients it receives.

        If required, returns all intermediary results.
        """
        if total_nb_steps is None:
            total_nb_steps = nb_steps
        sigma_min = max(sigma_min, self.denoiser.sigma_min)
        sigma_max = min(sigma_max, self.denoiser.sigma_max)
        z = []
        drift = []
        diffusion = []
        steps = self.sample_steps(total_nb_steps, sigma_min, sigma_max, rho, x.device)  # Selection of times
        z_t = self.sample_latent_like(x, sigma_max) if z_0 is None else z_0  # Initial sample from the prior
        for k, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            if return_intermediate:
                z.append(z_t)
            # One Euler step
            z_t, drift_t, diffusion_t = self.sde_update_step(z_t, t_cur.unsqueeze(0).expand(len(z_t)),
                                                             (t_next - t_cur).item(),
                                                             not sde or k == total_nb_steps - 1)
            if return_intermediate:
                drift.append(drift_t)
                diffusion.append(diffusion_t)
            if k == nb_steps - 1:
                break
        if return_intermediate:
            z.append(z_t)
            return torch.stack(z, dim=1), torch.stack(drift, dim=1), torch.stack(diffusion, dim=1)
        else:
            return z_t, torch.zeros_like(z_t), torch.zeros_like(z_t)

    def edm_generate_like(self, x: torch.Tensor, nb_steps: int, sigma_min: float, sigma_max: float, rho: float,
                          S_churn: float, S_min: float, S_max: float, S_noise: float, sde: bool,
                          total_nb_steps: int | None = None, z_0: torch.Tensor | None = None,
                          return_intermediate: bool = False) -> SolverTuple:
        """
        EDM / Heun sampler. Applies the given number of discretization steps, provided the overall discretization
        level. Uses the input initial particles if provided, otherwise samples them. Returns both the generated output
        and the gradients it receives.

        If required, returns all intermediary results.
        """
        if total_nb_steps is None:
            total_nb_steps = nb_steps
        sigma_min = max(sigma_min, self.denoiser.sigma_min)
        sigma_max = min(sigma_max, self.denoiser.sigma_max)
        z = []
        drift = []
        diffusion = []
        z_t = self.sample_latent_like(x, sigma_max) if z_0 is None else z_0  # Initial sample from the prior
        steps = self.sample_steps(total_nb_steps, sigma_min, sigma_max, rho, x.device)  # Selection of times
        for k, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            if return_intermediate:
                z.append(z_t)
            # One Heun step
            z_t, drift_t, diffusion_t = self.edm_update_step(z_t, t_cur.unsqueeze(0).expand(len(z_t)),
                                                             t_next.unsqueeze(0).expand(len(z_t)), S_churn, S_min,
                                                             S_max, S_noise, total_nb_steps, k == total_nb_steps - 1,
                                                             not sde)
            if return_intermediate:
                drift.append(drift_t)
                diffusion.append(diffusion_t)
            if k == nb_steps - 1:
                break
        if return_intermediate:
            z.append(z_t)
            return torch.stack(z, dim=1), torch.stack(drift, dim=1), torch.stack(diffusion, dim=1)
        else:
            return z_t, torch.zeros_like(z_t), torch.zeros_like(z_t)

    def sde_update_step(self, x: torch.Tensor, t: torch.Tensor, delta: float, deterministic: bool) -> SolverTuple:
        """
        Implements an Euler step on the generative SDE at the given time with the given step size.

        Uses the probability flow ODE if requested.
        """
        drift = (x - self.denoiser(x, t)) / fill_as(t, x) * delta
        if not deterministic:
            drift *= 2
            w = torch.normal(0, 1, x.size(), device=x.device)
            diffusion = fill_as(torch.sqrt(2 * t * abs(delta)), w) * w
            update = drift + diffusion
        else:
            update = drift
            diffusion = torch.zeros_like(drift)
        return x + update, drift, diffusion

    def edm_update_step(self, x: torch.Tensor, t_cur: torch.Tensor, t_next: torch.Tensor, S_churn: float, S_min: float,
                        S_max: float, S_noise: float, nb_steps: int, last_step: bool,
                        deterministic: bool) -> SolverTuple:
        """
        Implements an Heun step on the generative SDE at the given time with the given step size.

        Uses the probability flow ODE if requested.

        Code inspired by the EDM implementation (https://github.com/NVlabs/edm, file `generate.py`)
        """
        if not deterministic:
            # Adds noise
            gamma = torch.where(torch.logical_and(S_min <= t_cur, t_cur <= S_max),
                                min(S_churn / nb_steps, math.sqrt(2) - 1) * torch.ones_like(t_cur),
                                torch.zeros_like(t_cur))
            t_hat = (1 + gamma) * t_cur
            diffusion = torch.sqrt(fill_as(t_hat ** 2 - t_cur ** 2, x)) * torch.normal(0, S_noise, x.size(),
                                                                                       device=x.device)
            x_hat = x + diffusion
        else:
            x_hat, t_hat = x, t_cur
            diffusion = torch.zeros_like(x)

        # First derivative
        d_cur = (x_hat - self.denoiser(x_hat, t_hat)) / fill_as(t_hat, x_hat)
        drift = fill_as(t_next - t_hat, x_hat) * d_cur
        x_next = x_hat + drift

        if not last_step:
            # Averaged derivative with the next one
            d_prime = (x_next - self.denoiser(x_next, t_next)) / fill_as(t_next, x_next)
            drift_2 = fill_as(t_next - t_hat, x_hat) * (d_cur + d_prime) / 2
            x_next = x_hat + drift_2
            drift = drift_2

        return x_next, drift, diffusion

    def forward(self, x: torch.Tensor, op: Any) -> tuple[torch.Tensor]:
        """
        Denoising score matching loss.
        """
        sigma = self.sigma_distr.sample(x.size()[:1]).to(x)  # type: ignore
        weight = (sigma ** 2 + self.denoiser.sigma_data ** 2) / (sigma * self.denoiser.sigma_data) ** 2
        noise = torch.normal(0, 1, x.size(), device=x.device)
        std = fill_as(sigma, x)
        perturbed_x = x + noise * std
        x_hat = self.denoiser(perturbed_x, sigma)
        mse = EDM.mse_loss(x_hat, x)
        loss = weight * torch.sum(mse, dim=list(range(1, len(x.size()))))
        return loss,

    @classmethod
    def training_step(cls, step: int, batch: Batch, forward_fn: ForwardFn, optimizers: Optimizers, scalers: Scalers,
                      schedulers: Schedulers, device: torch.device, shared_rng: random.Random,
                      opt: ModelDict) -> tuple[float, Log, list[str]]:
        optimizer, scaler, scheduler = cls.extract_optimizer(EDM.score_optimizer_name, optimizers, scalers,
                                                             schedulers)
        optimizer.zero_grad()
        x = batch.x.to(device)
        loss, = forward_fn(x, None)
        loss = torch.mean(loss)
        optimizer_update(loss, optimizer, scaler, scheduler, opt)
        loss = loss.item()
        return loss, {'loss': loss}, ['loss']

    def inference(self, x: torch.Tensor, gen_params: EDMInferenceDict, return_intermediate: bool,
                  z_0: torch.Tensor | None = None) -> SolverTuple:
        nb_steps = gen_params.nb_steps
        sigma_min = gen_params.sigma_min
        sigma_max = gen_params.sigma_max
        rho = gen_params.rho
        if gen_params.method is EDM.Inference.Euler:
            return self.sde_generate_like(x, nb_steps, sigma_min, sigma_max, rho, gen_params.sde, z_0=z_0,
                                          return_intermediate=return_intermediate)
        elif gen_params.method is EDM.Inference.EDM:
            gen_params = EDMHeunInferenceDict(gen_params)
            return self.edm_generate_like(x, nb_steps, sigma_min, sigma_max, rho, gen_params.S_churn, gen_params.S_min,
                                          gen_params.S_max, gen_params.S_noise, gen_params.sde, z_0=z_0,
                                          return_intermediate=return_intermediate)
        else:
            raise ValueError(f'Inference method `{gen_params.method}` not implemented yet.')

    def evaluation_step(self, batch: Batch, device: torch.device, opt: ModelDict, config: TestDict) -> EvaluationTuple:
        """
        Generates samples following the inference configuration.
        """
        x = batch.x.to(device)
        score_loss, = self.forward(x, None)
        gen_params = EDMInferenceDict(config.test_params)
        return_intermediate = Test.GEN_QUALI in config.tests
        x_gen, drift, diffusion = self.inference(x, gen_params, return_intermediate)
        if Test.INTER_QUALI in config.tests:
            # Interpolations in the prior
            sigma_max = gen_params.sigma_max
            z1 = self.sample_latent_like(x, sigma_max)
            z2 = self.sample_latent_like(x, sigma_max)
            interpolated_z = interpolate(z1 / sigma_max, z2 / sigma_max,
                                         gen_params.resolution, 1, gen_params.interpolation) * sigma_max
            interpolated_z = interpolated_z.flatten(end_dim=1)
            interpolations = self.inference(x, gen_params, True, z_0=interpolated_z)[0]
            interpolations = interpolations.view((len(x), gen_params.resolution) + interpolations.size()[1:])
        else:
            interpolations = torch.zeros((len(x)))
        return x, x_gen, drift, diffusion, score_loss, interpolations

    def evaluation_logs(self, eval_results: EvaluationTuple, device: torch.device, opt: ModelDict,
                        config: TestDict) -> tuple[float, Log]:
        """
        If specified in the test configuration, computes the FID, creates sample images and/or saves the generated
        samples.
        """
        x, x_gen, drift, diffusion, score_loss, interpolations = eval_results
        score_loss = score_loss.mean().item()

        fid, gen_grid, inter_grid = None, None, None
        for test in config.tests:
            if test is Test.GEN_QUALI:
                grad_norm = torch.linalg.norm((drift + diffusion).flatten(start_dim=2), dim=2).mean().item() / 2
                drift /= (grad_norm + 1e-6)
                diffusion /= (grad_norm + 1e-6)
                gen_grid = [gen_quali(x, x_gen[:, i], config.nb_quali,
                                      gradients=[drift[:, i], diffusion[:, i]] if i < drift.size(1) else None,
                                      scale_grad=0.4) for i in range(x_gen.size(1))]
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
        logs: Log = {'loss': score_loss}
        if fid is not None:
            logs['fid'] = fid
            if Test.FID is config.metric:
                score = -fid
        if gen_grid is not None:
            logs['gen_quali'] = gen_grid
        if inter_grid is not None:
            logs['inter_quali'] = inter_grid
        if Test.GEN_SAVE in config.tests:
            logs['gen'] = x_gen
            if inter_grid is not None:
                logs['inter'] = interpolations

        return score if score is not None else -score_loss, logs
