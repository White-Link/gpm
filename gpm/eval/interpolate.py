import enum
import math
import scipy
import torch

import torch.distributions as D
import torch.nn.functional as F


cauchy = D.Cauchy(0, 1)
gaussian = D.Normal(0, 1)


@enum.unique
class Interpolation(str, enum.Enum):
    LINEAR = 'linear'  # Linear interpolation
    SPHERICAL = 'spherical'  # Spherical interpolation
    CAUCHY = 'cauchy'  # Cauchy-linear interpolation (https://openreview.net/forum?id=SyMhLo0qKQ)
    SPHERICAL_CAUCHY = 'spherical_cauchy'  # Spherical Cauchy-linear interpolation


def lerp(start: torch.Tensor, end: torch.Tensor, n: int) -> torch.Tensor:
    """
    Linear interpolation.
    """
    return torch.lerp(start.unsqueeze(1), end.unsqueeze(1),
                      torch.linspace(0, 1, n, device=start.device).unsqueeze(0).unsqueeze(-1))


def slerp(start: torch.Tensor, end: torch.Tensor, n: int, normalized: bool = False) -> torch.Tensor:
    """
    Spherical interpolation.
    """
    if normalized:
        cos = torch.clamp(torch.einsum('bd, bd -> b', start, end), min=-1, max=1)
    else:
        cos = torch.clamp(torch.einsum('bd, bd -> b', F.normalize(start), F.normalize(end)), min=-1, max=1)
    omega = torch.arccos(cos).unsqueeze(-1).unsqueeze(-1) / math.pi
    div = torch.sinc(omega)
    start = start.unsqueeze(1)
    end = end.unsqueeze(1)
    t = torch.linspace(0, 1, steps=n, device=start.device).unsqueeze(0).unsqueeze(-1)
    return (1 - t) * torch.sinc((1 - t) * omega) / div * start + t * torch.sinc(t * omega) / div * end


def g_clerp(x: torch.Tensor) -> torch.Tensor:
    """
    Utility function for Cauchy-linear interpolation.
    """
    return cauchy.icdf(gaussian.cdf(x))


def inv_g_clerp(x: torch.Tensor) -> torch.Tensor:
    """
    Utility function for Cauchy-linear interpolation.
    """
    return gaussian.icdf(cauchy.cdf(x))


def clerp(start: torch.Tensor, end: torch.Tensor, n: int) -> torch.Tensor:
    """
    Cauchy-linear interpolation (https://openreview.net/forum?id=SyMhLo0qKQ).
    """
    t = torch.linspace(0, 1, steps=n, device=start.device).unsqueeze(0).unsqueeze(-1)
    return g_clerp((1 - t) * inv_g_clerp(start).unsqueeze(1) + t * inv_g_clerp(end).unsqueeze(1))


def g_sclerp(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Utility function for spherical Cauchy-linear interpolation.
    """
    sp_in = (x * x).data.cpu().numpy()
    sp_out = torch.tensor(scipy.stats.chi2.cdf(sp_in, d)).to(x)
    return cauchy.icdf(sp_out)


def inv_g_sclerp(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Utility function for spherical Cauchy-linear interpolation.
    """
    sp_in = cauchy.cdf(x).data.cpu().numpy()
    sp_out = torch.tensor(scipy.stats.chi2.ppf(sp_in, d)).to(x)
    return torch.sqrt(sp_out)


def sclerp(start: torch.Tensor, end: torch.Tensor, n: int) -> torch.Tensor:
    """
    Spherical Cauchy-linear interpolation (https://openreview.net/forum?id=SyMhLo0qKQ).
    No backward is provided on this function because on the underlying use of scipy.
    """
    d = start.size(-1)
    t = torch.linspace(0, 1, steps=n, device=start.device).unsqueeze(0).unsqueeze(-1)
    unit_interpolation = slerp(F.normalize(start), F.normalize(end), n)
    start_norm = torch.linalg.norm(start, dim=-1, keepdim=True).unsqueeze(1)
    end_norm = torch.linalg.norm(end, dim=-1, keepdim=True).unsqueeze(1)
    norm_interpolation = g_sclerp((1 - t) * inv_g_sclerp(start_norm, d) + t * inv_g_sclerp(end_norm, d), d)
    return unit_interpolation * norm_interpolation


def interpolate(start: torch.Tensor, end: torch.Tensor, n: int, start_dim: int,
                method: Interpolation) -> torch.Tensor:
    """
    Produces n interpolations between two tensors of the same shape starting from a given dimension using one of four
    available methods: linear, spherical, Cauchy or Cauchy linear (https://openreview.net/forum?id=SyMhLo0qKQ).
    Adds the interpolations in the second dimension of the tensor, after the batch one.
    """
    flat_start = start.flatten(start_dim=start_dim).flatten(end_dim=start_dim - 1)
    flat_end = end.flatten(start_dim=start_dim).flatten(end_dim=start_dim - 1)
    if method is Interpolation.LINEAR:
        interpolation = lerp(flat_start, flat_end, n)
    elif method is Interpolation.SPHERICAL:
        interpolation = slerp(flat_start, flat_end, n)
    elif method is Interpolation.CAUCHY:
        interpolation = clerp(flat_start, flat_end, n)
    elif method is Interpolation.SPHERICAL_CAUCHY:
        interpolation = sclerp(flat_start, flat_end, n)
    else:
        raise ValueError(f'No interpolation named `{method}`')
    return interpolation.view((len(start), n) + start.size()[1:])
