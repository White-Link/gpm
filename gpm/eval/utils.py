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


import io
import torch

import matplotlib.pyplot as plt
import PIL.Image as Image

from matplotlib.axes import Axes


def make_2d_grid(x_min: float, x_max: float, y_min: float, y_max: float, res_x: int = 20,
                 res_y: int = 20) -> torch.Tensor:
    x_grid = torch.linspace(x_min, x_max, res_x)
    y_grid = torch.linspace(y_min, y_max, res_y)
    X_grid, Y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')
    return torch.stack([X_grid, Y_grid], -1).view(res_x * res_y, 2)


def scatter_2d_ax(x: torch.Tensor, *args, ax: Axes | None = None, **kwargs):
    target = plt if ax is None else ax
    x = x.numpy()
    target.scatter(x[:, 0], x[:, 1], *args, **kwargs)


def quiver_2d_ax(x: torch.Tensor, grad: torch.Tensor, *args, ax: Axes | None = None, **kwargs):
    target = plt if ax is None else ax
    x = x.numpy()
    target.quiver(x[:, 0], x[:, 1], grad[:, 0], grad[:, 1], *args, **kwargs)


def finalize_plot() -> Image.Image:
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer)
    buffer.seek(0)
    plot = Image.open(buffer)
    plt.close()
    return plot
