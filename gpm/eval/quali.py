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
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F

from matplotlib import cm
from PIL.Image import Image
from typing import Callable, Sequence

from gpm.eval.utils import finalize_plot, make_2d_grid, quiver_2d_ax, scatter_2d_ax


def is_image(x: torch.Tensor) -> bool:
    return len(x.size()) == 4


def is_flat(x: torch.Tensor) -> bool:
    return len(x.size()) == 2


def gen_quali(x: torch.Tensor, x_gen: torch.Tensor, nb_quali: int,
              gradients: Sequence[torch.Tensor | None] | torch.Tensor | None = None, scale_grad: float | None = None,
              loss: Callable[[torch.Tensor], torch.Tensor] | None = None, resolution: int | None = None,
              width: float = 5, height: float = 5) -> Image:
    """
    Creates a visualization of true and generated samples.

    For two-dimensional data, also plots input gradient vector fields and pointwise loss, when applicable.
    """

    if is_image(x) and is_image(x_gen):
        # Show images in a grid
        return F.to_pil_image(torchvision.utils.make_grid(torch.cat([x[:nb_quali], x_gen[:nb_quali]]), nrow=nb_quali,
                                                          normalize=True, value_range=(-1, 1), pad_value=1))
    else:
        # Plot everythings in a scatter plot
        assert is_flat(x) and is_flat(x_gen) and x.size(1) == x_gen.size(1) == 2
        plt.figure(figsize=(width, height))
        ax = plt.gca()
        scatter_2d_ax(x[:nb_quali], ax=ax, label='target', alpha=0.75, c='red', marker='D', edgecolors='white', s=50)
        scatter_2d_ax(x_gen[:nb_quali], ax=ax, label='generated', alpha=0.75, c='blue', marker='o', edgecolors='white',
                      s=50)
        if gradients is not None:
            if isinstance(gradients, Sequence):
                colors = iter(cm.get_cmap('plasma')(np.linspace(0, 1, len(gradients))))
                for gradient in gradients:
                    if gradient is not None:
                        quiver_2d_ax(x_gen, gradient, ax=ax, angles='xy', scale_units='xy', linestyle='solid',
                                     headwidth=3.5, width=0.006, alpha=0.9, color=next(colors), edgecolor='black',
                                     linewidth=0.2, scale=scale_grad)
            else:
                quiver_2d_ax(x_gen, gradients, ax=ax, angles='xy', scale_units='xy', linestyle='solid', headwidth=3.5,
                             width=0.006, alpha=0.9, color='white', edgecolor='black', linewidth=0.2, scale=scale_grad)
        if loss is not None:
            assert resolution is not None
            scatter_extent = ax.axis()
            grid = make_2d_grid(*scatter_extent, res_x=resolution, res_y=resolution)
            ax.imshow(loss(grid).view(resolution, resolution).numpy().transpose(), origin='lower',
                      extent=scatter_extent, aspect='auto')
        return finalize_plot()
