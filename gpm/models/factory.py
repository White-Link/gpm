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


from gpm.data.base import BaseDataset
from gpm.data.image.base import BaseImageDataset
from gpm.data.lowd.base import BaseLowDDataset
from gpm.models.base import BaseModel
from gpm.models.discr_flow.base import DiscrFlowDict
from gpm.models.discr_flow.models import HingeDiscrFlow, IPMDiscrFlow, VanillaDiscrFlow
from gpm.models.gan.base import GANDict, GANModel
from gpm.models.gan.models import HingeGAN, IPMGAN, VanillaGAN
from gpm.models.models import Model
from gpm.models.edm.edm import EDMDict, EDM
from gpm.models.score_gan.score_gan import ScoreGAN, ScoreGANDict
from gpm.utils.config import ModelDict


def model_factory(opt: ModelDict, dataset: BaseDataset) -> BaseModel:
    """
    Constructs a model based on the provided options.
    """
    if opt.model is Model.DISCR_FLOW:
        assert isinstance(dataset, (BaseLowDDataset, BaseImageDataset))
        config = DiscrFlowDict(opt.model_params)
        if config.model is GANModel.VANILLA:
            discr_diffusion_model = VanillaDiscrFlow
        elif config.model is GANModel.IPM:
            discr_diffusion_model = IPMDiscrFlow
        elif config.model is GANModel.HINGE:
            discr_diffusion_model = HingeDiscrFlow
        else:
            raise ValueError(f'No GAN model named `{config.model}`')
        return discr_diffusion_model(dataset, config)
    elif opt.model is Model.GAN:
        assert isinstance(dataset, (BaseLowDDataset, BaseImageDataset))
        config = GANDict(opt.model_params)
        if config.model is GANModel.VANILLA:
            gan_model = VanillaGAN
        elif config.model is GANModel.IPM:
            gan_model = IPMGAN
        elif config.model is GANModel.HINGE:
            gan_model = HingeGAN
        else:
            raise ValueError(f'No GAN model named `{config.model}`')
        return gan_model(dataset, config)
    elif opt.model is Model.EDM:
        assert isinstance(dataset, (BaseLowDDataset, BaseImageDataset))
        config = EDMDict(opt.model_params)
        return EDM(dataset, config)
    elif opt.model is Model.SCORE_GAN:
        assert isinstance(dataset, (BaseLowDDataset, BaseImageDataset))
        config = ScoreGANDict(opt.model_params)
        return ScoreGAN(dataset, config)
    else:
        raise ValueError(f'No model named `{opt.model}`')
