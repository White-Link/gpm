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


import torch

from gpm.data.base import BaseDataset
from gpm.data.image.base import BaseImageDataset
from gpm.networks.activations import activation_factory
from gpm.networks.conv.dcgan import DCGANEncoder, DCGANEncoderDict, DCGANDecoder, DCGANDecoderDict
from gpm.networks.conv.fastgan.fastgan import FastGANDecoder, FastGANDecoderDict, FastGANEncoder, FastGANEncoderDict
from gpm.networks.conv.resnet import ResNetDecoder, ResNetDecoderDict, ResNetEncoder, ResNetEncoderDict
from gpm.networks.mlp import MLP, MLPEncoderDict, MLPDecoderDict, MLPScoreDict
from gpm.networks.score.ddpm import SongUNet, SongUNetDict
from gpm.networks.score.edm import EDMDenoiser
from gpm.networks.utils import Encoder, EncoderDict, Decoder, DecoderDict, Score, EDMDenoiserDict, ScoreDict


def encoder_factory(dataset: BaseDataset | None, in_size: tuple[int, ...], out_size: int, config: EncoderDict,
                    additional_input: int | None = None, decoder: torch.nn.Module | None = None) -> torch.nn.Module:
    """
    Constructs an encoder based on the provided parameters.

    Offers the option to add a second input to the encoder, if compatible with the architecture.
    """
    name = config.name
    if name is Encoder.MLP:
        config = MLPEncoderDict(config)
        encoder = MLP(in_size[0] if len(in_size) == 1 else in_size, out_size, config, compl_size=additional_input)
    else:
        assert isinstance(dataset, BaseImageDataset)
        if name is Encoder.DCGAN:
            assert len(in_size) == 3 and (in_size[1] == in_size[2] == 64 or in_size[1] == in_size[2] == 32)
            config = DCGANEncoderDict(config)
            encoder = DCGANEncoder(in_size[0], in_size[1], out_size, config, additional_input=additional_input)
        elif name is Encoder.RESNET:
            assert len(in_size) == 3 and in_size[1] == in_size[2] and in_size[1] in {32, 64, 128, 256, 512}
            config = ResNetEncoderDict(config)
            encoder = ResNetEncoder(in_size[0], in_size[1], out_size, config, additional_input=additional_input)
        elif name is Encoder.FASTGAN:
            assert len(in_size) == 3 and in_size[1] == in_size[2] and in_size[1] in {32, 64, 128, 256, 512}
            config = FastGANEncoderDict(config)
            encoder = FastGANEncoder(in_size[0], in_size[1], out_size, config, additional_input=additional_input)
        else:
            raise ValueError(f'Encoder `{name}` not yet implemented')
    if config.final_activation is not None:
        return torch.nn.Sequential(encoder, activation_factory(config.final_activation))
    else:
        return encoder


def decoder_factory(dataset: BaseDataset | None, in_size: int, out_size: int | tuple[int, ...], config: DecoderDict,
                    skip: bool = False) -> torch.nn.Module:
    """
    Constructs a decoder based on the provided parameters.
    """
    name = config.name
    if name is Decoder.MLP:
        config = MLPDecoderDict(config)
        decoder = MLP(in_size, out_size[0] if isinstance(out_size, tuple) and len(out_size) == 1 else out_size, config)
    else:
        assert isinstance(dataset, BaseImageDataset)
        if name is Decoder.DCGAN:
            assert isinstance(out_size, tuple)
            assert len(out_size) == 3 and (out_size[1] == out_size[2] == 64 or out_size[1] == out_size[2] == 32)
            config = DCGANDecoderDict(config)
            decoder = DCGANDecoder(in_size, out_size[0], out_size[1], config, skip=skip)
        elif name is Decoder.RESNET:
            assert isinstance(out_size, tuple)
            assert len(out_size) == 3 and out_size[1] == out_size[2] and out_size[1] in {32, 64, 128, 256, 512}
            config = ResNetDecoderDict(config)
            decoder = ResNetDecoder(in_size, out_size[0], out_size[1], config)
        elif name is Decoder.FASTGAN:
            assert isinstance(out_size, tuple)
            assert len(out_size) == 3 and out_size[1] == out_size[2] and out_size[1] in {32, 64, 128, 256, 512}
            config = FastGANDecoderDict(config)
            decoder = FastGANDecoder(in_size, out_size[0], out_size[1], config)
        else:
            raise ValueError(f'Decoder `{name}` not yet implemented')
    if config.final_activation is not None:
        decoder = torch.nn.Sequential(decoder, activation_factory(config.final_activation))
    return decoder


def score_factory(dataset: BaseDataset | None, in_size: tuple[int, ...], config: ScoreDict,
                  embedding_size: int = 0) -> torch.nn.Module:
    """
    Constructs a score function based on the provided parameters.
    """
    name = config.name
    if name is Score.MLP:
        config = MLPScoreDict(config)
        network = MLP(in_size, in_size, config, compl_size=embedding_size)
    elif name is Score.DDPM:
        assert isinstance(dataset, BaseImageDataset)
        config = SongUNetDict(config)
        network = SongUNet(dataset.image_size, dataset.channels, dataset.channels, embedding_size, config)
    else:
        raise ValueError(f'Network `{name}` not yet implemented')
    if config.final_activation is not None:
        network = torch.nn.Sequential(network, activation_factory(config.final_activation))
    return network


def edm_denoiser_factory(dataset: BaseDataset | None, in_size: tuple[int, ...],
                         config: EDMDenoiserDict) -> EDMDenoiser:
    """
    Constructs a score function based on the provided parameters and wraps it as an EDM score function with the
    dedicated parameters.
    """
    network = score_factory(dataset, in_size, config, embedding_size=config.embedding_size)
    return EDMDenoiser(config.sigma_min, config.sigma_max, config.sigma_data, network, config.embedding,
                       config.embedding_size)
