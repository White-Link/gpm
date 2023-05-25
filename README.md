<!-- Copyright 2023 Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickaël Chen, Alain Rakotomamonjy

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->


# GPM: Unifying GANs and Score-Based Diffusion as Generative Particle Models

Official implementation of the paper *Unifying GANs and Score-Based Diffusion as Generative Particle Models* (Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickaël Chen, Alain Rakotomamonjy).

## [Animated Samples](https://jyfranceschi.fr/publications/gpm/)

![Discriminator Flow samples on MNIST](https://jyfranceschi.fr/wp-content/uploads/2023/05/discr_flow_mnist.webp)

![Discriminator Flow samples on Gaussians](https://jyfranceschi.fr/wp-content/uploads/2023/05/discr_flow_gaussians.webp)

## Requirements

All models were trained with Python 3.10.4 and PyTorch 1.13.1 using CUDA 11.8. The [`requirements.txt`](requirements.txt) file lists Python package dependencies.

## Launch an Experiment

To launch an experiment, you need:
- a path to the data `$DATA_PATH`;
- a path to a YAML config file (whose format depends on the chosen model, but examples are shown in the [`config`](config) folder) `$CONFIG_FILE`;
- a path where the logs and checkpoints should be saved `$LOG_PATH`;
- the chosen name of the experiment `$NAME`.

You can then launch in the root folder of the project the following command:
```bash
bash launch.sh --data_path $DATA_PATH --configs $CONFIG_FILE --save_path $LOG_PATH --save_name $NAME
```

There are some optional arguments to this command which you can find with `bash launch.sh --help`. In particular, `--device $DEVICE1 $DEVICE2 ...` launches training on GPUs `$DEVICE1`, `$DEVICE2`, etc. (e.g. `--device 0 1 2 3`). By default, multi-GPU training uses PyTorch's [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and PyTorch's launching utility [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html). As of now, multi-GPU training only supports single-node parallelism but extending it to multi-node should be straightforward.

Please do not include any log file or data in code folders as it may prevent this code from properly working. Indeed, this code automatically saves the content of the project root folder in the log directory.

## Testing and Loading

By default, a test in performed at the end of training on the model corresponding to the last iteration only. If you want to test the best saved model, you can first load the checkpoint using the `--load` option, then load the best model by additionnally using the `--load_best` option. This will resume training if it was not completed; if you want to only test the model, add the `--only_test` option.

Before loading a model, please create a backup of the experiment folder as some files may be overwritten in the process.

## Implemented Models

Implemented models are listed in [`gpm/models/models.py`](gpm/models/models.py). In particular: a basic GAN implementation, a score-based model (reimplementation of [EDM](https://github.com/NVlabs/edm)), our Discriminator Flow, and our ScoreGAN model which requires a pretrained data score network.

The [`config`](config) folder lists configuration files with suggested parameters that were used in the paper. A documented example can be found at [`config/discr_flow/celeba/config.yaml`](config/discr_flow/celeba/config.yaml). All possible parameter dictionaries are documented in the code (cf. discussion below).

## Framework Design

This code is meant to be generic enough so that it can be used to create pretty much any PyTorch deep model, and for example extend the proposed models. On the downside, it lets the user responsible of many technical details and of complying with the general organization.

All training initialization and iterations (seed, devices, parallelism on a single node, model creation, data loading, training and testing loops, etc.) are handled in [`gpm/train.py`](gpm/train.py); options for the launch command are in [`gpm/utils/args.py`](gpm/utils/args.py). All neural networks used in the models are implemented in the different files of [`gpm/networks`](gpm/networks); models are coded separately in [`gpm/models`](gpm/models).

To keep the code generic, model configurations are provided as YAML files, with examples in the [`config`](config) folder. The expected arguments differ depending on the chosen models and networks. To specify which arguments are needed for e.g. a network, we use the following kind of code (here is an example for [an MLP](gpm/networks/mlp.py)).
```python
class MLPDict(DotDict):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        assert {'depth', 'width', 'activation'}.issubset(self.keys())
        self.depth: int  # Number of hidden layers
        self.width: int  # Width of hidden layers
        self.activation: ActivationDict | list[ActivationDict]  # Activations of hidden layers
        if isinstance(self.activation, list):
            self.activation = list(map(ActivationDict, self.activation))
        else:
            self.activation = ActivationDict(self.activation)
        self.batch_norm: bool  # Whether to incorporate batch norm (default, False)
        if 'batch_norm' not in self:
            self.batch_norm = False
        self.spectral_norm: bool  # Whether to perform spectral norm (default, False)
        if 'spectral_norm' not in self:
            self.spectral_norm = False
```
The `assert` ensures that all required arguments are in the provided configuration file. The rest of the class attributes is just syntax sugar to provide typing information on these arguments, or specifies default values for some parameters.

Model coding is based on object-oriented programming: each model is a single `torch.nn.Module` containing all the model parameters and inheriting from the abstract class [`BaseModel`](gpm/models/base.py) which indicates which methods any model should reimplement. You can follow the documentation for more details. Datasets follow the same philosophy, inheriting the `Dataset` class from PyTorch and the custom abstract class [`BaseDataset`](gpm/data/base.py).

Log folders are organized as follows, as coded by the [internal logger](gpm/utils/logger.py):
- a `chkpt` folder containing checkpoints needed to resume training, including the last checkpoint and the best checkpoint w.r.t. the evaluation metric;
- an `eval` and a `test` folder containing evaluations on the validation and test sets (there can be several evaluation configurations at once);
- a `config.json` file containing the training configuration;
- `logs.*.json` files containing training, validation and testing logs, including all logges metrics;
- `result*.json` files containing information on the saved checkpoints (last and best);
- a `source.zip` file containing the launched source code;
- a `test.yaml` file containing the original YAML configuration file.

## A Note on License

This code is open-source. We share most of it under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

However, we reuse code from [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch) and [EDM](https://github.com/NVlabs/edm) where were released under more restrictive licenses (respectively, [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) and [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) that require redistribution under the same license or equivalent. Hence, the corresponding parts of our code (respectively, [`gpm/networks/conv/fastgan`](gpm/networks/conv/fastgan) and [`gpm/networks/score`](gpm/networks/score)) are open-sourced using the original licenses of these works and not Apache. See the corresponding folders for the details.
