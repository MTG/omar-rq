from argparse import ArgumentParser
from pathlib import Path
import traceback

import gin.torch
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from torch import nn

from cosineannealingscheduler import CosineAnnealingCallback
from data import DATASETS
from modules import MODULES
from nets import NETS
from utils import gin_config_to_readable_dictionary


# Register all modules, datasets and networs with gin
for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)


@gin.configurable
def train(
    project_name: str,
    save_dir: str,
    module: L.LightningModule,
    datamodule: L.LightningDataModule,
    net: nn.Module,
    representation: nn.Module,
    params: dict,
) -> None:
    """Train a model using the given module, datamodule and netitecture"""

    net = net()
    representation = representation()
    module = module(net=net, representation=representation)
    datamodule = datamodule()

    # get the lightning wandb logger wrapper and log the config
    gin_config_dict = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)
    wandb_logger = WandbLogger(project=project_name, save_dir=save_dir)
    wandb_logger.log_hyperparams(gin_config_dict)

    # log the number of parameters in the network (required to compute scaling laws)
    # wandb_logger.experiment.config["param_count"] = net.get_parameter_count()

    # create callbacks
    cosine_annealing_callback = CosineAnnealingCallback(total_steps=params["max_steps"])
    callbacks = [cosine_annealing_callback]

    # create the trainer and fit the model
    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **params)
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser("Train SSL models using gin config")
    parser.add_argument("--config-file", type=Path, default="cfg/config.gin")

    args = parser.parse_args()

    try:
        gin.parse_config_file(args.config_file)
        gin.finalize()

        train()
    except Exception:
        traceback.print_exc()
