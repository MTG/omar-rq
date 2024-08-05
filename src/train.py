from argparse import ArgumentParser
from datetime import datetime
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
from utils import gin_config_to_readable_dictionary, build_module, build_dev_datamodule
from callbacks import GinConfigSaverCallback


# Register all modules, datasets and networs with gin
for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)

for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)


@gin.configurable
def train(
    module: L.LightningModule,
    datamodule: L.LightningDataModule,
    params: dict,
    wandb_params: dict,
    config_file: Path,
) -> None:
    """Train a model using the given module, datamodule and netitecture"""

    # get the lightning wandb logger wrapper and log the config
    gin_config_dict = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)
    wandb_logger = WandbLogger(**wandb_params)
    wandb_logger.log_hyperparams(gin_config_dict)

    # log the number of parameters in the network (required to compute scaling laws)
    # tb_logger.experiment.config["param_count"] = net.get_parameter_count()

    # create callbacks
    cosine_annealing_callback = CosineAnnealingCallback(total_steps=params["max_steps"])
    config_save_callback = GinConfigSaverCallback(config_file)
    callbacks = [cosine_annealing_callback, config_save_callback]

    # create the trainer and fit the model
    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **params)
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser("Train SSL models using gin config")
    parser.add_argument(
        "--train-config",
        type=Path,
        default="cfg/train_config.gin",
        help="Path to the gin config file for training.",
    )

    args = parser.parse_args()

    try:
        gin.parse_config_file(args.train_config)

        module = build_module()
        datamodule = build_dev_datamodule()

        gin.finalize()

        train(module, datamodule, config_file=args.train_config)
    except Exception:
        traceback.print_exc()
