from argparse import ArgumentParser
from pathlib import Path
import traceback

import gin.torch
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from torch import nn

from modules import MODULES
from data import DATASETS
from nets import NETS

from utils import gin_config_to_readable_dictionary


# Register all modules, datasets and netitectures with gin
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
    params: dict = {},
) -> None:
    """Train a model using the given module, datamodule and netitecture"""

    net = net()
    module = module(net=net)
    datamodule = datamodule()

    # get all parameters tracked by gin config
    gin_config_dict = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)

    # get the lighting wandb wrapper to log training
    wandb_logger = WandbLogger(project=project_name, save_dir=save_dir)
    wandb_logger.log_hyperparams(gin_config_dict)

    # log the number of parameters in the network
    wandb_logger.experiment.config["param_count"] = net.get_parameter_count()

    trainer = Trainer(logger=wandb_logger, **params)

    try:
        trainer.fit(
            model=module,
            datamodule=datamodule,
        )
    except Exception:
        traceback.print_exc()
        pass

    # trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-file", type=Path, default="cfg/config.gin")

    args = parser.parse_args()

    gin.parse_config_file(args.config_file)
    gin.finalize()

    train()
