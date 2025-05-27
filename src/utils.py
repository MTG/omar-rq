from pathlib import Path
from typing import List

import gin.torch
import pytorch_lightning as L
from torch import nn

from data import DATASETS


for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)


def gin_config_to_readable_dictionary(gin_config: dict):
    """
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B

    Copied from https://github.com/google/gin-config/issues/154

    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = v

    return data


@gin.configurable
def build_module(
    representation: nn.Module | List,
    net: nn.Module,
    module: L.LightningModule,
    trainer: L.Trainer = None,
    ckpt_path: Path = None,
):
    """Build the module from the provided references. If a checkpoint path is provided,
    load the checkpoint. Otherwise, create a new model. Returns the checkpoint path so that
    Lightning Trainer can use it to restore the training."""

    # Evaluate the provided references, i.e. convert the strings to the actual objects
    if isinstance(representation, list):
        representation = nn.ModuleList([r() for r in representation])
    else:
        representation = representation()

    net = net()

    if ckpt_path is not None:  # Load the checkpoint if provided
        print(f"Loading checkpoint from {ckpt_path}")
        if trainer is not None:
            # NOTE: this is necessary for prediciton, it correctly sets the model precision
            # https://github.com/Lightning-AI/pytorch-lightning/discussions/7730
            with trainer.init_module(empty_init=True):
                module = module.load_from_checkpoint(
                    ckpt_path, net=net, representation=representation, strict=False
                )
        else:
            module = module.load_from_checkpoint(
                ckpt_path, net=net, representation=representation, strict=False
            )
    else:  # Otherwise, create from random initialization
        print("Creating a new model")
        module = module(net=net, representation=representation)

    return module, ckpt_path


@gin.configurable
def build_dev_datamodule(
    datamodule: L.LightningDataModule,
):
    datamodule = datamodule()
    return datamodule
