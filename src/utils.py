import gin.torch
import pytorch_lightning as L
from torch import nn


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
    representation: nn.Module,
    net: nn.Module,
    module: L.LightningModule,
    ckpt_path: str = None,
):

    # Evaluate the provided references, i.e. convert the strings to the actual objects
    representation = representation()
    net = net()

    if ckpt_path is not None:
        # Load the checkpoint if provided
        print(f"Loading checkpoint from {ckpt_path}")
        module = module.load_from_checkpoint(
            ckpt_path, net=net, representation=representation, strict=False
        )
    else:
        # Otherwise, create from random initialization
        print("Creating new model")
        module = module(net=net, representation=representation)

    return module, ckpt_path


@gin.configurable
def build_dev_datamodule(
    datamodule: L.LightningDataModule,
):
    datamodule = datamodule()
    return datamodule


@gin.configurable
def build_test_datamodule(
    datamodule: L.LightningDataModule,
):
    # datamodule = datamodule()
    # return datamodule
    raise NotImplementedError
