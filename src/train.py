from argparse import ArgumentParser
from pathlib import Path

import gin.torch
import pytorch_lightning as L
from pytorch_lightning import Trainer
from torch import nn

from modules import MODULES
from data import DATASETS
from archs import ARCHS


# Register all modules, datasets and architectures with gin
for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for arch_name, arch in ARCHS.items():
    gin.external_configurable(arch, arch_name)


@gin.configurable
def train(
    module: L.LightningModule,
    datamodule: L.LightningDataModule,
    arch: nn.Module,
) -> None:
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=1)

    module = module(arch=arch())
    datamodule = datamodule()
    trainer.fit(
        model=module,
        datamodule=datamodule,
    )

    trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-file", type=Path, default="cfg/config.gin")

    args = parser.parse_args()

    gin.parse_config_file(args.config_file)

    train()
