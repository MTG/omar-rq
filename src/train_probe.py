from argparse import ArgumentParser
from pathlib import Path
import traceback

import gin.torch
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from utils import gin_config_to_readable_dictionary
from callbacks import GinConfigSaverCallback
from fine_tune.modules.module import MTTProbe
from fine_tune.datamodules.mtt import MTTEmbeddingLoadingDataModule


# # @gin.configurable
# def train(
#     module: L.LightningModule,
#     datamodule: L.LightningDataModule,
#     params: dict,
#     wandb_params: dict,
#     config_path: Path,
#     ckpt_path: Path = None,
# ) -> None:
#     """Train a model using the given module, datamodule and netitecture"""

#     # get the lightning wandb logger wrapper and log the config
#     gin_config_dict = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)
#     wandb_logger = WandbLogger(**wandb_params)
#     wandb_logger.log_hyperparams(gin_config_dict)

#     # create callbacks
#     callbacks = [GinConfigSaverCallback(config_path)]

#     # create the trainer
#     trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **params)

#     # TODO monitor the best model
#     # fit the model
#     trainer.fit(model=module, datamodule=datamodule, ckpt_path=ckpt_path)

#     # TODO: Choose the best model
#     # Test the model # TODO: logger for testing?
#     trainer.test(model=module, datamodule=datamodule)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the gin config file for training.",
    )
    parser.add_argument(
        "test_config",
        type=Path,
        help="Path to the config file for the embeddings.",
    )

    args = parser.parse_args()

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        # max_steps=10,
        limit_train_batches=5,
        limit_val_batches=10,
        max_epochs=2,
        # strategy="ddp_find_unused_parameters_true",
        # precision="bf16-mixed",
        num_sanity_val_steps=0,
    )

    datamodule = MTTEmbeddingLoadingDataModule(
        Path("/gpfs/projects/upf97/embeddings/cy1uafdv/magnatagatune/"),
        Path(
            "/gpfs/projects/upf97/downstream_datasets/magnatagatune/metadata/annotations_final.csv"
        ),
        Path("/home/upf/upf455198/ssl-mtg/data/magnatagatune/train.txt"),
        Path("/home/upf/upf455198/ssl-mtg/data/magnatagatune/validation.txt"),
        Path("/home/upf/upf455198/ssl-mtg/data/magnatagatune/test.txt"),
        64,
        0,
        "mean",
        "chunk",
        "mean",
    )
    module = MTTProbe(None, 768, 188)
    # len(datamodule.train_dataset.labels[0])

    trainer.fit(model=module, datamodule=datamodule)

    trainer.test(model=module, datamodule=datamodule)

    # try:
    #     gin.parse_config_file(args.gin_config)
    #     train(ckpt_path=args.ckpt_path)
    # except Exception as e:
    #     print(e)
    #     traceback.print_exc()
