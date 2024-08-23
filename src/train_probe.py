from argparse import ArgumentParser
import yaml
from pathlib import Path
import traceback

import gin.torch
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from utils import gin_config_to_readable_dictionary
from callbacks import GinConfigSaverCallback
from probe.modules.module import MTTProbe
from probe.datamodules.mtt import MTTEmbeddingLoadingDataModule


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


# TODO fix seed
# TODO use a function
# TODO use gin
# TODO use wandb

if __name__ == "__main__":

    parser = ArgumentParser()
    # parser.add_argument(
    #     "embedding_dir",
    #     type=Path,
    #     help="Directory containing the embeddings extracted from the SSL model.",
    # )
    parser.add_argument(
        "ssl_model_id",
        type=str,
        help="ID of the SSL model used to extract the embeddings.",
    )
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the gin config file for training.",
    )
    # parser.add_argument(
    #     "test_config",  # TODO
    #     type=Path,
    #     help="Path to the config file for the dataset.",
    # )

    args = parser.parse_args()

    with open(args.train_config, "r") as in_f:
        test_config = yaml.safe_load(in_f)

    # We save the embeddings in <output_dir>/<model_id><dataset_name>/
    embedding_dir = (
        Path(test_config["output_dir"])
        / args.ssl_model_id
        / test_config["dataset_name"]
    )

    # Build the datamodule
    datamodule = MTTEmbeddingLoadingDataModule(
        embedding_dir,
        test_config["gt_path"],
        **test_config["splits"],
        **test_config["probe"]["data_loader"],
        **test_config["probe"]["embedding_processing"],
    )

    # Build the module # TODO: provide a net
    module = MTTProbe(**test_config["probe"]["model"])

    # Define the trainer
    trainer = Trainer(
        # accelerator=test_config["probe"]["model"]["accelerator"],
        # devices=test_config["probe"]["model"]["devices"],
        **test_config["probe"]["device"],
        # max_steps=10,
        log_every_n_steps=10,
        # limit_train_batches=2,
        # limit_val_batches=2,
        max_epochs=20,
        num_sanity_val_steps=0,
    )

    # Train the probe
    trainer.fit(model=module, datamodule=datamodule)

    # Test the best probe
    trainer.test(datamodule=datamodule, ckpt_path="best")

    # try:
    #     gin.parse_config_file(args.gin_config)
    #     train(ckpt_path=args.ckpt_path)
    # except Exception as e:
    #     print(e)
    #     traceback.print_exc()
