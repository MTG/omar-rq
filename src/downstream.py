""" This script is the main training and evaluation script for training a probe on 
a downstream task. It expects pre-extracted embeddings from a self-supervised model 
and a config file for the downstream task. The config file should contain the details
of the dataset and the parameters of the probe. The script will train the probe on the
embeddings and evaluate it on the corresponding downstream task.

# TODO fix seed
# TODO load best ckpt for test
"""

import traceback
from pathlib import Path
import argparse

import gin.torch
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from probe.data.harmonix import HarmonixEmbeddingLoadingDataModule
from probe.modules import StructureClassProbe
from utils import gin_config_to_readable_dictionary
from probe.modules import SequenceMultiLabelClassificationProbe
from probe.data import MTTEmbeddingLoadingDataModule


@gin.configurable
def build_module_and_datamodule(
    ssl_model_id: str, dataset_name: str, embeddings_dir: Path
):
    # We saved the embeddings in <output_dir>/<model_id>/<dataset_name>/
    embeddings_dir = Path(embeddings_dir) / ssl_model_id / dataset_name
    if dataset_name == "magnatagatune":

        # Build the datamodule
        datamodule = MTTEmbeddingLoadingDataModule(
            embeddings_dir,
        )

        # Get the number of features from the dataloader
        in_features = datamodule.embedding_dimension

        # Build the DataModule
        module = SequenceMultiLabelClassificationProbe(
            in_features=in_features,
        )
    elif dataset_name == "harmonix":
        # Build the datamodule
        datamodule = HarmonixEmbeddingLoadingDataModule(embeddings_dir)

        # Get the number of features from the dataloader
        in_features = datamodule.embedding_dimension
        class_weights = datamodule.class_weights

        # Build the DataModule
        module = StructureClassProbe(
            in_features=in_features,
            class_weights=class_weights,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return module, datamodule


@gin.configurable
def train_probe(
    module: L.LightningModule,
    datamodule: L.LightningDataModule,
    ssl_model_id: str,
    wandb_params: dict,
    train_params: dict,
):

    # Define the logger
    wandb_logger = WandbLogger(**wandb_params)

    # Get the gin config as a dictionary and log it to wandb
    _gin_config_dict = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)
    wandb_logger.log_hyperparams({"ssl_model_id": ssl_model_id, **_gin_config_dict})

    # Define the trainer
    trainer = Trainer(logger=wandb_logger, **train_params)

    # Train the probe
    trainer.fit(model=module, datamodule=datamodule)

    # Test the best probe
    trainer.test(datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "ssl_model_id",
        type=str,
        help="ID of the SSL model used to extract the embeddings.",
    )
    parser.add_argument(
        "downstream_config",
        type=Path,
        help="Path to the config file for the downstream task.",
    )

    args = parser.parse_args()

    try:
        # Load the downstream config
        gin.parse_config_file(args.downstream_config, skip_unknown=True)

        # Build the module and datamodule
        module, datamodule = build_module_and_datamodule(args.ssl_model_id)

        # Train the probe
        train_probe(module, datamodule, args.ssl_model_id)

    except Exception:
        traceback.print_exc()
