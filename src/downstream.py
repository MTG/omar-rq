"""This script is the main training and evaluation script for training a probe on
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
import wandb
from bayes_opt import BayesianOptimization
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import gin_config_to_readable_dictionary
from probe.modules import (
    SequenceMultiLabelClassificationProbe,
    SequenceMultiClassClassificationProbe,
)
from probe.data import (
    MTTEmbeddingLoadingDataModule,
    NSynthPitchEmbeddingLoadingDataModule,
)
from cosineannealingscheduler import CosineAnnealingCallback


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

    if dataset_name == "nsynth":
        # Build the datamodule
        datamodule = NSynthPitchEmbeddingLoadingDataModule(
            embeddings_dir,
        )

        # Get the number of features from the dataloader
        in_features = datamodule.embedding_dimension

        # Build the DataModule
        module = SequenceMultiClassClassificationProbe(
            in_features=in_features,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return module, datamodule


@gin.configurable
def optimize_probe(
    datamodule: L.LightningDataModule,
    ssl_model_id: str,
    bound_conditions: dict,
    init_points: int,
    n_iter: int,
    seed: int,
    optim_process: bool,
):
    if not optim_process:
        return False

    def train_probe_proxy(**params):
        # round discrete values
        # kwargs["num_layers"] = round(kwargs["num_layers"])
        params["hidden_size"] = round(params["hidden_size"])

        module = SequenceMultiLabelClassificationProbe(
            in_features=datamodule.embedding_dimension, **params
        )

        return train_probe(
            module=module,
            datamodule=datamodule,
            ssl_model_id=ssl_model_id,
            optim_process=True,
            enable_progress_bar=True,
            enable_model_summary=False,
            probe_params=params,
        )

    optimizer = BayesianOptimization(
        f=train_probe_proxy,
        pbounds=bound_conditions,
        random_state=seed,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    return True


@gin.configurable
def train_probe(
    module: L.LightningModule,
    datamodule: L.LightningDataModule,
    ssl_model_id: str,
    wandb_params: dict,
    train_params: dict,
    monitor: str,
    monitor_mode: str,
    optim_process: bool = False,
    probe_params: dict | None = None,
    **kwargs,
):
    # Define the logger
    wandb_logger = WandbLogger(**wandb_params)

    # Get the gin config as a dictionary and log it to wandb
    _gin_config_dict = gin_config_to_readable_dictionary(gin.config._OPERATIVE_CONFIG)

    if probe_params:
        _gin_config_dict.update({"probe_parameters": probe_params})

    wandb_logger.log_hyperparams({"ssl_model_id": ssl_model_id, **_gin_config_dict})

    # replace train_params with the opmized values
    train_params.update(kwargs)

    # Create the callbacks
    cosine_annealing_callback = CosineAnnealingCallback(
        total_steps=train_params["max_steps"]
    )
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=monitor_mode,
    )
    callbacks = [cosine_annealing_callback, checkpoint]

    # Define the trainer
    trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **train_params)

    # Train the probe
    trainer.fit(model=module, datamodule=datamodule)

    # Test the best probe
    trainer.test(datamodule=datamodule, ckpt_path="best")

    if optim_process:
        wandb.finish()

        return module.best_val_metric[monitor]


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

        optimized_probe = optimize_probe(datamodule, args.ssl_model_id)

        # Train the probe with the default values if there was no optimization process
        if not optimized_probe:
            train_probe(module, datamodule, args.ssl_model_id)

    except Exception:
        traceback.print_exc()
