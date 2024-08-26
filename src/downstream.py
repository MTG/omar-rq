""" This script is the main training and evaluation script for training a probe on 
a downstream task. It expects pre-extracted embeddings from a self-supervised model 
and a config file for the downstream task. The config file should contain the details
of the dataset and the parameters of the probe. The script will train the probe on the
embeddings and evaluate it on the corresponding downstream task.

# TODO fix seed
# TODO use gin for the probe
# TODO use gin for the dataset
# TODO load best ckpt for test
# TODO why does it take 5 minutes to start training?
"""

import yaml
import traceback
from pathlib import Path
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "ssl_model_id",
        type=str,
        help="ID of the SSL model used to extract the embeddings.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the config file for the downstream task.",
    )

    args = parser.parse_args()

    with open(args.config, "r") as in_f:
        config = yaml.safe_load(in_f)
    dataset_name = config["dataset_name"]

    try:

        if dataset_name == "magnatagatune":

            from probe.modules import SequenceMultiLabelClassificationProbe
            from probe.data import MTTEmbeddingLoadingDataModule

            # We saved the embeddings in <output_dir>/<model_id>/<dataset_name>/
            embeddings_dir = (
                Path(config["embeddings_dir"]) / args.ssl_model_id / dataset_name
            )

            # Build the datamodule
            datamodule = MTTEmbeddingLoadingDataModule(
                embeddings_dir,
                config["gt_path"],
                **config["splits"],
                **config["probe"]["data_loader"],
                **config["probe"]["embedding_processing"],
            )

            # TODO: all with gin
            model_dict = config["probe"]["model"]
            # Get the embedding dimension from the dataloader
            model_dict["in_features"] = datamodule.embedding_dimension
            # Add the labels to the model dict for plotting confusion matrix
            model_dict["labels"] = config["labels"]
            model_dict["plot_dir"] = (
                Path(config["evaluation_dir"]) / args.ssl_model_id / dataset_name
            )

            # TODO: provide a net with gin
            # TODO: write the metrics to the eval dir too
            # Build the module
            module = SequenceMultiLabelClassificationProbe(**model_dict)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Define the logger
        wandb_logger = WandbLogger(**config["probe"]["wandb_params"])
        wandb_logger.log_hyperparams(config)

        # Define the trainer
        trainer = Trainer(logger=wandb_logger, **config["probe"]["trainer"])

        # Train the probe
        trainer.fit(model=module, datamodule=datamodule)

        # Test the best probe
        trainer.test(datamodule=datamodule, ckpt_path="best")

    except Exception:
        traceback.print_exc()
