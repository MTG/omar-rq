from argparse import ArgumentParser
import yaml
from pathlib import Path
import traceback

import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from probe.magnatagatune import MTTProbe, MTTEmbeddingLoadingDataModule

# TODO fix seed
# TODO use gin

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

    try:

        # We save the embeddings in <output_dir>/<model_id><dataset_name>/
        embedding_dir = (
            Path(config["output_dir"]) / args.ssl_model_id / config["dataset_name"]
        )

        # Build the datamodule
        datamodule = MTTEmbeddingLoadingDataModule(
            embedding_dir,
            config["gt_path"],
            **config["splits"],
            **config["probe"]["data_loader"],
            **config["probe"]["embedding_processing"],
        )

        # Build the module # TODO: provide a net with gin
        module = MTTProbe(**config["probe"]["model"])

        # Define the logger
        wandb_logger = WandbLogger(**config["probe"]["wandb_params"])
        wandb_logger.log_hyperparams(config)

        # Define the trainer
        trainer = Trainer(logger=wandb_logger, **config["probe"]["trainer"])

        # Train the probe
        trainer.fit(model=module, datamodule=datamodule)

        # Test the best probe # TODO: how does it determine the best?
        trainer.test(datamodule=datamodule, ckpt_path="best")

    except Exception:
        traceback.print_exc()
