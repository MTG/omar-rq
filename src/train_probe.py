from argparse import ArgumentParser
import yaml
from pathlib import Path
import traceback

import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from probe.modules.module import MTTProbe
from probe.datamodules.mtt import MTTEmbeddingLoadingDataModule

# TODO fix seed
# TODO use a function
# TODO use gin

if __name__ == "__main__":

    parser = ArgumentParser()
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

    args = parser.parse_args()

    with open(args.train_config, "r") as in_f:
        test_config = yaml.safe_load(in_f)

    try:

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

        # Define the logger
        wandb_logger = WandbLogger(**test_config["probe"]["wandb_params"])
        wandb_logger.log_hyperparams(test_config)

        # Define the trainer
        trainer = Trainer(logger=wandb_logger, **test_config["probe"]["trainer"])

        # Train the probe
        trainer.fit(model=module, datamodule=datamodule)

        # Test the best probe
        trainer.test(datamodule=datamodule, ckpt_path="best")

    except Exception:
        traceback.print_exc()
