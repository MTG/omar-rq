import os
import yaml
from argparse import ArgumentParser
from pathlib import Path
import traceback

import numpy as np

import torch
import gin.torch
import pytorch_lightning as L
from torch.utils.data import DataLoader


from data import DATASETS
from modules import MODULES
from nets import NETS
from utils import gin_config_to_readable_dictionary, build_module
from callbacks import EmbeddingWriter
from eval.dataset import AudioEmbeddingDataModule

# Register all modules, datasets and networs with gin
for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)


if __name__ == "__main__":
    parser = ArgumentParser("Evaluate SSL models using gin config")
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the model config of a trained model.",
    )
    parser.add_argument(
        "test_config",
        type=Path,
        help="Path to the config file of the downstream task's dataset.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default="./embeddings/",
        help="Path to the directory where the evaluation results will be stored.",
    )

    args = parser.parse_args()

    try:

        # Load the test config
        with open(args.test_config, "r") as f:
            test_config = yaml.safe_load(f)

        # TODO: parse multiple gin configs. Or use YAML for test config
        # Convert the config string to a gin config
        gin.parse_config_file(args.train_config, skip_unknown=True)

        callbacks = [EmbeddingWriter(args.output_dir)]

        # Get the precision for model initialization
        trainer = L.Trainer(
            precision=test_config["model"]["precision"], callbacks=callbacks
        )

        # Build the module and load the weights
        module, _ = build_module(trainer=trainer)

        gin.finalize()

        # Get the data module
        data_module = AudioEmbeddingDataModule(**test_config["audio"])

        trainer.predict(
            module,
            data_module,
            # **params,
        )

        print("Embedding extraction completed successfully.")

    except Exception:
        traceback.print_exc()
