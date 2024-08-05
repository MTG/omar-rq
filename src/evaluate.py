from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import traceback

import gin.torch
import pytorch_lightning as L
from pytorch_lightning import Trainer

from torch import nn

from data import DATASETS
from modules import MODULES, MaskingModel
from nets import NETS
from utils import gin_config_to_readable_dictionary, build_module

# Register all modules, datasets and networs with gin
for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)


# @gin.configurable
def evaluate(
    module: L.LightningModule,
    # datamodule: L.LightningDataModule,
    # params: dict,
) -> None:
    """Evaluate a trained model using the given module, datamodule and netitecture. Loads the
    weights from the checkpoint and evaluates the model on the test set."""

    # disable randomness, dropout, etc...
    module.eval()


if __name__ == "__main__":
    parser = ArgumentParser("Evaluate SSL models using gin config")
    parser.add_argument(
        "ckpt_dir",
        type=Path,
        help="Path to the directory containing the model checkpoint and its training gin config.",
    )
    parser.add_argument(
        "--test-config",
        type=Path,
        default="cfg/test_config.gin",
        help="Path to the gin config file for testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to the directory where the evaluation results will be stored.",
    )

    args = parser.parse_args()

    try:

        # Load the checkpoint and the gin config
        train_config_path = list(args.ckpt_dir.glob("**/*.gin"))[0]
        print(f"Train config path: {train_config_path}")

        # Read the training config to get the model config path and the
        # model checkpoint path
        # We store these information in the 2nd and 5th lines in the train gin config
        with open(train_config_path, "r") as f:
            train_config_str = f.read()
        train_config_str = train_config_str.split("\n")
        ckpt_path = Path(train_config_str[1].split(" = ")[1].replace("'", ""))
        model_config_path = train_config_str[4].replace("include ", "")
        print(f"Model checkpoint path: {ckpt_path}")
        print(f"Model config path: {model_config_path}")

        # Read the test config as a string
        with open(args.test_config, "r") as f:
            test_config_str = f.read()
        # Append the model config to the test config
        test_config_str += f"\n\n# Model config\ninclude {model_config_path}"

        # Convert the config string to a gin config
        gin.parse_config(test_config_str)

        # Build the module and datamodule
        module = build_module(ckpt_path=ckpt_path)
        # datamodule = build_test_datamodule()

        gin.finalize()

        evaluate(module)

    except Exception:
        traceback.print_exc()

    print("Evaluation completed successfully.")
