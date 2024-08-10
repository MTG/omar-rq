import os
from argparse import ArgumentParser
from pathlib import Path
import traceback

import gin.torch
import pytorch_lightning as L


from data import DATASETS
from modules import MODULES
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
        "config_path",
        type=Path,
        help="Path to the model config of a trained model.",
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

        # Read the test config as a string
        with open(args.test_config, "r") as f:
            test_config = f.read()
        # Append the model config path to the test config
        test_config += (
            f"\n\n# Model config\ninclude '{os.path.abspath(args.config_path)}'\n"
        )

        # Convert the config string to a gin config
        gin.parse_config(test_config)

        # Build the module and datamodule
        module, _ = build_module()
        # datamodule = build_test_datamodule()

        gin.finalize()

        evaluate(module)

    except Exception:
        traceback.print_exc()

    print("Evaluation completed successfully.")
