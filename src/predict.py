from argparse import ArgumentParser
from pathlib import Path
import traceback
from typing import List

import gin.torch
import pytorch_lightning as L

from data import DATASETS
from modules import MODULES
from nets import NETS
from utils import build_module
from prediction.callbacks import EmbeddingWriter
from prediction.dataset import AudioEmbeddingDataModule

# Register all modules, datasets and networs with gin
for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)


def define_embeddings_dir(ckpt_path: Path, dataset_name: str, root_output_dir: str):
    """We use the following structure for the embeddings directory:
        root_output_dir/ssl_model_id/dataset_name/

    Inside dataset_name, we have the following structure:
        dataset_name/audio_name[:3]/audio_name.pt"""

    ssl_model_id = Path(ckpt_path).parent.parent.name
    return Path(root_output_dir) / ssl_model_id / dataset_name


@gin.configurable
def predict(
    ckpt_path: Path, embeddings_dir: Path, device_dict: dict, embedding_layer: List[int]
):
    """Wrapper function. Basically overrides some train parameters."""

    # Set the output directory with model id and dataset name
    embeddings_dir = define_embeddings_dir(ckpt_path, **embeddings_dir)

    train(
        embeddings_dir=embeddings_dir,
        device_dict=device_dict,
        embedding_layer=embedding_layer,
    )


@gin.configurable
def train(
    embeddings_dir: Path,
    params: dict,
    device_dict: dict,
    embedding_layer: List[int],
    wandb_params=None,
):
    """The name is train, but we are actually predicting the embeddings for
    the downstream task. This is done to leverage the gin config of the training.

    NOTE: you have to keep wandb_params argument of this function. Otherwise,
    Gin can not use the training config."""

    # Add the callback to write the embeddings
    callbacks = [EmbeddingWriter(embeddings_dir)]

    # Overwride the device params of the training with the prediction params
    device_dict = {**params, **device_dict}

    # Create the trainer first
    trainer = L.Trainer(callbacks=callbacks, **device_dict)

    # Build the module and load the weights
    module, _ = build_module(trainer=trainer)

    # Set the embedding layer
    module.embedding_layer = embedding_layer

    # Get the data module
    data_module = AudioEmbeddingDataModule()

    # Extract embeddings with the model
    trainer.predict(
        module,
        data_module,
    )


if __name__ == "__main__":
    parser = ArgumentParser("Evaluate SSL models using gin configs.")
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the model config of a trained model.",
    )
    parser.add_argument(
        "predict_config",
        type=Path,
        help="Path to the config file of the downstream task's dataset.",
    )

    args = parser.parse_args()

    try:

        # Parse the gin configs.
        for config_file in [args.train_config, args.predict_config]:
            gin.parse_config_file(config_file, skip_unknown=True)

        gin.finalize()

        # Get the ckpt path from the gin config
        ckpt_path = Path(gin.query_parameter("build_module.ckpt_path"))

        predict(ckpt_path=ckpt_path)

        print("Embedding extraction completed successfully!")

    except Exception:
        traceback.print_exc()
