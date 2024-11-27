"""Evaluate SSL models using gin configs. This script is used to extract
embeddings from a pre-trained SSL model for downstream tasks. The embeddings are
saved in the embeddings_dir specified in the downstream task's gin config.
By default, the embeddings won't be aggregated and will be saved as they are.
(L, B, T, C,)
    where L = len(layers),
    B = number of audio chunks
    T = number of melspec frames the model can accomodate
    C = model output dimension
"""

import traceback
import argparse
from pathlib import Path
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


@gin.configurable
def predict(
    ckpt_path: Path,
    dataset_name: str,
    embeddings_dir: Path,
    device_dict: dict,
    embedding_layer: List[int],
    overlap_ratio: float,
):
    """Wrapper function. Basically overrides some train parameters."""

    # We use the following structure for the embeddings directory:
    # root_output_dir/ssl_model_id/dataset_name/ Inside dataset_name,
    # we have the following structure: dataset_name/audio_name[:3]/audio_name.pt
    ssl_model_id = Path(ckpt_path).parent.parent.name
    embeddings_dir = Path(embeddings_dir) / ssl_model_id / dataset_name

    train(
        embeddings_dir=embeddings_dir,
        device_dict=device_dict,
        embedding_layer=embedding_layer,
        overlap_ratio=overlap_ratio,
    )


@gin.configurable
def train(
    embeddings_dir: Path,
    params: dict,
    device_dict: dict,
    embedding_layer: List[int],
    overlap_ratio: float,
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
    module.downstream_embedding_layer = embedding_layer

    # Set the overlap ratio
    module.overlap_ratio = overlap_ratio

    # Get the data module
    data_module = AudioEmbeddingDataModule(
        orig_freq=module.orig_freq,
        new_freq=module.new_freq,
        mono=module.mono,
    )

    # Extract embeddings with the model
    trainer.predict(
        module,
        data_module,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
