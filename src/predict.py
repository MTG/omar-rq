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
from eval.dataset import AudioEmbeddingDataset

# Register all modules, datasets and networs with gin
for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)


@torch.no_grad()
def evaluate(
    module: L.LightningModule,
    dataloader: DataLoader,
    output_dir: Path,
    # **params: dict,
) -> None:
    """Evaluate a trained model using the given module, datamodule and netitecture. Loads the
    weights from the checkpoint and evaluates the model on the test set."""

    precision = gin.query_parameter("train.precision")

    trainer = L.Trainer(precision=precision)

    # Create the output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # disable randomness, dropout, melspec aug..
    module.eval()
    module = module.to("cuda:0")  # TODO device?

    for audio, audio_path in dataloader:
        audio = audio.squeeze(0).to("cuda:0")  # (Ta, )
        audio_name = audio_path.stem
        output_path = output_dir / f"{audio_name}.npy"
        # Get the embeddings
        embeddings = module.extract_embeddings(audio)  # (L, Tc, C)
        embeddings = embeddings.cpu().numpy()
        # Save the embeddings
        np.save(output_path, embeddings)  # TODO: output precision?
        break


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
        type=Path,
        default="./embeddings/",
        help="Path to the directory where the evaluation results will be stored.",
    )
    parser.add_argument(
        "--nworkers", type=int, default=10, help="Number of workers for the dataloader."
    )

    args = parser.parse_args()

    try:

        # TODO: parse multiple gin configs. Or use YAML for test config
        # Convert the config string to a gin config
        gin.parse_config_file(args.train_config, skip_unknown=True)

        # Build the module and load the weights
        module, _ = build_module()

        gin.finalize()

        # Load the test config
        with open(args.test_config, "r") as f:
            test_config = yaml.safe_load(f)

        # Get the dataloader
        dataset = AudioEmbeddingDataset(**test_config)  # TODO: gin?
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.nworkers,
            collate_fn=dataset.collate_fn,
        )  # TODO , pin_memory=True?

        evaluate(module, dataloader, args.output_dir)

    except Exception:
        traceback.print_exc()

    print("Embedding extraction completed successfully.")
