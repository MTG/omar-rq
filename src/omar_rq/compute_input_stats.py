import traceback
import json
from argparse import ArgumentParser
from pathlib import Path
from collections import defaultdict

import gin.torch
import torch
import pytorch_lightning as L
from tqdm import tqdm

from .data import DATASETS
from .modules import MODULES
from .nets import NETS
from .utils import build_module, build_dev_datamodule


for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)

for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)


@gin.configurable
def compute_input_stats(
    module: L.LightningModule,
    datamodule: L.LightningDataModule,
) -> None:
    """Train a model using the given module, datamodule and netitecture"""

    module.cuda()
    module.eval()
    module.freeze()
    datamodule.setup("fit")

    # manually predict with module for the entire datamodule
    n_batches = 0
    acc_mean = defaultdict(float)
    acc_std = defaultdict(float)
    acc_mean_dims = defaultdict(torch.Tensor)
    acc_std_dims = defaultdict(torch.Tensor)

    for batch in tqdm(datamodule.train_dataloader()):
        x = batch[0].cuda().float()

        for rep in module.representation:
            rep_name = type(rep).__name__
            rep_batch = rep(x)

            mean = torch.mean(torch.mean(rep_batch, dim=(-2, -1)), dim=0).item()
            std = torch.mean(torch.std(rep_batch, dim=(-2, -1)), dim=0).item()

            acc_mean[rep_name] += mean
            acc_std[rep_name] += std

            mean_dims = torch.mean(torch.mean(rep_batch, dim=-1), dim=0)
            std_dims = torch.mean(torch.std(rep_batch, dim=-1), dim=0)

            if not len(acc_mean_dims[rep_name]):
                acc_mean_dims[rep_name] = mean_dims.cpu()
                acc_std_dims[rep_name] = std_dims.cpu()
            else:
                acc_mean_dims[rep_name] += mean_dims.cpu()
                acc_std_dims[rep_name] += std_dims.cpu()

        n_batches += 1

    for rep_name in acc_mean.keys():
        acc_mean[rep_name] /= n_batches
        acc_std[rep_name] /= n_batches
        acc_mean_dims[rep_name] /= n_batches
        acc_std_dims[rep_name] /= n_batches

        # acc_mean_dims[rep_name] = acc_mean_dims[rep_name].tolist()
        # acc_std_dims[rep_name] = acc_std_dims[rep_name].tolist()

        # put in dict and log as a json
        stats = {
            "mean": acc_mean[rep_name],
            "std": acc_std[rep_name],
            # "mean_dims": acc_mean_dims,
            # "std_dims": acc_std_dims,
        }

        print(f"Input stats for {rep_name}: {stats}")

        with open(f"input_stats_{rep_name}.json", "w") as f:
            json.dump(stats, f)


if __name__ == "__main__":
    parser = ArgumentParser("Train SSL models using gin config")
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the gin config file for training.",
    )

    args = parser.parse_args()

    try:
        gin.parse_config_file(args.train_config, skip_unknown=True)

        module, _ = build_module()
        datamodule = build_dev_datamodule()

        gin.finalize()

        compute_input_stats(module, datamodule)

    except Exception:
        traceback.print_exc()
