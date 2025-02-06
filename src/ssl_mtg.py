import gin

from pathlib import Path
from typing import List

from torch import nn
import pytorch_lightning as L

import nets
import modules


# Dummy class to catch the gin bindings
@gin.configurable
def build_module(
    representation: nn.Module | List,
    net: nn.Module,
    module: L.LightningModule,
    ckpt_path: Path = None,
):
    pass


def get_patch_size(representation: nn.Module) -> tuple:
    if representation == nets.MelSpectrogram:
        return (96, 4)
    elif representation == nets.CQT:
        return (144, 4)
    elif representation == nets.EnCodec:
        return (128, 5)
    elif representation == nn.ModuleList:
        raise NotImplementedError(f"Patch size for {representation} not implemented.")
    else:
        raise NotImplementedError(f"Patch size for {representation} not implemented.")


def get_model(config_file: Path, device: str = "cpu") -> L.LightningModule:
    """Returns the model from the provided config file.

    Args:
        config_file (Path): Path to the model config of a trained model.
        device (str): Device to use for the model. Defaults to "cpu".

    Output:
        module: The model from the provided config file.
        eps (float): Embeddings per second.
            e.g., torch.arange(T) / eps gives the timestamps of the embeddings.


    Module usage:

    Args:
        audio (torch.Tensor): 2D mono audio tensor (B, T'). Where B is
            the batch size and T' is the number of samples.
        layers (set): Set of layer indices to extract embeddings from.
            By default, it extracts embeddings from the last layer (logits).

    Output:
        torch.Tensor: Extracted embeddings. The output tensor has shape
            (L, B, T, C,) where L = len(layers), B is the batch size, T is
            the number of output timestamps, and C = embedding dimension.


    Example:

    >>> x = torch.randn(1, 16000 * 4).cpu()
    >>>
    >>> model, eps = get_model(config_file, device="cpu")
    >>>
    >>> embeddings = model.extract_embeddings(x, layers=(6))
    >>>
    >>> timestamps = torch.arange(embeddings.shape[2]) / eps



    >> NOTE: The model's embedding rate depends on the model's configuration.
        For example, the melspectrogram model has an embedding rate of 16ms.
        audio should be a sequence with a sample rate as inditacted in the
        config file and up to 30s.
    """

    # Init representation related variables
    sr, hop_len, patch_size = None, None, None

    config_file = Path(config_file)

    # Parse the gin config
    gin.parse_config_file(config_file, skip_unknown=True)
    gin.finalize()

    gin_config = gin.get_bindings(build_module)

    # get classes of interest
    net = gin_config["net"]
    representation = gin_config["representation"]
    module = gin_config["module"]

    # Make the checkpoint path relative to the config file location
    # insted of taking the absolute path
    ckpt_path = Path(gin_config["ckpt_path"])
    ckpt_path = config_file.parent / ckpt_path.name

    # Instantiate the classes
    net = net()

    # The model can feature one or multiple representations (multi-view models)
    if isinstance(representation, list):
        representation = nn.ModuleList([r() for r in representation])
    else:
        # In the single view case, extract the params from the rep class and get
        # a hardcoded patch size parameter (since it was not included in the gin config)
        patch_size = get_patch_size(representation)
        representation = representation(patch_size=patch_size)
        sr = representation.sr
        hop_len = representation.hop_len

    module = module.load_from_checkpoint(
        ckpt_path,
        net=net,
        representation=representation,
        strict=False,
    )

    module.to(device)
    module.eval()

    # In the multi-vew case, we only need the params of the rep used as input.
    # Get them from the module instance.
    if (
        hasattr(module, "patch_size")
        and hasattr(module, "sr")
        and hasattr(module, "hop_length")
    ):
        patch_size = module.patch_size
        sr = module.sr
        hop_len = module.hop_length

    # compute timestamps
    eps = sr / (hop_len * patch_size[1])

    return module, eps
