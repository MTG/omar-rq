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
        return (96, 4)
    elif representation == nets.EnCodec:
        return (144, 4)
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
        audio (torch.Tensor): 1D audio tensor.
        layers (set): Set of layer indices to extract embeddings from.
            By default, it extracts embeddings from the last layer.

    Output:
        torch.Tensor: Extracted embeddings.
            Even in the case of aggregation or single layer embeddings,
            the output tensor will have the same shape (L, B, T, C,)
            where L = len(layer), B is the number of chunks
            T is the number of melspec frames the model can accomodate
            C = model output dimension. No aggregation is applied.
            audio: torch.Tensor of shape (batch_size, num_samples)
            embeddings: torch.Tensor of shape (lbatch_size, timestamps, embedding_size)


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

    # Select the correct patch size
    patch_size = get_patch_size(representation)

    # Instantiate the classes
    net = net()
    representation = representation(patch_size=patch_size)
    module = module.load_from_checkpoint(
        ckpt_path,
        net=net,
        representation=representation,
        strict=False,
    )

    module.to(device)
    module.eval()

    # compute timestamps
    eps = representation.sr / (representation.hop_len * patch_size[1])

    return module, eps
