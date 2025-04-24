import gin

from pathlib import Path
from typing import List

from torch import nn
import pytorch_lightning as L

import nets
import modules
from utils import build_module


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


def get_model(
    config_file: Path | str,
    device: str = "cpu",
    encodec_weights_path: str | None = None,
) -> tuple[L.LightningModule, float]:
    """Returns the model from the provided config file.

    Args:
        config_file (Path): Path to the model config of a trained model.
        device (str): Device to use for the model. Defaults to "cpu".
        encodec_weights_path (str): Path to the EnCodec weights. When set, it will
            override the value in the config file. Note that it can be a local path
            or or a Hugging Face model ID. This parameter only affects to models
            that use EnCodec as representation. Defaults to None.

            https://huggingface.co/docs/transformers/en/model_doc/encodec

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

    # Read previous config bindings
    bindings = []
    cfg_str = gin.config_str()

    # If these are not empty, this model is part of a larger setup
    # Do not finish the configuration now
    finalize_config = False
    if cfg_str != "":
        finalize_config = True

    lines = cfg_str.split("\n")
    bindings.extend(lines)

    if encodec_weights_path is not None:
        bindings.append(f"nets.encodec.EnCodec.weights_path = '{encodec_weights_path}'")
        bindings.append("nets.encodec.EnCodec.stats_path = None")

    # Parse the gin config
    gin.parse_config_files_and_bindings(
        [config_file],
        bindings,
        skip_unknown=True,
        finalize_config=finalize_config,
    )

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
        representation = nn.ModuleList([r().to(device) for r in representation])
    else:
        # In the single view case, extract the params from the rep class and get
        # a hardcoded patch size parameter (since it was not included in the gin config)
        patch_size = get_patch_size(representation)
        representation = representation(patch_size=patch_size).to(device)
        sr = representation.sr
        hop_len = representation.hop_len

    module = module.load_from_checkpoint(
        ckpt_path,
        net=net,
        representation=representation,
        strict=False,
        map_location=device,
    )

    # Set the model to eval mode
    module.eval()

    # Move the model to the device
    module.to(device)

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
