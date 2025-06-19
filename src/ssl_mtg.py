import gin

from pathlib import Path

import torch
import pytorch_lightning as L
from torch import nn
from huggingface_hub import hf_hub_download

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
    model_id: str | None = None,
    config_file: Path | str | None = None,
    device: str = "cpu",
    quantization_targets: bool = False,
) -> tuple[L.LightningModule, float]:
    """Returns the model from the provided config file.

    Args:
        model_id (str): Hugging Face model ID. If provided, it will download the
            model_id config and checkpoint.
        config_file (Path): Path to the model config of a trained model.
        device (str): Device to use for the model. Defaults to "cpu".
        quantization_targets (bool): If True, it will create the quantization
            targets for SSL pre-training of the model. Defaults to False.

    Output:
        module: The model from the provided config file.


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
    >>> model = get_model(config_file, device="cpu")
    >>>
    >>> embeddings = model.extract_embeddings(x, layers=(6))
    >>>
    >>> timestamps = torch.arange(embeddings.shape[2]) / model.eps



    >> NOTE: The model's embedding rate depends on the model's configuration.
        For example, the melspectrogram model has an embedding rate of 16ms.
        audio should be a sequence with a sample rate as inditacted in the
        config file and up to 30s.
    """

    # Init representation related variables
    sr, hop_len, patch_size, ckpt_path = None, None, None, None

    if model_id:
        config_file = hf_hub_download(
            repo_id=f"mtg-upf/{model_id}", filename="config.gin"
        )
        ckpt_path = hf_hub_download(
            repo_id=f"mtg-upf/{model_id}", filename="model.ckpt"
        )

    # When no config file is provided, it is assumed that an external
    # gin-config file with all the required fileds has already been parsed.
    # Don't try to moddify the gin configuration nor load a checkpoint.
    if config_file != "":
        # Parse the gin config
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                config_files=[str(config_file)], bindings=None, skip_unknown=True
            )

    gin_config = gin.get_bindings(build_module)

    # get classes of interest
    net = gin_config["net"]
    representation = gin_config["representation"]
    module = gin_config["module"]

    # Instantiate the classes
    net = net()

    # The model can feature one or multiple representations (multi-view models)
    if isinstance(representation, list) and quantization_targets is False:
        representation = None
    elif isinstance(representation, list) and quantization_targets is True:
        representation = nn.ModuleList([r() for r in representation])
    else:
        # In the single view case, extract the params from the rep class and get
        # a hardcoded patch size parameter (since it was not included in the gin config)
        patch_size = get_patch_size(representation)
        representation = representation(patch_size=patch_size).to(device)
        sr = representation.sr
        hop_len = representation.hop_len

    module = module(
        net=net,
        representation=representation,
        quantization_targets=quantization_targets,
    )

    if config_file != "":
        assert model_id is None, (
            "When a config file is provided, mudel_id should not be configured."
        )

        # Make the checkpoint path relative to the config file location
        # insted of taking the absolute path
        if not ckpt_path:
            ckpt_path = Path(gin_config["ckpt_path"])
            ckpt_path = Path(config_file).parent / ckpt_path.name

        # Load the checkpoint weights it exists
        state_dict = torch.load(ckpt_path, map_location=device)
        for key in ["net", "embedding_layer"]:
            net_weigths = {
                k[len(key) + 1 :]: v for k, v in state_dict.items() if k.startswith(key)
            }
            getattr(module, key).load_state_dict(net_weigths, strict=True)

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

    # compute embeddings per second
    eps = sr / (hop_len * patch_size[1])
    module.eps = eps

    return module
