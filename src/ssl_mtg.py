import gin

from pathlib import Path


from modules import MODULES
from nets import NETS


def get_model(config_file: Path, device: str = "cpu"):
    """Returns the model from the provided config file.

    Args:

    config_file (Path): Path to the model config of a trained model.
    device (str): Device to use for the model. Defaults to "cpu".

    Returns:

    module: The model from the provided config file.


    Module usage:

    embeddings = model.extract_embeddings(audio)

        audio: torch.Tensor of shape (batch_size, num_samples)
        embeddings: torch.Tensor of shape (batch_size, 1, timestamps, embedding_size)

    The model's embedding rate depends on the model's configuration.
    For the melspectrogram model, the embedding rate is 16ms.
    samples should be a sequence of audio samples, with a sample as inditacted in the
    config file and up to 30s.

    Example:

    >>> model = get_model(config_file, "cpu")

    >>> x = torch.randn(1, 16000 * 4).cpu()
    >>> embeddings = model.extract_embeddings(x)
    """

    config_file = Path(config_file)

    # Parse the gin configs.
    gin.parse_config_file(config_file, skip_unknown=True)
    gin.finalize()

    # get classes of interest
    net = NETS["conformer"]
    representation = NETS["melspectrogram"]
    module = MODULES["maskingmodel"]

    # workaround to get the checkpoint relative to the config file
    ckpt_path = list(config_file.parent.glob("*.ckpt"))

    # just in case
    assert (
        len(ckpt_path) == 1
    ), f"Found multiple or no checkpoint files in {config_file.parent}."

    ckpt_path = ckpt_path[0]

    # instantiate the classes
    net = net()
    representation = representation(patch_size=(96, 4))
    module = module.load_from_checkpoint(
        ckpt_path,
        net=net,
        representation=representation,
        strict=False,
    )

    module.to(device)
    module.eval()

    return module

    # if __name__ == "__main__":
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument(
    #         "config_file",
    #         type=Path,
    #         help="Path to the model config of a trained model.",
    #     )

    # args = parser.parse_args()
