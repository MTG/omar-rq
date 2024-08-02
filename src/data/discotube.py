import gin.torch

from .data_utils import AudioDataModule, MultiViewAudioDataModule


# Create DiscoTube-specific data modules so that we can use them in the gin config

@gin.configurable
class DiscotubeAudioDataModule(AudioDataModule):
    """AudioDataModule for the Discogs dataset."""


@gin.configurable
class DiscotubeMultiViewAudioDataModule(MultiViewAudioDataModule):
    """MultiViewAudioDataModule for the Discogs dataset."""
