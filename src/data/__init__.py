from .mnist import MNISTDataModule
from .discotube import DiscotubeAudioDataModule, DiscotubeMultiViewAudioDataModule

DATASETS = {
    "mnist": MNISTDataModule,
    "discotube": DiscotubeAudioDataModule,
    "discotube_multiview": DiscotubeMultiViewAudioDataModule,
}
