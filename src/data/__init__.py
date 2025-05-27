from .mnist import MNISTDataModule
from .discotube import DiscotubeAudioDataModule, DiscotubeMultiViewAudioDataModule
from .discotube_text_audio import DiscotubeTextAudioDataModule

DATASETS = {
    "mnist": MNISTDataModule,
    "discotube": DiscotubeAudioDataModule,
    "discotube_multiview": DiscotubeMultiViewAudioDataModule,
    "discotube_text_audio": DiscotubeTextAudioDataModule,
}
