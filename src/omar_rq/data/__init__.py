from .discotube import DiscotubeAudioDataModule, DiscotubeMultiViewAudioDataModule
from .discotube_text_audio import DiscotubeTextAudioDataModule

DATASETS = {
    "discotube": DiscotubeAudioDataModule,
    "discotube_multiview": DiscotubeMultiViewAudioDataModule,
    "discotube_text_audio": DiscotubeTextAudioDataModule,
}
