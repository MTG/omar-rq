from .mnist import MNISTDataModule
from .discotube import DiscotubeDataModule

DATASETS = {
    "mnist": MNISTDataModule,
    "discotube": DiscotubeDataModule,
}
