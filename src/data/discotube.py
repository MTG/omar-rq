from pathlib import Path

import gin.torch
import pytorch_lightning as L
from torch.utils.data import DataLoader

from .data_utils import AudioDataset


@gin.configurable
class DiscotubeDataModule(L.LightningDataModule):
    """DataModule for the Discogs dataset."""

    def __init__(
        self,
        batch_size: int,
        data_dir: Path,
        filelist_train: Path,
        filelist_val: Path,
    ):
        super().__init__()

        self.batch_size = batch_size

        self.data_dir = Path(data_dir)
        self.filelist_train = Path(filelist_train)
        self.filelist_val = Path(filelist_val)

    def setup(self, stage: str):
        self.dataset_train = AudioDataset(
            self.data_dir,
            filelist=self.filelist_train,
        )
        self.dataset_val = AudioDataset(
            self.data_dir,
            filelist=self.filelist_val,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)
