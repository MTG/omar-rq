from typing import Union
from pathlib import Path

import torch
import torchaudio
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample


class AudioEmbeddingDataset(Dataset):
    """Dataset for loading audio files."""

    def __init__(
        self,
        data_dir: Path,
        file_format: str,
        orig_freq: int,
        new_freq: int,
        mono: bool,
        half_precision: bool,
    ):

        self.data_dir = Path(data_dir)
        self.filelist = sorted(self.data_dir.rglob(f"*.{file_format}"))
        assert len(self.filelist) > 0, f"No files found in {self.data_dir}"
        print(f"Found {len(self.filelist)} files in {self.data_dir}.")

        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resample = Resample(self.orig_freq, self.new_freq)
        self.mono = mono
        self.half_precision = half_precision

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        # Get the file path
        file_path = self.data_dir / self.filelist[idx]

        # load audio
        audio, sr = torchaudio.load(file_path)  # (C, T)

        # resample if necessary
        if sr != self.new_freq:
            audio = self.resample(audio)  # (C, T')

        # downmix to mono if necessary
        if audio.shape[0] > 1 and self.mono:
            audio = torch.mean(audio, dim=0, keepdim=True)  # (1, T')

        # TODO: On CPU created problems with half precision
        # work with 16-bit precission
        if self.half_precision:
            audio = audio.half()

        return audio, file_path

    @staticmethod
    def collate_fn(items):
        # TODO: Find a better way to do this
        return [item[0] for item in items][0], [item[1] for item in items][0]
