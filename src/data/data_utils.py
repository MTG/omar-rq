from typing import Union
from pathlib import Path

import gin.torch
import torch
from torch.utils.data import Dataset
from torchaudio import load
from torchaudio.transforms import Resample


@gin.configurable
class AudioDataset(Dataset):
    """Generic audio dataset."""

    def __init__(
        self,
        data_dir: Path,
        filelist: Path,
        frame_offset: Union[int, str],
        num_frames: int,
        new_freq: int = 16000,
        mono: bool = True,
    ):
        self.data_dir = data_dir
        with open(filelist, "r") as f:
            self.filelist = [l.rstrip() for l in f.readlines()]

        self.frame_offset = frame_offset
        self.num_frames = num_frames
        self.new_freq = new_freq
        self.resample = Resample(new_freq=self.new_freq)
        self.mono = mono

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file_path = self.data_dir / self.filelist[idx]

        # load audio
        audio, sr = self.load_audio(file_path)

        # downmix to mono if necessary
        if audio.shape[0] > 1 and self.mono:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # resample if necessary
        if sr != self.new_freq:
            audio = self.resample_audio(audio, sr)

        return audio

    def load_audio(self, file_path: Path):
        # TODO fix random
        if self.frame_offset == "random":
            offset = torch.randint(0, 1000, (1,)).item()
        else:
            offset = self.frame_offset

        audio, sr = load(
            file_path,
            frame_offset=offset,
            num_frames=self.num_frames,
        )
        return audio, sr

    def resample_audio(self, audio, sr):
        if self.resample.orig_freq != sr:
            self.resample = Resample(new_freq=self.new_freq, orig_freq=sr)
        return self.resample(audio)
