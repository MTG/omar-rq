from typing import Union
from pathlib import Path

import gin.torch
import numpy
import torch
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample


@gin.configurable
class AudioDataset(Dataset):
    """Generic audio dataset."""

    def __init__(
        self,
        data_dir: Path,
        filelist: Path,
        num_frames: int,
        orig_freq: int,
        new_freq: int,
        mono: bool,
        half_precision: bool,
        frame_offset: Union[int, str] = "random",
    ):
        self.data_dir = data_dir
        with open(filelist, "r") as f:
            self.filelist = [line.rstrip() for line in f.readlines()]

        self.num_frames = num_frames
        self.frame_offset = frame_offset
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resample = Resample(orig_freq=self.orig_freq, new_freq=self.new_freq)
        self.mono = mono
        self.half_precision = half_precision

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file_path = self.data_dir / self.filelist[idx]

        # load audio
        audio = self.load_audio(file_path, frame_offset=self.frame_offset)

        return [audio]

    def load_audio(
        self, file_path: Path, frame_offset: Union[int, str, torch.Tensor] = 0
    ):
        if frame_offset == "random":
            n_samples = self.get_audio_duration(file_path)
            # if num samples is less than self.num_frames, we just return the audio and pad it
            if n_samples < self.num_frames:
                offset_floats = 0
                offset_bytes = 0
                mmap = numpy.memmap(
                    file_path,
                    offset=offset_bytes,
                    dtype="float16",
                    mode="r",
                    shape=(1, n_samples),
                )
                audio = numpy.array(mmap)
                audio = numpy.pad(
                    audio,
                    ((0, 0), (0, self.num_frames - audio.shape[1])),
                    mode="constant",
                )
                del mmap

            else:
                offset_floats = torch.randint(
                    0, n_samples - self.num_frames, (1,)
                ).item()
                offset_bytes = offset_floats * 2  # 2 bytes per float
                mmap = numpy.memmap(
                    file_path,
                    offset=offset_bytes,
                    dtype="float16",
                    mode="r",
                    shape=(1, self.num_frames),
                )
                audio = numpy.array(mmap)
                del mmap

        elif isinstance(frame_offset, int):
            raise NotImplementedError("frame_offset as int is not implemented yet")
        else:
            raise ValueError(f"Invalid frame_offset: {frame_offset}")

        # downmix to mono if necessary
        if audio.shape[0] > 1 and self.mono:
            audio = torch.mean(audio, dim=0, keepdim=False)

        audio = torch.from_numpy(audio)

        # resample if necessary
        if self.orig_freq != self.new_freq:
            # only works with float tensors
            audio = audio.float()
            audio = self.resample(audio)

        audio = audio.squeeze(0)

        # work with 16-bit precission
        if self.half_precision:
            audio = audio.half()
        else:
            audio = audio.float()

        return audio

    @staticmethod
    def get_audio_duration(filepath: Path):
        path = Path(filepath)
        bytes = path.stat().st_size
        return bytes // 2  # porque lo guardamos como halfs, 2 byes por float


@gin.configurable
class AudioDataModule(L.LightningDataModule):
    """DataModule for the AudioDataset."""

    def __init__(
        self,
        batch_size: int,
        data_dir: Path,
        filelist_train: Path,
        filelist_val: Path,
        num_workers: int,
    ):
        super().__init__()

        self.batch_size = batch_size

        self.data_dir = Path(data_dir)
        self.filelist_train = Path(filelist_train)
        self.filelist_val = Path(filelist_val)

        self.num_workers = num_workers

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
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class MultiViewAudioDataset(AudioDataset):
    """Multiview audio dataset.

    The views are returned as a dictionary with keys "view_0", "view_1", ..., "view_N".
    This class only supports two views for now.
    """

    def __getitem__(self, idx):
        file_path = self.data_dir / self.filelist[idx]

        try:
            # get the number of samples in the audio
            n_samples = self.get_audio_duration(file_path)

            # compute the offsets for the two views
            offsets = self.get_views_offset(n_samples)

            # audio_full, sr = torchaudio.load(file_path)
            # sr = 44100
            # audio_full = torch.randn(1, n_samples)

            views = dict()
            for i, offset in enumerate(offsets):
                audio = self.load_audio(file_path, frame_offset=offset)
                # audio = audio_full[:, offset : offset + self.num_frames]

                # downmix to mono if necessary
                if audio.shape[0] > 1 and self.mono:
                    audio = torch.mean(audio, dim=0, keepdim=False)

                # resample if necessary
                if self.orig_freq != self.new_freq:
                    audio = self.resample(audio)

                views[f"view_{i}"] = audio.squeeze(0)

            return views

        except Exception as e:
            # TODO add this to the lightning logger
            print(f"Error loading {file_path}, {e}")
            return self.__getitem__(idx + 1)

    def get_views_offset(self, length: int, prob_floor: float = 0.1):
        """Get indices (offsets) for two non-overlapping vews of the audio.

        Other works consider sampling the views within a fixed time window
        (e.g., 5 seconds, https://arxiv.org/pdf/2210.03799)

        In our appraoch, we sample the offset for the second view from a
        triangle-shaped distribution center around the first view, while not
        allowing for overlap between views.
        The distribution follows this shape (/|_|\\) where `_` has prob=0
        as this is the region around the first view.
        """

        # sample the first offset from a uniform distribution
        offset_0 = torch.randint(0, length - self.num_frames, (1,)).item()

        # we downsample the audio by a factor of 1000 to make the computation
        # of the second offset efficient.
        scale_factor = 1000
        offset_0_ds = offset_0 // scale_factor
        length_ds = length // scale_factor
        num_frames_ds = self.num_frames // scale_factor

        # compute probs for the full song using a simple triangle distribution.
        # This favours offsets to be close while not allowing for overlap.
        ramp_up = torch.arange(0, offset_0_ds + 1) / offset_0_ds
        n_steps_down = length_ds - num_frames_ds - offset_0_ds
        ramp_down = torch.arange(n_steps_down, 0, -1) / n_steps_down

        probs = torch.cat([ramp_up, ramp_down], dim=0)

        # add a floor value and normalize
        probs = (probs + prob_floor) / torch.sum(probs + prob_floor)

        # zero out indices that would overlap with the first view
        bound_l = max(0, offset_0_ds - num_frames_ds)
        bound_r = min(len(probs), offset_0_ds + num_frames_ds)

        probs[bound_l:bound_r] = torch.zeros(bound_r - bound_l)

        assert torch.sum(probs) > 0, "All probs are zero"

        # sample the second offset and upsample
        offset_1_ds = torch.multinomial(probs, 1).item()
        offset_1 = offset_1_ds * scale_factor

        # shift to the left if we are out of bounds
        last_sample = offset_1 + self.num_frames - length
        if last_sample > 0:
            offset_1 -= last_sample

        assert offset_1 + self.num_frames < length, "Second view is out of bounds"

        return offset_0, offset_1


class MultiViewAudioDataModule(AudioDataModule):
    """DataModule for the AudioDataset."""

    def setup(self, stage: str):
        self.dataset_train = MultiViewAudioDataset(
            self.data_dir,
            filelist=self.filelist_train,
        )
        self.dataset_val = MultiViewAudioDataset(
            self.data_dir,
            filelist=self.filelist_val,
        )
