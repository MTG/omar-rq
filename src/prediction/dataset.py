import pickle as pk
from pathlib import Path

import torch
import torchaudio
import gin.torch
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from tqdm import tqdm


class AudioEmbeddingDataset(Dataset):
    """Dataset for loading audio files."""

    def __init__(
        self,
        data_dir: Path,
        file_format: str,
        new_freq: int,
        mono: bool,
        half_precision: bool,
        overlap_ratio: float,
        n_seconds: int,
        last_chunk_ratio: float,
    ):
        self.data_dir = Path(data_dir)
        self.filelist = sorted(self.data_dir.rglob(f"*.{file_format}"))
        assert len(self.filelist) > 0, f"No files found in {self.data_dir}"
        print(f"Found {len(self.filelist)} *.{file_format} files in {self.data_dir}.")

        self.orig_freq = None
        self.new_freq = new_freq
        self.resample = Resample()
        self.mono = mono
        self.half_precision = half_precision

        self.index = dict()  # idx: (file_path, seg)
        self.track2sr = dict()  # file_path: sr

        self.overlap_ratio = overlap_ratio
        self.last_chunk_ratio = last_chunk_ratio

        self.n_seconds = n_seconds

        assert (
            self.overlap_ratio >= 0 and self.overlap_ratio < 1
        ), "Overlap ratio must be between 0 and 1."

        self.compute_segments_per_file()

        print(f"Found {len(self)} segments in {len(self.filelist)} files.")

    def compute_segments_per_file(self):
        self.index = dict()
        self.track2sr = dict()

        # check if we have cached the number of segments
        for file, data in ((".index", self.index), (".sr", self.track2sr)):
            if (self.data_dir / file).exists():
                with open(self.data_dir / file, "rb") as f:
                    data.update(pk.load(f))

        if not self.index or not self.track2sr:
            print("Computing segments per file...")

            i = 0
            for filepath in tqdm(self.filelist):
                try:
                    metadata = torchaudio.info(self.data_dir / filepath)

                    # number of input samples
                    n_x = metadata.num_frames
                    # number of chunk samples
                    n_c = self.n_seconds * metadata.sample_rate
                    # number of samples to hop
                    n_h = n_c * (1 - self.overlap_ratio)

                    # number of segments
                    n_s = int(1 + max(0, (n_x - n_c) // n_h))

                    # number of tailing samples
                    n_t = n_x - ((n_s - 1) * n_h + n_c)

                    # if tailing samples are more than a ratio of the chunk, add a segment
                    if n_t > n_c * self.last_chunk_ratio:
                        n_s += 1

                    self.track2sr[filepath] = metadata.sample_rate

                    for j in range(n_s):
                        self.index[i] = (filepath, j)
                        i += 1
                except Exception as e:
                    print(f"Error processing file {filepath}")

            # cache the number of segments for later use
            for file, data in ((".index", self.index), (".sr", self.track2sr)):
                with open(self.data_dir / file, "wb") as f:
                    pk.dump(data, f)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Get the file path
        file_path, segment = self.index[idx]

        # load audio
        try:
            num_frames = int(self.n_seconds * self.track2sr[file_path])
            frame_offset = num_frames * segment * (1 - self.overlap_ratio)
            audio, sr = torchaudio.load(
                self.data_dir / file_path,
                num_frames=num_frames,
                frame_offset=frame_offset,
            )  # (C, T)

            assert audio.shape[-1] <= num_frames, "Audio is longer than expected."

            # zero pad if necessary
            if audio.shape[-1] < num_frames:
                pad = torch.zeros((audio.shape[0], num_frames - audio.shape[-1]))
                audio = torch.cat([audio, pad], dim=-1)

            # TODO: why don't we fix mono? The rest of the code is not ready for 2 channel audio
            # downmix to mono if necessary
            if self.mono:
                # Do not keep the channel dimension for consistency with the training dataloader
                audio = torch.mean(audio, dim=0, keepdim=False)  # (T')

            # resample if necessary
            if sr != self.new_freq:
                # cache the resample object
                if sr != self.orig_freq:
                    self.resample = Resample(orig_freq=sr, new_freq=self.new_freq)
                    self.orig_freq = sr

                audio = audio.float()
                audio = self.resample(
                    audio
                )  # (T') | (C, T'), only the former is supported for now

            # TODO: On CPU created problems with half precision
            # work with 16-bit precision
            if self.half_precision:
                audio = audio.half()
            else:
                audio = audio.float()

            return audio, str(file_path)

        except Exception:
            print(f"Error loading file {file_path}")
            return None, file_path


@gin.configurable
class AudioEmbeddingDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        file_format: str,
        new_freq: int,
        mono: bool,
        half_precision: bool,
        num_workers: int,
        batch_size: int,
        overlap_ratio: float,
        n_seconds: int,
        last_chunk_ratio: float,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.file_format = file_format
        self.new_freq = new_freq
        self.mono = mono
        self.half_precision = half_precision
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.overlap_ratio = overlap_ratio
        self.n_seconds = n_seconds
        self.last_chunk_ratio = last_chunk_ratio

        assert 0 < self.overlap_ratio < 1, "overlap_ratio must be between 0 and 1."
        assert (
            0 < self.last_chunk_ratio < 1
        ), "last_chunk_ratio must be between 0 and 1."

    def setup(self, stage: str) -> None:
        if stage == "predict":
            self.dataset = AudioEmbeddingDataset(
                data_dir=self.data_dir,
                file_format=self.file_format,
                new_freq=self.new_freq,
                mono=self.mono,
                half_precision=self.half_precision,
                overlap_ratio=self.overlap_ratio,
                n_seconds=self.n_seconds,
                last_chunk_ratio=self.last_chunk_ratio,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
