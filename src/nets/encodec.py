from typing import Tuple

import json
import math
from pathlib import Path

import gin.torch
import torch
from transformers import EncodecModel
from torchaudio.transforms import Resample

# Normalization stats obtained from the training data (Discogs23 dataset)
ENCODEC_MEAN = -0.5577428463574771
ENCODEC_STD = 5.863781862801963


@gin.configurable
class EnCodec(torch.nn.Module):
    def __init__(
        self,
        weights_path: Path,
        norm_type: str | None,
        stats_path: Path | None,
        orig_sr: int,
        patch_size: Tuple[int, int],
    ):
        super().__init__()

        self.net = EncodecModel.from_pretrained(weights_path)
        self.norm_type = norm_type
        self.orig_sr = orig_sr
        self.patch_size = patch_size

        if self.norm_type is not None:
            if stats_path is None:
                if self.norm_type == "dimensionwise":
                    raise ValueError(
                        "stats_path must be provided when using dimensionwise normalization"
                    )
                elif self.norm_type == "global":
                    # Fallback to hardcoded values
                    mean = torch.tensor(ENCODEC_MEAN)
                    std = torch.tensor(ENCODEC_STD)

            else:
                with open(stats_path, "r") as f:
                    stats = json.load(f)

                if self.norm_type == "dimensionwise":
                    mean = torch.tensor(stats["mean_dims"])
                    std = torch.tensor(stats["std_dims"])

                elif self.norm_type == "global":
                    mean = torch.tensor(stats["mean"])
                    std = torch.tensor(stats["std"])
                else:
                    raise ValueError(f"Invalid norm_type: {self.norm_type}")

            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

        # Hardcoded values, not parameters
        self.sr = 24000
        self.hop_len = 320
        self.rep_dims = 128

        self.resample = Resample(orig_freq=self.orig_sr, new_freq=self.sr)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # We do not consider finetuning EnCodec for now

        if self.orig_sr != self.sr:
            waveform = self.resample(waveform)

        with torch.no_grad():
            self.eval()

            if waveform.ndim == 1:
                # Asume raw audio of arbitrary length.
                # form batches of 30 seconds
                n_batches = math.ceil(len(waveform) / (self.sr * 30))
                pad_right = n_batches * self.sr * 30 - len(waveform)
                waveform = torch.nn.functional.pad(waveform, (0, pad_right))
                waveform = waveform.view(n_batches, -1)
                waveform = waveform.unsqueeze(1)

                # EnCodeC expects float32
                waveform = waveform.to(dtype=torch.float32)

            elif waveform.ndim == 2:
                # Assume batched audio with appropriate shape.
                # Let's skip the processor for efficiency.
                # Manually add channel dimension and cast to float32.
                waveform = waveform.unsqueeze(1)
                # EnCodeC expects float32
                waveform = waveform.to(dtype=torch.float32)

            elif waveform.ndim == 3:
                # Assume batched audio with channel dimension.
                # Just cast to float32.
                waveform = waveform.to(dtype=torch.float32)

            else:
                raise ValueError(
                    f"Expected waveform to have 1, 2 or 3 dimensions, got {waveform.ndim}"
                )

            reps = self.net.encoder(waveform)

            if self.norm_type == "dimensionwise":
                reps = (reps - self.mean[:, None]) / self.std[:, None]

            elif self.norm_type == "global":
                reps = (reps - self.mean) / self.std

        # reps = reps.squeeze(0)

        return reps
