from typing import Tuple

import gin.torch
import torch


@gin.configurable
class Waveform(torch.nn.Module):
    def __init__(
        self,
        sr: int,
        norm_mean: float | None,
        norm_std: float | None,
        patch_size: Tuple[int, int],
    ):
        super().__init__()

        self.mean = norm_mean
        self.std = norm_std

        # store the number of dimensions to be used by the training modules
        self.sr = sr
        self.hop_len = 1
        self.rep_dims = 1
        self.patch_size = patch_size

    def znorm(self, input_values: torch.Tensor) -> torch.Tensor:
        return (input_values - (self.mean)) / (self.std)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.unsqueeze(-2)

        # normalize
        if self.mean is not None and self.std is not None:
            waveform = self.znorm(waveform)

        return waveform
