from pathlib import Path

import gin.torch
import torch
from transformers import EncodecModel


@gin.configurable
class EnCodec(torch.nn.Module):
    def __init__(
        self,
        weights_path: Path,
    ):
        super().__init__()

        self.net = EncodecModel.from_pretrained(weights_path)
        self.net.eval()

        # Hardcoded values, not parameters
        self.sr = 24000
        self.hop_len = 320
        self.rep_dims = 128

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # We do not consider finetuning EnCodec for now
        with torch.no_grad():
            self.eval()

            waveform = waveform.unsqueeze(1)
            # EnCodeC expects float32
            waveform = waveform.to(dtype=torch.float32)

            reps = self.net.encoder(waveform)

        return reps
