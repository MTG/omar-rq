import torch
from torchaudio.transforms import (
    Spectrogram,
    MelScale,
    TimeStretch,
    FrequencyMasking,
    TimeMasking,
)

# According to Pytoch, mel-spectrogram is implemented as a module:Jk w
# https://pytorch.org/audio/stable/transforms.html


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        n_mel: int = 96,
        stretch_factor: float = 0.8,
        sr: int = 16000,
        freq_mask_param: int = 80,
        time_mask_param: int = 80,
    ):
        super().__init__()

        self.spec = Spectrogram(n_fft=n_fft, power=2)

        self.spec_aug = torch.nn.Sequential(
            TimeStretch(stretch_factor, fixed_rate=True),
            FrequencyMasking(freq_mask_param=freq_mask_param),
            TimeMasking(time_mask_param=time_mask_param),
        )

        self.mel_scale = MelScale(n_mels=n_mel, sample_rate=sr, n_stft=n_fft // 2 + 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        # resampled = self.resample(waveform)

        # Convert to power spectrogram
        spec = self.spec(waveform)

        # Apply SpecAugment
        spec = self.spec_aug(spec)

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        return mel
