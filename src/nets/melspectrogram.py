import gin.torch
import torch
from torchaudio.transforms import (
    Spectrogram,
    MelScale,
    TimeStretch,
    FrequencyMasking,
    TimeMasking,
)

# According to Pytoch, mel-spectrogram should be implemented as a module:
# https://pytorch.org/audio/stable/transforms.html


@gin.configurable
class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sr: int,
        win_len: int,
        hop_len: int,
        power: int,
        n_mel: int,
        stretch_factor: float,
        freq_mask_param: int,
        time_mask_param: int,
        norm: str,
        mel_scale: str,
    ):
        super().__init__()

        self.spec = Spectrogram(
            n_fft=win_len,
            win_length=win_len,
            hop_length=hop_len,
            power=power,
        )

        self.spec_aug = torch.nn.Sequential(
            TimeStretch(stretch_factor, fixed_rate=True),
            FrequencyMasking(freq_mask_param=freq_mask_param),
            TimeMasking(time_mask_param=time_mask_param),
        )

        self.mel_scale = MelScale(
            n_mels=n_mel,
            sample_rate=sr,
            n_stft=win_len // 2 + 1,
            norm=norm,
            mel_scale=mel_scale,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        # resampled = self.resample(waveform)

        # Convert to power spectrogram
        spec = self.spec(waveform)

        # Apply SpecAugment
        spec = self.spec_aug(spec)

        # Convert to mel-scale
        mel = self.mel_scale(spec)

        # Apply logC compression
        logmel = torch.log10(1 + mel * 10000)

        return logmel
