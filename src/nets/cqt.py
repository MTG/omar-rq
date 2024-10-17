import gin.torch

import math
import warnings

import torch
from torch import Tensor
from torchaudio.transforms import Resample
from typing import Callable, Tuple

# CQT implementation for torchaudio from PR #3804 by @d-dawg78:
# https://github.com/pytorch/audio/pull/3804


def frequency_set(
    f_min: float,
    n_bins: int,
    bins_per_octave: int,
    dtype: torch.dtype,
) -> Tuple[Tensor, int]:
    r"""Return a set of frequencies that assumes an equal temperament tuning system.

    .. devices:: CPU

    Adapted from librosa: https://librosa.org/doc/main/generated/librosa.interval_frequencies.html

    Args:
        f_min (float): minimum frequency in Hz.
        n_bins (int): number of frequency bins.
        bins_per_octave (int): number of bins per octave.
    Returns:
        torch.Tensor: frequencies.
        int: number of octaves
    """
    if f_min < 0.0 or n_bins < 1 or bins_per_octave < 1:
        raise ValueError(
            "f_min must be positive. n_bins and bins_per_octave must be ints and superior to 1."
        )

    n_octaves = math.ceil(n_bins / bins_per_octave)
    ratios = 2.0 ** (
        torch.arange(0, bins_per_octave * n_octaves, dtype=dtype) / bins_per_octave
    )
    return f_min * ratios[:n_bins], n_octaves


def relative_bandwidths(freqs: Tensor, n_bins: int, bins_per_octave: int) -> Tensor:
    r"""Compute relative bandwidths for specified frequencies.

    .. devices:: CPU

    Adapted from librosa: https://librosa.org/doc/main/generated/librosa.filters.wavelet_lengths.html

    Args:
        freqs (Tensor): set of frequencies.
        n_bins (int): number of frequency bins.
        bins_per_octave (int): number of bins per octave.
    Returns:
        torch.Tensor: relative bandwidths for set of frequencies.
    """
    if min(freqs) < 0.0 or n_bins < 1 or bins_per_octave < 1:
        raise ValueError(
            "freqs must be positive. n_bins and bins_per_octave must be positive ints."
        )

    if n_bins > 1:
        # Approximate local octave resolution around each frequency
        bandpass_octave = torch.empty_like(freqs)
        log_freqs = torch.log2(freqs)

        # Reflect at the lowest and highest frequencies
        bandpass_octave[0] = 1 / (log_freqs[1] - log_freqs[0])
        bandpass_octave[-1] = 1 / (log_freqs[-1] - log_freqs[-2])

        # Centered difference
        bandpass_octave[1:-1] = 2 / (log_freqs[2:] - log_freqs[:-2])

        # Relative bandwidths
        alpha = (2.0 ** (2 / bandpass_octave) - 1) / (2.0 ** (2 / bandpass_octave) + 1)
    else:
        # Special case when single basis frequency is used
        rel_band_coeff = 2.0 ** (1.0 / bins_per_octave)
        alpha = torch.tensor(
            [(rel_band_coeff**2 - 1) / (rel_band_coeff**2 + 1)], dtype=freqs.dtype
        )

    return alpha


def wavelet_lengths(
    freqs: Tensor,
    sr: float,
    alpha: Tensor,
    gamma: float,
) -> Tuple[Tensor, float]:
    r"""Length of each filter in a wavelet basis.

    .. devices:: CPU

    Source:
        * https://librosa.org/doc/main/generated/librosa.filters.wavelet_lengths.html

    Args:
        freqs (Tensor): set of frequencies.
        sr (float): sample rate.
        alpha (Tensor): relative bandwidths for set of frequencies.
        gamma (float): bandwidth offset for filter length computation.

    Returns:
        Tensor: filter lengths.
        float: cutoff frequency of highest bin.
    """
    if gamma < 0.0 or sr < 0.0:
        raise ValueError("gamma and sr must be positive!")

    if min(freqs) < 0.0 or min(alpha) < 0.0:
        raise ValueError("freqs and alpha must be positive!")

    # We assume filter_scale (librosa param) is 1
    Q = 1.0 / alpha

    # Output upper bound cutoff frequency
    # 3.0 > all common window function bandwidths
    # https://librosa.org/doc/main/_modules/librosa/filters.html
    cutoff_freq = max(freqs * (1 + 0.5 * 3.0 / Q) + 0.5 * gamma)

    # Convert frequencies to filter lengths
    lengths = Q * sr / (freqs + gamma / alpha)

    return lengths, cutoff_freq


def wavelet_fbank(
    freqs: Tensor,
    sr: float,
    alpha: Tensor,
    gamma: float,
    window_fn: Callable[..., Tensor],
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    r"""Wavelet filterbank constructed from set of center frequencies.

    .. devices:: CPU

    Source:
        * https://librosa.org/doc/main/generated/librosa.filters.wavelet.html

    Args:
        freqs (Tensor): set of frequencies.
        sr (float): sample rate.
        alpha (Tensor): relative bandwidths for set of frequencies.
        gamma (float): bandwidth offset for filter length computation.
        window_fn (Callable[..., Tensor]): a function to create a window tensor.

    Returns:
        Tensor: wavelet filters.
        Tensor: wavelet filter lengths.
    """
    # First get filter lengths
    lengths, _ = wavelet_lengths(freqs=freqs, sr=sr, alpha=alpha, gamma=gamma)

    # Next power of 2
    pad_to_size = 1 << (int(max(lengths)) - 1).bit_length()

    for index, (ilen, freq) in enumerate(zip(lengths, freqs)):
        # Build filter with length ceil(ilen)
        t = torch.arange(-ilen // 2, ilen // 2, dtype=dtype) * 2 * torch.pi * freq / sr
        sig = torch.cos(t) + 1j * torch.sin(t)

        # Multiply with window
        sig_len = len(sig)
        sig = sig * window_fn(sig_len)

        # L1 normalize
        sig = torch.nn.functional.normalize(sig, p=1.0, dim=0)

        # Pad signal left and right to correct size
        l_pad = math.floor((pad_to_size - sig_len) / 2)
        r_pad = math.ceil((pad_to_size - sig_len) / 2)
        sig = torch.nn.functional.pad(sig, (l_pad, r_pad), mode="constant", value=0.0)
        sig = sig.unsqueeze(0)

        if index == 0:
            filters = sig
        else:
            filters = torch.cat([filters, sig], dim=0)

    return filters, lengths


class VQT(torch.nn.Module):
    r"""Create the variable Q-transform for a raw audio signal.
    .. devices:: CPU CUDA
    .. properties:: Autograd
    Sources
        * https://librosa.org/doc/main/generated/librosa.vqt.html
        * https://www.aes.org/e-lib/online/browse.cfm?elib=17112
        * https://newt.phys.unsw.edu.au/jw/notes.html
    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        hop_length (int, optional): Length of hop between VQT windows. (Default: ``400``)
        f_min (float, optional): Minimum frequency, which corresponds to first note.
            (Default: ``32.703``, or the frequency of C1 in Hz)
        n_bins (int, optional): Number of VQT frequency bins, starting at ``f_min``. (Default: ``84``)
        gamma (float, optional): Offset that controls VQT filter lengths. Larger values
            increase the time resolution at lower frequencies. (Default: ``0.``)
        bins_per_octave (int, optional): Number of bins per octave. (Default: ``12``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interp_hann``, ``sinc_interp_kaiser``] (Default: ``"sinc_interp_hann"``)
        dtype (torch.device, optional):
            Determines the precision that kernels are pre-computed and cached in. Note that complex
            bases are either cfloat or cdouble depending on provided precision.
            Options: [``torch.float``, ``torch.double``] (Default: ``torch.float``)
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.VQT(sample_rate)
        >>> vqt = transform(waveform)  # (..., n_bins, time)
    """

    __constants__ = ["resample", "forward_params"]

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 256,
        f_min: float = 32.703,
        n_bins: int = 84,
        gamma: float = 0.0,
        bins_per_octave: int = 12,
        window_fn: Callable[..., Tensor] = torch.hann_window,
        resampling_method: str = "sinc_interp_hann",
        dtype: torch.dtype = torch.float,
    ) -> None:
        super(VQT, self).__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.VQT")

        n_filters = min(bins_per_octave, n_bins)
        frequencies, n_octaves = frequency_set(f_min, n_bins, bins_per_octave, dtype)
        alpha = relative_bandwidths(frequencies, n_bins, bins_per_octave)
        _, cutoff_freq = wavelet_lengths(frequencies, sample_rate, alpha, gamma)

        self.resample = Resample(2, 1, resampling_method, dtype=dtype)

        # Generate errors or warnings if needed
        # Number of divisions by 2 before number becomes odd
        num_hop_downsamples = len(str(bin(hop_length)).split("1")[-1])
        nyquist = sample_rate / 2

        if cutoff_freq > nyquist:
            raise ValueError(
                f"Maximum bin cutoff frequency is approximately {cutoff_freq} and superior to the "
                f"Nyquist frequency {nyquist}. Try to reduce the number of frequency bins."
            )
        if n_octaves - 1 > num_hop_downsamples:
            warnings.warn(
                f"Hop length can be divided {num_hop_downsamples} times by 2 before becoming "
                f"odd. The VQT is however being computed for {n_octaves} octaves. Consider setting "
                "the hop length to a ``more even'' number for more accurate results."
            )
        if nyquist / cutoff_freq > 4:
            warnings.warn(
                f"The Nyquist frequency {nyquist} is significantly higher than the highest filter's approximate "
                f"cutoff frequency {cutoff_freq}. Consider resampling your signal to a lower sample "
                "rate or increasing the number of bins before VQT computation for more accurate results."
            )

        # Now pre-compute what's needed for forward loop
        self.forward_params = []
        temp_sr, temp_hop = float(sample_rate), hop_length
        register_index = 0

        for oct_index in range(n_octaves - 1, -1, -1):
            # Slice out correct octave
            indices = slice(n_filters * oct_index, n_filters * (oct_index + 1))

            octave_freqs = frequencies[indices]
            octave_alphas = alpha[indices]

            # Compute wavelet filterbanks
            basis, lengths = wavelet_fbank(
                octave_freqs, temp_sr, octave_alphas, gamma, window_fn, dtype
            )
            n_fft = basis.shape[1]

            # Normalize wrt FFT window length
            factors = lengths.unsqueeze(1) / float(n_fft)
            basis *= factors

            # Wavelet basis FFT
            fft_basis = torch.fft.fft(basis, n=n_fft, dim=1)[:, : (n_fft // 2) + 1]
            fft_basis *= math.sqrt(sample_rate / temp_sr)

            self.register_buffer(f"fft_basis_{register_index}", fft_basis)
            self.forward_params.append((temp_hop, n_fft))

            register_index += 1

            if temp_hop % 2 == 0:
                temp_sr /= 2.0
                temp_hop //= 2

        # Create ones on the correct device in the forward pass
        self.ones = lambda x: torch.ones(x, dtype=dtype, device=self.fft_basis_0.device)

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., channels, time).
                2D or 3D; batch dimension is optional.
        Returns:
            Tensor: variable-Q transform of size (..., channels, ``n_bins``, time).
        """
        # Iterate down the octaves
        for buffer_index, (temp_hop, n_fft) in enumerate(self.forward_params):
            # STFT matrix
            if waveform.ndim == 3:
                # torch stft does not support 3D computation yet
                # iterate through channels for stft computation
                for channel in range(waveform.shape[1]):
                    channel_dft = torch.stft(
                        waveform[:, channel, :],
                        n_fft=n_fft,
                        hop_length=temp_hop,
                        window=self.ones(n_fft),
                        pad_mode="constant",
                        return_complex=True,
                    )

                    if channel == 0:
                        dft = channel_dft.unsqueeze(1)
                    else:
                        dft = torch.cat([dft, channel_dft.unsqueeze(1)], dim=1)

            else:
                dft = torch.stft(
                    waveform,
                    n_fft=n_fft,
                    hop_length=temp_hop,
                    window=self.ones(n_fft),
                    pad_mode="constant",
                    return_complex=True,
                )

            # Compute octave vqt
            temp_vqt = torch.einsum(
                "ij,...jk->...ik", getattr(self, f"fft_basis_{buffer_index}"), dft
            )

            if buffer_index == 0:
                vqt = temp_vqt
            else:
                vqt = torch.cat([temp_vqt, vqt], dim=-2)

            if temp_hop % 2 == 0:
                waveform = self.resample(waveform)
                waveform /= math.sqrt(0.5)

        return vqt


@gin.configurable
class CQT(torch.nn.Module):
    r"""Create the constant Q-transform for a raw audio signal.
    .. devices:: CPU CUDA
    .. properties:: Autograd
    Sources
        * https://librosa.org/doc/main/generated/librosa.cqt.html
        * https://www.aes.org/e-lib/online/browse.cfm?elib=17112
        * https://newt.phys.unsw.edu.au/jw/notes.html
    Args:
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        hop_length (int, optional): Length of hop between CQT windows. (Default: ``400``)
        f_min (float, optional): Minimum frequency, which corresponds to first note.
            (Default: ``32.703``, or the frequency of C1 in Hz)
        n_bins (int, optional): Number of CQT frequency bins, starting at ``f_min``. (Default: ``84``)
        bins_per_octave (int, optional): Number of bins per octave. (Default: ``12``)
        window_fn (Callable[..., Tensor], optional): A function to create a window tensor
            that is applied/multiplied to each frame/window. (Default: ``torch.hann_window``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interp_hann``, ``sinc_interp_kaiser``] (Default: ``"sinc_interp_hann"``)
        dtype (torch.device, optional):
            Determines the precision that kernels are pre-computed and cached in. Note that complex
            bases are either cfloat or cdouble depending on provided precision.
            Options: [``torch.float``, ``torch.double``] (Default: ``torch.float``)
    Example
        >>> waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
        >>> transform = transforms.CQT(sample_rate)
        >>> cqt = transform(waveform)  # (..., n_bins, time)
    """

    __constants__ = ["transform"]

    def __init__(
        self,
        sr: int,
        hop_len: int,
        power: int,
        bins_per_octave: int,
        n_bins: int,
        f_min: float,
        magnitude: bool,
        logC: bool,
        norm_mean: float | None,
        norm_std: float | None,
        patch_size: Tuple[int, int],
        ###
        window_fn: Callable[..., Tensor] = torch.hann_window,
        resampling_method: str = "sinc_interp_hann",
        dtype: torch.dtype = torch.float,
    ) -> None:
        super(CQT, self).__init__()
        torch._C._log_api_usage_once("torchaudio.transforms.CQT")

        self.sr = sr
        self.hop_len = hop_len
        self.magnitude = magnitude
        self.power = power
        self.logC = logC
        self.mean = norm_mean
        self.std = norm_std

        self.patch_size = patch_size

        # CQT corresponds to a VQT with gamma set to 0
        self.transform = VQT(
            sample_rate=sr,
            hop_length=hop_len,
            f_min=f_min,
            n_bins=n_bins,
            gamma=0.0,
            bins_per_octave=bins_per_octave,
            window_fn=window_fn,
            resampling_method=resampling_method,
            dtype=dtype,
        )

        self.rep_dims = n_bins

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., channels, time).
                2D or 3D; batch dimension is optional.
        Returns:
            Tensor: constant-Q transform spectrogram of size (..., channels, ``n_bins``, time).
        """
        cqt = self.transform(waveform)

        if self.magnitude:
            cqt = torch.abs(cqt)

        if self.power > 1:
            cqt = cqt**self.power

        if self.logC:
            cqt = torch.log10(1 + 10000 * cqt)

        if self.mean is not None and self.std is not None:
            cqt = self.znorm(cqt)

        return cqt

    def znorm(self, input_values: torch.Tensor) -> torch.Tensor:
        return (input_values - (self.mean)) / (self.std)
