from typing import Any

import torch.nn as nn
import gin.torch

from .melspectrogram import MelSpectrogram


@gin.configurable
class Net(nn.Module):
    """Auxilliary class to define common parameters and methods for all networks"""

    def __init__(
        self,
        input_shape: tuple = (64, 32),
        output_shape: tuple = (8, 8),
        hidden_activation: Any = nn.ReLU(),
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_activation = hidden_activation

        # define the mel spectrogram layer
        self.melspectrogram = MelSpectrogram()
        # self.melspectrogram.to(self.device)

    def get_parameter_count(self):
        """Log the number of trainable parameters in the network"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
