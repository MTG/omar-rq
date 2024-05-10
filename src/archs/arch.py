from typing import Any

import torch.nn as nn
import gin.torch


@gin.configurable
class Arch(nn.Module):
    """Auxilliary class to define common parameters for all architectures"""

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
