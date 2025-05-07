from typing import Any, Tuple
from torch import nn

from .net import Net


class MLP(Net):
    def __init__(
        self,
        hidden_shape: Tuple[Tuple] = ((32, 16), (16, 8)),
        hidden_activation: Any = nn.ReLU(),
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.hidden_activation = hidden_activation

        self.l1 = nn.Sequential(nn.Linear(*self.input_shape), self.hidden_activation)

        hidden_layers = [
            nn.Sequential(nn.Linear(*hs), self.hidden_activation)
            for hs in self.hidden_shape
        ]
        self.l2 = nn.Sequential(*hidden_layers)
        self.l3 = nn.Linear(*self.output_shape)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)

        assert x.shape[1] == self.input_shape[0], (
            f"Expected shape {self.input_shape}, got {x.shape}"
        )

        x = self.l1(x)
        if self.hidden_shape:
            x = self.l2(x)

        return self.l3(x)
