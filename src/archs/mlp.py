from typing import Any
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: tuple = (64, 32),
        hidden_shape: tuple[tuple] = ((32, 16), (16, 8)),
        output_shape: tuple = (8, 8),
        hidden_activation: Any = nn.ReLU(),
    ):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.hidden_activation = hidden_activation

        self.l1 = nn.Sequential(nn.Linear(*input_shape), self.hidden_activation)

        hidden_layers = [
            nn.Sequential(nn.Linear(*hs), self.hidden_activation)
            for hs in self.hidden_shape
        ]
        self.l2 = nn.Sequential(*hidden_layers)
        self.l3 = nn.Linear(*output_shape)

    def forward(self, x):
        x = self.l1(x)
        if self.hidden_shape:
            x = self.l2(x)

        return self.l3(x)
