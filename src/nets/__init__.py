from .mlp import MLP
from .net import Net
from .melspectrogram import MelSpectrogram
from .transformer import Transformer

NETS = {
    "net": Net,
    "mlp": MLP,
    "melspectrogram": MelSpectrogram,
    "transformer": Transformer,
}
