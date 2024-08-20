from .mlp import MLP
from .net import Net
from .melspectrogram import MelSpectrogram
from .transformer import Transformer
from .conformer import Conformer

NETS = {
    "net": Net,
    "mlp": MLP,
    "melspectrogram": MelSpectrogram,
    "transformer": Transformer,
    "conformer": Conformer,
}
