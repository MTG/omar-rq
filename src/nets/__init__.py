from .mlp import MLP
from .net import Net
from .melspectrogram import MelSpectrogram
from .transformer import Transformer
from .conformer import Conformer
#from .encodec import EnCodec
from .xlstm import XLSTM
from .cqt import CQT

NETS = {
    "net": Net,
    "mlp": MLP,
    "melspectrogram": MelSpectrogram,
    "transformer": Transformer,
    "conformer": Conformer,
    #"encodec": EnCodec,
    "cqt": CQT,
}
