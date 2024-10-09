import math
import pdb

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from .common_former import DeepNorm
from .rope import RotaryEmbedding
from .xlstm_module import xLSTM


def small_init_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param



@gin.configurable
class XLSTM(nn.Module):
    """
    XLSTM Encoder Module.

    Parameters:

    Inputs:
      x (Tensor): input spectrogram of dimension (batch_size, time, d_input)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices

    Outputs:
      Tensor (batch_size, time, embed_dim): Output tensor from the conformer encoder
    """

    def __init__(
        self,
        patch_size,
        embed_dim: int,
        depth: int,
        ker_size: int,
        signature: tuple,
        p_factor: tuple,
        head_num: int,
        head_dim: int,
        input_dropout: float,
        num_patches: int,
    ):
        super(XLSTM, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.signature = signature
        self.depth = depth
        self.ker_size = ker_size
        self.p_factor = p_factor
        self.head_num = head_num
        self.head_dim = head_dim
        self.num_patches = num_patches

        self.input_dropout = nn.Dropout(input_dropout)

        self.model = xLSTM(
            vocab_size=self.embed_dim,
            num_layers=self.depth,
            signature=self.signature,
            inp_dim=self.embed_dim,
            head_dim=self.head_dim,
            head_num=self.head_num,
            p_factor=self.p_factor,
            ker_size=self.ker_size
        )

    def forward(self, x):
        x = self.input_dropout(x)
        # pdb.set_trace()
        out, h = self.model(x)
        return out
