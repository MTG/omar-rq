import math

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from .common_former import DeepNorm
from .rope import RotaryEmbedding


class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        num_heads,
        dropout=0.0,
        qkv_bias=False,
        use_rope=False,
        max_len=10000,
    ):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        self.d_in = d_in
        self.use_rope = use_rope
        self.rope_dim = self.head_dim

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_in, d_out)
        self.dropout = dropout

        # Initialize positional encodings or rotary embeddings
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(dim=self.rope_dim)
        else:
            self.positional_encoder = PositionalEncoder(
                embed_dim=d_out, max_len=max_len
            )

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        if self.use_rope:
            # Apply rotary embeddings to queries and keys in each attention layer
            queries = self.rotary_emb.rotate_queries_or_keys(queries)
            keys = self.rotary_emb.rotate_queries_or_keys(keys)
        else:
            # Apply sinusoidal positional encodings to queries and keys
            pos_encodings = (
                self.positional_encoder(num_tokens).unsqueeze(0).unsqueeze(1)
            )
            queries += pos_encodings
            keys += pos_encodings

        use_dropout = 0.0 if not self.training else self.dropout
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            context_vec = nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=use_dropout,
                is_causal=True,
            )

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )

        context_vec = self.proj(context_vec)

        return context_vec


class PositionalEncoder(nn.Module):
    """Generate positional encodings used in the relative multi-head attention module.
    These encodings are the same as the original transformer model: https://arxiv.org/abs/1706.03762

    Parameters:
      max_len (int): Maximum sequence length (time dimension)

    Inputs:
      len (int): Length of encodings to retrieve

    Outputs
      Tensor (len, embed_dim): Positional encodings
    """

    def __init__(self, embed_dim, max_len=10000):
        super(PositionalEncoder, self).__init__()
        self.embed_dim = embed_dim
        encodings = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
        encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
        self.register_buffer("encodings", encodings)

    def forward(self, len):
        return self.encodings[:len, :]


class ConvBlock(nn.Module):
    """
    Conformer convolutional block.

    Parameters:
      embed_dim (int): Dimension of the model
      kernel_size (int): Size of kernel to use for depthwise convolution
      dropout (float): Dropout probability

    Inputs:
      x (Tensor): (batch_size, time, embed_dim)
      mask: Unused

    Outputs:
      Tensor (batch_size, time, embed_dim): Output tensor from the convolution module

    """

    def __init__(self, embed_dim=144, kernel_size=31, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim, eps=6.1e-5)
        kernel_size = 31
        self.module = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=1),
            # first pointwise with 2x expansion
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                padding="same",
                groups=embed_dim,
            ),  # depthwise
            nn.BatchNorm1d(embed_dim, eps=6.1e-5),
            nn.GELU(),  # swish activation
            nn.Conv1d(
                in_channels=embed_dim, out_channels=embed_dim, kernel_size=1
            ),  # second pointwise
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        x = self.module(x)
        return x.transpose(1, 2)


class FeedForwardBlock(nn.Module):
    """
    Conformer feed-forward block.

    Parameters:
      embed_dim (int): Dimension of the model
      expansion (int): Expansion factor for first linear layer
      dropout (float): Dropout probability

    Inputs:
      x (Tensor): (batch_size, time, embed_dim)
      mask: Unused

    Outputs:
      Tensor (batch_size, time, embed_dim): Output tensor from the feed-forward module

    """

    def __init__(self, embed_dim=144, expansion=4, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.module = nn.Sequential(
            nn.LayerNorm(embed_dim, eps=6.1e-5),
            nn.Linear(
                embed_dim, int(embed_dim * expansion)
            ),  # expand to embed_dim * expansion
            nn.GELU(),  # swish activation
            nn.Dropout(dropout),
            nn.Linear(
                int(embed_dim * expansion), embed_dim
            ),  # project back to embed_dim
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.module(x)


class Conv2dSubsampling(nn.Module):
    """
    2d Convolutional subsampling.
    Subsamples time and freq domains of input spectrograms by a factor of 4, embed_dim times.

    Parameters:
      embed_dim (int): Dimension of the model

    Inputs:
      x (Tensor): Input spectrogram (batch_size, time, d_input)

    Outputs:
      Tensor (batch_size, time, embed_dim * (d_input // 4)): Output tensor from the conlutional subsampling module

    """

    def __init__(self, embed_dim=144):
        super(Conv2dSubsampling, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=2
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.module(x.unsqueeze(1))  # (batch_size, 1, time, d_input)
        batch_size, embed_dim, subsampled_time, subsampled_freq = output.size()
        output = output.permute(0, 2, 1, 3)
        output = output.contiguous().view(
            batch_size, subsampled_time, embed_dim * subsampled_freq
        )
        return output


class ConformerBlock(nn.Module):
    """
    Conformer Encoder Block.

    Parameters:
      embed_dim (int): Dimension of the model
      conv_kernel_size (int): Size of kernel to use for depthwise convolution
      feed_forward_residual_factor (float): output_weight for feed-forward residual connections
      feed_forward_expansion_factor (int): Expansion factor for feed-forward block
      num_heads (int): Number of heads to use for multi-head attention
      positional_encoder (nn.Module): PositionalEncoder module
      dropout (float): Dropout probability

    Inputs:
      x (Tensor): (batch_size, time, embed_dim)
      mask (Tensor): (batch_size, time, time) Optional mask to zero out attention score at certain indices

    Outputs:
      Tensor (batch_size, time, embed_dim): Output tensor from the conformer block.

    """

    def __init__(
        self,
        embed_dim=144,
        conv_kernel_size=31,
        feed_forward_residual_factor=0.5,
        feed_forward_expansion_factor=4,
        num_heads=4,
        dropout=0.1,
        use_deepnorm=False,
        alpha=0.1,
        beta=0.1,
        use_rope=False,
    ):
        super(ConformerBlock, self).__init__()
        self.feed_forward_residual_factor = feed_forward_residual_factor
        self.use_deepnorm = use_deepnorm
        self.alpha = alpha
        self.beta = beta
        self.use_rope = use_rope

        self.ff1 = FeedForwardBlock(embed_dim, feed_forward_expansion_factor, dropout)
        self.attention = MHAPyTorchScaledDotProduct(
            d_in=embed_dim,
            d_out=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
        )
        self.conv_block = ConvBlock(embed_dim, conv_kernel_size, dropout)
        self.ff2 = FeedForwardBlock(embed_dim, feed_forward_expansion_factor, dropout)
        if use_deepnorm:
            self.norm1 = DeepNorm(alpha, embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        else:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        with torch.no_grad():
            # Initialize linear projections of MLP
            self.ff1.module[1].weight *= beta
            self.ff1.module[4].weight *= beta
            self.ff2.module[1].weight *= beta
            self.ff2.module[4].weight *= beta
            # Initialize only values projection in self.attn
            # Separate weights for q, k, v
            qkv_weight = self.attention.qkv.weight.view(
                3, self.attention.d_out, self.attention.d_in
            )
            # Apply beta to value weights (third set of weights)
            qkv_weight[2] *= beta
            # Initialize output projection of attention
            self.attention.proj.weight *= beta

    def forward(self, x):
        # Apply first feedforward block
        x = x + (self.feed_forward_residual_factor * self.ff1(x))
        # Apply attention block with DeepNorm
        if self.use_deepnorm:
            gx = self.attention(x)
            x = self.norm1(x, gx)
        else:
            x = x + self.attention(x)
        # Apply convolution block
        x = x + self.conv_block(x)
        # Apply second feedforward block
        x = x + (self.feed_forward_residual_factor * self.ff2(x))
        # Final normalization
        return self.norm2(x)

    # Original conformer forward code
    # def forward(self, x, mask=None):
    #    x = x + (self.feed_forward_residual_factor * self.ff1(x))
    #    x = x + self.positional_encoder(x.size(1))
    #    x = x + self.attention(x, mask=mask)
    #    x = x + self.conv_block(x)
    #    x = x + (self.feed_forward_residual_factor * self.ff2(x))
    # return self.layer_norm(x)


@gin.configurable
class Conformer(nn.Module):
    """
    Conformer Encoder Module.

    Parameters:
      d_input (int): Dimension of the input
      embed_dim (int): Dimension of the model
      num_layers (int): Number of conformer blocks to use in the encoder
      conv_kernel_size (int): Size of kernel to use for depthwise convolution
      feed_forward_residual_factor (float): output_weight for feed-forward residual connections
      feed_forward_expansion_factor (int): Expansion factor for feed-forward block
      num_heads (int): Number of heads to use for multi-head attention
      dropout (float): Dropout probability

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
        conv_kernel_size: int,
        mlp_ratio: int,
        mlp_residual_factor: float,
        num_heads: int,
        dropout: float,
        input_dropout: float,
        alpha_deepnorm: float,
        beta_deepnorm: float,
        use_deepnorm: bool,
        use_rope: bool,
    ):
        super(Conformer, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.feed_forward_expansion_factor = mlp_ratio
        self.feed_forward_residual_factor = mlp_residual_factor
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha_deepnorm = alpha_deepnorm
        self.beta_deepnorm = beta_deepnorm
        self.use_deepnorm = use_deepnorm
        self.use_rope = use_rope

        self.input_dropout = nn.Dropout(input_dropout)

        # define global positional encoder to limit model parameters
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    embed_dim=self.embed_dim,
                    conv_kernel_size=self.conv_kernel_size,
                    feed_forward_expansion_factor=self.feed_forward_expansion_factor,
                    feed_forward_residual_factor=self.feed_forward_residual_factor,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    use_deepnorm=self.use_deepnorm,
                    alpha=self.alpha_deepnorm,
                    beta=self.beta_deepnorm,
                    use_rope=self.use_rope,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x,
        layers: Set[int] = set([-1]),
    ):
        # Downsampling (preprocessing) -> patching and projection layer (model masking) -> here
        x = self.input_dropout(x)

        # Convert to positive indices
        l_avail = list(range(len(self.layers)))
        layers = set([l_avail[l] for l in list(layers)])

        results = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in layers:
                results.append(x)

        if len(results) == 1:
            return results[0]
        else:
            return x
