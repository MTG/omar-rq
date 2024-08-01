import torch
import torch.nn as nn
import gin.torch

from .net import Net


class PatchEmbed(nn.Module):
    """Split spectrogram into patches and embed them."""

    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, F, T) -> (B, E, N)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.proj(x)  # (B, E, H, W)
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x


class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(
        self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False
    ):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_in, d_out)
        self.dropout = dropout

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

        use_dropout = 0.0 if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True
        )

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_tokens, self.d_out)
        )

        context_vec = self.proj(context_vec)

        return context_vec


class TransformerEncoder(nn.Module):
    """Transformer Encoder Block with Multihead Attention"""

    def __init__(
        self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, context_length=1850
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MHAPyTorchScaledDotProduct(
            embed_dim,
            embed_dim,
            num_heads,
            dropout=dropout,
            context_length=context_length,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # First normalization layer
        x = self.norm1(x)

        # Multihead attention layer
        attn_output = self.attn(x)
        x = attn_output + x  # Skip connection

        # Second normalization layer
        x = self.norm2(x)

        # Feed forward layer
        x = self.mlp(x) + x  # Skip connection

        return x


@gin.configurable
class Transformer(Net):
    """Vision Transformer with adaptations for audio spectrogram."""

    def __init__(
        self,
        patch_size,
        context_length,
        in_chans,
        embed_dim,
        head_dims,
        depth,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        do_classification=False,
        do_vit_tokenization=False
    ):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.context_length = context_length

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        if self.do_classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Initial positional embeddings (dynamically resized later)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList(
            [
                TransformerEncoder(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    context_length=context_length,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, head_dims)
        self.do_classification = do_classification
        self.do_vit_tokenization = do_vit_tokenization

    def forward(self, x):
        if self.do_vit_tokenization:
            x = self.patch_embed(x)  # Embed the patches
        B, N, _ = x.shape

        if self.do_classification:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1) # Add class token

        # Ensure positional embeddings cover the entire sequence length
        if x.size(1) > self.pos_embed.size(1):
            # Extend pos_embed to match the sequence length
            new_pos_embed = torch.zeros(
                1, x.size(1), self.embed_dim, device=self.pos_embed.device
            )
            new_pos_embed[:, : self.pos_embed.size(1)] = self.pos_embed
            self.pos_embed = nn.Parameter(new_pos_embed)

        x = x + self.pos_embed
        x = self.dropout(x)

        for layer in self.transformer:
            x = layer(x)

        if self.do_classification:
            x = self.norm(x)
            x = x[:, 0]  # Extract the class token
            x = self.head(x)
        return x

    # class VisionTransformerTiny(VisionTransformer):
    #     """Tiny Vision Transformer for testing purposes."""

    # configurations = {
    #     "12_layers": {
    #         "embed_dim": 768,
    #         "depth": 12,
    #         "num_heads": 12,
    #         "head_dims": 500,
    #     },
    #     "24_layers": {
    #         "embed_dim": 1024,
    #         "depth": 24,
    #         "num_heads": 16,
    #         "head_dims": 500,
    #     },
    #     "36_layers": {
    #         "embed_dim": 1200,
    #         "depth": 36,
    #         "num_heads": 20,
    #         "head_dims": 500,
    #     },
    #     "48_layers": {
    #         "embed_dim": 1440,
    #         "depth": 48,
    #         "num_heads": 24,
    #         "head_dims": 500,
    #     },
    # }
