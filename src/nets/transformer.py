from typing import Union, List

import torch
import torch.nn as nn
import gin.torch

from .common_former import MHAPyTorchScaledDotProduct, DeepNorm
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


class TransformerEncoder(nn.Module):
    """Transformer Encoder Block with Multihead Attention and optional deepnorm"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        use_deepnorm=False,
        alpha=0.1,
        beta=0.1,
    ):
        super().__init__()

        # Initialize norm layers based on the use_deepnorm flag
        self.use_deepnorm = use_deepnorm
        self.alpha = alpha

        if use_deepnorm:
            self.norm1 = DeepNorm(alpha, embed_dim)
            self.norm2 = DeepNorm(alpha, embed_dim)
        else:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        self.attn = MHAPyTorchScaledDotProduct(
            embed_dim, embed_dim, num_heads, dropout=dropout
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            # Initialize linear projections of MLP
            self.mlp[0].weight *= beta
            self.mlp[3].weight *= beta
            # Initialize only values projection in self.attn
            # Separate weights for q, k, v
            qkv_weight = self.attn.qkv.weight.view(3, self.attn.d_out, self.attn.d_in)
            # Apply beta to value weights (third set of weights)
            qkv_weight[2] *= beta
            # Initialize output projection of attention
            self.attn.proj.weight *= beta

    def forward(self, x):
        # Apply the first normalization
        if self.use_deepnorm:
            gx = self.attn(x)
            x = self.norm1(x, gx)
        else:
            x = self.norm1(x)
            # Multihead attention layer
            attn_output = self.attn(x)
            x = attn_output + x  # Skip connection
        # Apply the second normalization
        if self.use_deepnorm:
            gx = self.mlp(x)
            x = self.norm2(x, gx)
        else:
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
        in_chans,
        embed_dim,
        head_dims,
        depth,
        num_heads,
        num_patches=1,
        mlp_ratio=4.0,
        dropout=0.1,
        alpha_deepnorm=0.1,
        beta_deepnorm=0.1,
        do_classification=False,
        do_vit_tokenization=False,
        do_deepnorm=False,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.do_classification = do_classification
        self.do_vit_tokenization = do_vit_tokenization
        self.do_deepnorm = do_deepnorm
        self.alpha_deepnorm = alpha_deepnorm
        self.beta_deepnorm = beta_deepnorm

        if self.do_vit_tokenization:
            self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        else:
            self.patch_embed = nn.Identity()
        if self.do_classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Initial positional embeddings (dynamically resized later)
        # During initialization use a dynamic patch size. This value will be
        # Updated and stored in self.num_patches during the forward pass
        # It will be written to the gin config file
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList(
            [
                TransformerEncoder(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    use_deepnorm=self.do_deepnorm,
                    beta=self.beta_deepnorm,
                    alpha=self.alpha_deepnorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, head_dims)

    def forward(self, x):

        # Embed the patches. If do_vit_tokenization is False, the input
        # is already tokenized and the patch_embed layer is an identity
        x = self.patch_embed(x)

        B, _, _ = x.shape

        if self.do_classification:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)  # Add class token

        # Ensure positional embeddings cover the entire sequence length
        if x.size(1) > self.pos_embed.size(1):
            # Extend pos_embed to match the sequence length
            new_pos_embed = torch.zeros(
                1, x.size(1), self.embed_dim, device=self.pos_embed.device
            )
            new_pos_embed[:, : self.pos_embed.size(1)] = self.pos_embed
            self.num_patches = x.size(1)  # Update the number of patches
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
