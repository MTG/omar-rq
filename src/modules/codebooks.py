import torch

from torch import nn, einsum


class RandomProjectionQuantizer(nn.Module):
    """
    Random projection and codebook lookup module

    Some code is borrowed from:
     https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/random_projection_quantizer.py
    But I did normalization using pre-computed global mean & variance instead of using layer norm.
    """
    def __init__(
        self,
        input_dim,
        codebook_dim,
        codebook_size,
        seed=142,
    ):
        super().__init__()

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.input_dim = input_dim

        # random seed
        torch.manual_seed(seed)

        # randomly initialized projection
        random_projection = torch.empty(input_dim, codebook_dim)
        nn.init.xavier_normal_(random_projection)
        self.register_buffer("random_projection", random_projection)

        # randomly initialized codebook
        codebook = torch.empty(codebook_size, codebook_dim)
        nn.init.normal_(codebook)
        self.register_buffer("codebook", codebook)

    def codebook_lookup(self, x):
        # reshape
        b, n, e = x.shape
        x = x.view(b * n, e)

        # L2 normalization
        normalized_x = nn.functional.normalize(x, dim=1, p=2)
        normalized_codebook = nn.functional.normalize(self.codebook, dim=1, p=2)

        # compute distances
        distances = torch.cdist(normalized_codebook, normalized_x)

        # get nearest
        nearest_indices = torch.argmin(distances, dim=0)

        # reshape
        xq = nearest_indices.view(b, n)
        return xq

    @torch.no_grad()
    def forward(self, x):
        # Set to evaluation mode
        self.eval()
        # Apply random projection
        x = einsum("b n d, d e -> b n e", x, self.random_projection)
        # Perform codebook lookup
        xq = self.codebook_lookup(x)
        return xq