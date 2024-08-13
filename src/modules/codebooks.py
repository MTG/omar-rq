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


class Codebook(nn.Module):
    def __init__(self, num_codes, code_dim):
        """
        Initializes the Codebook with the specified number of codes and code dimensionality.

        Parameters:
        -----------
        num_codes : int
            The number of codes in the codebook.
        code_dim : int
            The dimensionality of each code.
        """
        super(Codebook, self).__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.num_codes = num_codes
        self.code_dim = code_dim

    def forward(self, x):
        """
        Quantizes the input tensor `x` by finding the nearest code in the codebook.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor with shape (batch_size, num_patches, code_dim).

        Returns:
        --------
        quantized : torch.Tensor
            The quantized tensor with the same shape as `x`.
        codes : torch.Tensor
            The indices of the codes in the codebook that are closest to each patch in the input tensor.
        """
        # Flatten the input tensor to shape (batch_size * num_patches, code_dim)
        B, _, _ = x.shape
        flattened = x.view(-1, self.code_dim)

        # Compute the distances between the flattened input and the codebook embeddings
        distances = (
            torch.sum(flattened**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(flattened, self.codebook.weight.t())
        )

        # Find the indices of the closest codes in the codebook
        codes = torch.argmin(distances, dim=1)
        codes = codes.view(B, -1)

        # Retrieve the quantized embeddings corresponding to the closest codes
        quantized = self.codebook(codes).view(x.shape)

        return codes
