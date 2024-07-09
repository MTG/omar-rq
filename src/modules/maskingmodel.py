import gin
import torch
from torch import nn, einsum
from einops import rearrange
import lightning as L


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
        b = x.shape[0]
        x = rearrange(x, "b n e -> (b n) e")

        # L2 normalization
        normalized_x = nn.functional.normalize(x, dim=1, p=2)
        normalized_codebook = nn.functional.normalize(self.codebook, dim=1, p=2)

        # compute distances
        distances = torch.cdist(normalized_codebook, normalized_x)

        # get nearest
        nearest_indices = torch.argmin(distances, dim=0)

        # reshape
        xq = rearrange(nearest_indices, "(b n) -> b n", b=b)

        return xq

    @torch.no_grad()
    def forward(self, x):
        # always eval
        self.eval()

        # random projection [batch, length, input_dim] -> [batch, length, codebook_dim]
        x = einsum("b n d, d e -> b n e", x, self.random_projection)

        # codebook lookup
        xq = self.codebook_lookup(x)

        return xq

@gin.configurable
class maskingmodel(L.LightningModule):
    """
    MusicFM

    Input: 128-band mel spectrogram
    Frontend: 2-layer Residual convolution
    Backend: 12-layer Conformer
    Quantizer: a codebook for mel spectrogram
    """

    def __init__(
        self,
        net: nn.Module,
        num_codebooks=1,
        codebook_dim=16,
        codebook_size=4096,
        hop_length=240,
        n_mels=128,
        mask_hop=0.4,
        mask_prob=0.6,
    ):
        super(MaskingModel, self).__init__()

        # global variables
        self.hop_length = hop_length
        self.mask_hop = mask_hop
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.net = net

        self.linear = nn.Linear(self.net.d_out, codebook_size)
        # random quantizer
        seed = 142
        for i in range(num_codebooks):
            setattr(
                self,
                f"quantizer_mel_{i}",
                RandomProjectionQuantizer(
                    n_mels * 4, codebook_dim, codebook_size, seed=seed + i
                ),
            )
        # loss function
        self.loss = nn.CrossEntropyLoss()

    def masking(self, x):
        """random masking of 400ms with given probability"""
        mx = x.clone()
        b, t = mx.shape
        len_masking_raw = int(24000 * self.mask_hop)
        len_masking_token = int(24000 / self.hop_length / 2 / 2 * self.mask_hop)

        # get random mask indices
        start_indices = torch.rand(b, t // len_masking_raw) < self.mask_prob
        time_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_raw, dim=1)
        )
        token_domain_masked_indices = torch.nonzero(
            start_indices.repeat_interleave(len_masking_token, dim=1)
        )

        # mask with random values
        masking_noise = (
            torch.randn(time_domain_masked_indices.shape[0], dtype=x.dtype) * 0.1
        )  # 0 mean 0.1 std
        mx[tuple(time_domain_masked_indices.t())] = masking_noise.to(x.device)

        return mx, token_domain_masked_indices


    @torch.no_grad()
    def rearrange(self, x):

        return rearrange(x, "b f (t s) -> b t (s f)", s=4)

    @torch.no_grad()
    def tokenize(self, x):
        # TODO if more than one codebook modify here
        layer = getattr(self, "quantizer_mel_0")
        return layer(x)

    def get_targets(self, x):
        x = self.rearrange(x)
        target_tokens = self.tokenize(x)
        return target_tokens

    def get_loss(self, logits, target_tokens, masked_indices):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            masked_logits = logits[key][tuple(masked_indices.t())]
            masked_tokens = target_tokens[key][tuple(masked_indices.t())]
            losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
        return losses, accuracies

    def forward(self, x):
        # get target feature tokens
        target_tokens = self.get_targets(x)

        # masking
        x, masked_indices = self.masking(x)

        # forward
        logits = self.linear(x)
        logits = {
            key: logits[:, :, i * self.codebook_size: (i + 1) * self.codebook_size]
            for i, key in enumerate(self.features)
        }

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, masked_indices)

        return logits, losses, accuracies

    def training_step(self, batch, batch_idx):
        x = batch
        logits, losses, accuracies = self.forward(x)
        loss = sum(losses.values())
        self.log('train_loss', loss)
        for key in accuracies:
            self.log(f'train_acc_{key}', accuracies[key])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        logits, losses, accuracies = self.forward(x)
        loss = sum(losses.values())
        self.log('val_loss', loss)
        for key in accuracies:
            self.log(f'val_acc_{key}', accuracies[key])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer





# Test the MaskingModel
def test_masking_model():
    class SimpleNet(nn.Module):
        def __init__(self, d_out=512):
            super(SimpleNet, self).__init__()
            self.d_out = d_out
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc = nn.Linear(64 * 32 * 32, d_out)  # Adjust according to input size

        def forward(self, x):
            x = nn.functional.relu(self.conv1(x))
            x = nn.functional.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    # Instantiate the SimpleNet and MaskingModel
    net = SimpleNet()
    model = MaskingModel(net=net, num_codebooks=1, codebook_dim=16, codebook_size=4096, n_mels=128)

    # Generate some random input data
    batch_size = 4
    n_mels = 128
    seq_len = 240  # example sequence length for mel spectrograms
    input_data = torch.randn(batch_size, n_mels, seq_len)

    # Run the forward pass
    logits, losses, accuracies = model(input_data)

    # Print the outputs
    print("Logits:", logits)
    print("Losses:", losses)
    print("Accuracies:", accuracies)


# Run the test
if __name__ == "__main__":
    test_masking_model()







