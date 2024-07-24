import math
import pdb

import gin
import torch
from torch import nn, einsum
from einops import rearrange
import pytorch_lightning as L


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
        # Set to evaluation mode
        self.eval()
        # Apply random projection
        x = einsum("b n d, d e -> b n e", x, self.random_projection)
        # Perform codebook lookup
        xq = self.codebook_lookup(x)
        return xq

@gin.configurable
class MaskingModel(L.LightningModule):
    """
    MaskingModel

    This model is used to train a model with a masking laguage modelling mechanism.
    net is the model that will be trained
    representation is the module that will be used to extract the features
    patch_frames is the number of frames that will be used to create a patch (default 16 x 16)
    num_codebooks is the number of codebooks that will be used (default 1)
    codebook_size is the size of the codebook (default 4096)
    mask_seconds is the number of seconds that will be masked (default 0.4)
    mask_prob is the probability that a mask will be applied (default 0.6)
    """

    def __init__(
        self,
        net: nn.Module,
        representation: nn.Module,
        patch_frames=16,
        num_codebooks=1,
        codebook_size=4096,
        mask_seconds=0.4,
        mask_prob=0.6,
    ):
        super(MaskingModel, self).__init__()

        # global variables
        self.mask_seconds = mask_seconds
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.patch_frames = patch_frames

        self.net = net
        self.representation = representation
        # pdb.set_trace()
        self.linear = nn.Linear(self.net.head.out_features, codebook_size)
        # TODO the representation model is compatible with other representations
        self.sr = representation.sr
        self.hop_length = representation.hop_len
        self.n_mel = representation.n_mel
        self.rproj_input_dim = patch_frames*patch_frames
        # random quantizer
        seed = 142
        self.codebook = RandomProjectionQuantizer(
            self.rproj_input_dim, patch_frames, codebook_size, seed=seed
        )

        # loss function
        self.loss = nn.CrossEntropyLoss()

    def pad_spectrogram(self, spectrogram, patch_size=16):
        B, F, T = spectrogram.shape

        # Calculate padding sizes
        pad_f = (patch_size - F % patch_size) % patch_size
        pad_t = (patch_size - T % patch_size) % patch_size

        # Apply padding (only on F and T dimensions)
        padded_spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_t, 0, pad_f), mode='constant', value=0)

        return padded_spectrogram

    def vit_tokenization(self, spectrogram, patch_size=16):
        B, F, T = spectrogram.shape

        # Number of patches
        num_patches_f = F // patch_size
        num_patches_t = T // patch_size

        # Reshape the spectrogram to (B, num_patches_f, patch_size, num_patches_t, patch_size)
        patches = spectrogram.reshape(B, num_patches_f, patch_size, num_patches_t, patch_size)

        # Move the patch dimensions next to each other and then flatten the patches
        patches = patches.permute(0, 1, 3, 2, 4).reshape(B, num_patches_f * num_patches_t, patch_size * patch_size)
        tokens = self.codebook(patches)
        return patches, tokens

    def random_masking(self, tokens, mask_ratio=0.5):
        B, num_patches, patch_size = tokens.shape
        num_masked = int(mask_ratio * num_patches)

        # Generate random mask indices
        mask_indices = torch.rand(B, num_patches).argsort(dim=1)[:, :num_masked]

        # Create a mask array with the same shape as tokens, initialized to False
        mask = torch.zeros(B, num_patches, dtype=torch.bool)

        # Use advanced indexing to set the mask indices to True
        mask[torch.arange(B).unsqueeze(1), mask_indices] = True

        # Apply the mask to the tokens
        masked_tokens = tokens.clone()
        masked_tokens[mask] = 0  # Here, 0 can be replaced with any mask value or token

        return masked_tokens.to(tokens.device), mask.to(tokens.device)

    # def masking_raw_audio(self, x):
    #     """random masking of 400ms with given probability"""
    #     mx = x.clone()
    #     b, f, t = mx.shape
    #
    #     len_masking_spec_frames = math.ceil(self.mask_seconds * self.sr / self.hop_length)
    #     len_masking_spec_tokens = math.ceil(len_masking_spec_frames / self.patch_frames)
    #
    #     # get random mask indices
    #     start_indices = torch.rand(b, t // len_masking_spec_frames) < self.mask_prob
    #     time_domain_masked_indices = torch.nonzero(
    #         start_indices.repeat_interleave(len_masking_spec_frames, dim=1)
    #     )
    #     token_domain_masked_indices = torch.nonzero(
    #         start_indices.repeat_interleave(len_masking_spec_tokens, dim=1)
    #     )
    #     # trim
    #     time_domain_masked_indices = time_domain_masked_indices[:t].transpose(0, 1)
    #     token_domain_masked_indices = token_domain_masked_indices[:t//self.patch_frames].transpose(0, 1)
    #
    #     # mask with random values
    #     mx = mx.transpose(1, 2)
    #     masking_noise = (
    #         torch.randn(mx.shape, dtype=x.dtype) * 0.1
    #     )  # 0 mean 0.1 std
    #     # Ensure the indices are in the right format
    #     batch_indices = torch.arange(x.size(0)).unsqueeze(1).expand_as(time_domain_masked_indices)
    #     # Apply masking in parallel
    #     mx[batch_indices, time_domain_masked_indices, :] = masking_noise[batch_indices, time_domain_masked_indices, :].to(device=x.device)
    #     mx = mx.transpose(1, 2)
    #     return mx, token_domain_masked_indices

    # @torch.no_grad()
    # def rearrange(self, x):
    #     return rearrange(x, "b f (t s) -> b t (s f)", s=self.patch_frames)

    # @torch.no_grad()
    # def tokenize(self, x):
    #     # TODO if more than one codebook modify here
    #     layer = getattr(self, "quantizer_mel_0")
    #     return layer(x)
    #
    # def get_targets(self, x):
    #     x = self.rearrange(x)
    #     target_tokens = self.tokenize(x)
    #     return target_tokens

    def get_loss(self, logits, target_tokens, mask):
        losses = {}
        accuracies = {}
        # remove cls first token from logits
        logits = {key: logit_out[:, 1:] for key, logit_out in logits.items()}
        for key in logits.keys():
            logit_out = logits[key]
            # zeros boolean with the shape of logit_out
            masked_logits = logit_out[mask]
            masked_tokens = target_tokens[key][mask]
            losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
        return losses, accuracies

    def forward(self, x):
        x = self.representation(x[0])
        # get target feature tokens
        x, target_tokens = self.vit_tokenization(x)

        # masking
        x, mask = self.random_masking(x)

        x = self.net(x)

        # forward q
        logits = self.linear(x)
        logits = {"spectrogram": logits}
        target_tokens = {"spectrogram": target_tokens}

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, mask)

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
        self.log('val_loss', loss, prog_bar=True, on_step=True)
        for key in accuracies:
            self.log(f'val_acc_{key}', accuracies[key])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer









