import math
import pdb
from collections import Counter

import gin
import torch
from torch import nn
import pytorch_lightning as L

from modules.codebooks import RandomProjectionQuantizer


@gin.configurable
class MaskingModel(L.LightningModule):
    """
    MaskingModel
    inspired https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/model/musicfm_25hz.py#L125

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
        lr: float,
        representation: nn.Module,
        num_codebooks: int,
        codebook_size: int,
        mask_seconds: float,
        mask_prob: float,
    ):
        super(MaskingModel, self).__init__()

        # global variables
        self.mask_seconds = mask_seconds
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.patch_size = net.patch_size
        self.net = net
        self.representation = representation
        self.embedding_layer = nn.Linear(self.patch_size[0]*self.patch_size[1], self.net.head.out_features)
        self.linear = nn.Linear(self.net.head.out_features, codebook_size)
        self.lr = lr
        self.tokens_coverage = Counter()

        if hasattr(representation, "sr") and hasattr(representation, "hop_len") and hasattr(representation, "n_mel"):
            self.sr = representation.sr
            self.hop_length = representation.hop_len
            self.n_mel = representation.n_mel
            # random quantizer
            self.codebook = RandomProjectionQuantizer(
                input_dim=self.patch_size[0]*self.patch_size[1],
                codebook_dim=self.net.head.out_features,
                codebook_size=codebook_size,
            )
        else:
            raise NotImplementedError(f"Representation {type(self.representation)} is supported")

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

    def vit_tokenization(self, spectrogram):
        B, F, T = spectrogram.shape
        # Number of patches
        num_patches_f = F // self.patch_size[0]
        num_patches_t = T // self.patch_size[1]
        # Reshape spectrogram into patches
        patches = spectrogram.unfold(1, self.patch_size[0], self.patch_size[0])
        patches = patches.unfold(2, self.patch_size[1], self.patch_size[1])
        # Reshape to (B, num_patches_f * num_patches_t, patch_frames_f, patch_frames_t)
        patches = patches.contiguous().view(B, num_patches_f * num_patches_t, self.patch_size[0], self.patch_size[1])
        # Flatten patches to tokens
        patches = patches.view(B, num_patches_f * num_patches_t, -1)
        # Return patches and tokens
        tokens = self.codebook(patches)
        return patches, tokens

    def random_masking_simple(self, spectrogram):
        B, num_patches, patch_size = spectrogram.shape
        num_masked = int(self.mask_prob * num_patches)
        # we have a windows_random
        # Generate random mask indices
        mask_indices = torch.rand(B, num_patches).argsort(dim=1)[:, :num_masked]
        # Create a mask array with the same shape as tokens, initialized to False
        mask = torch.zeros(B, num_patches, dtype=torch.bool)
        mask[torch.arange(B).unsqueeze(1), mask_indices] = True
        masked_spec = spectrogram.clone()
        masking_noise = torch.randn_like(masked_spec) * 0.1
        masked_spec[mask] = masking_noise[mask]
        return masked_spec, mask.to(spectrogram.device)

    def random_masking(self, patches):
        B, num_patches, patch_size = patches.shape
        mx = patches.clone()

        len_masking_spec_frames = math.ceil(self.mask_seconds * self.sr / self.hop_length)
        windows_tokens = len_masking_spec_frames // self.patch_size[0] * (self.n_mel // self.patch_size[1])

        # Generate random mask indices
        start_indices = torch.rand(B, math.ceil(num_patches / windows_tokens)) < self.mask_prob
        mask = start_indices.repeat_interleave(windows_tokens, dim=1)

        # Trim mask to fit the number of patches
        if mask.size(1) > num_patches:
            mask = mask[:, :num_patches]

        # Mask with random values
        masking_noise = (torch.randn(mx.shape, dtype=patches.dtype) * 0.1).to(patches.device)  # 0 mean 0.1 std
        # Apply masking in parallel
        mx[mask] = masking_noise[mask]
        return mx, mask.to(patches.device)

    def get_loss(self, logits, target_tokens, mask):
        # zeros boolean with the shape of logit_out
        masked_logits = logits[mask]
        masked_tokens = target_tokens[mask]
        losses = self.loss(masked_logits, masked_tokens)
        accuracies = (
            torch.sum(masked_logits.argmax(-1) == masked_tokens)
            / masked_tokens.numel()
        )
        return losses, accuracies

    def forward(self, x):
        x = self.representation(x[0])
        # get target feature tokens
        x, target_tokens = self.vit_tokenization(x) # B x t x (16 x 4)
        # masking
        x, mask = self.random_masking(x)
        x = self.embedding_layer(x)
        x = self.net(x)
        logits = self.linear(x)
        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, mask)
        self.log()
        return logits, losses, accuracies

    def training_step(self, batch, batch_idx):
        x = batch
        logits, loss, accuracies = self.forward(x)
        self.log('train_loss', loss, prog_bar=True)
        self.log(f'train_acc', accuracies)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        logits, loss, accuracies = self.forward(x)
        self.log('val_loss', loss, prog_bar=True)
        self.log(f'val_acc', accuracies)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
