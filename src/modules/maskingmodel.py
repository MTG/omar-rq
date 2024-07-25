

import gin
import torch
from torch import nn
import pytorch_lightning as L

from src.modules.codebooks import Codebook
from src.nets.transformer import PatchEmbed


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

        if hasattr(representation, "sr") and hasattr(representation, "hop_len") and hasattr(representation, "n_mel"):
            self.sr = representation.sr
            self.hop_length = representation.hop_len
            self.n_mel = representation.n_mel
            # random quantizer
            self.codebook = Codebook(
                self.codebook_size, self.patch_size[0] * self.patch_size[1]
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

    def random_masking(self, spectrogram):
        B, num_patches, patch_size = spectrogram.shape
        num_masked = int(self.mask_prob * num_patches)
        # Generate random mask indices
        mask_indices = torch.rand(B, num_patches).argsort(dim=1)[:, :num_masked]
        # Create a mask array with the same shape as tokens, initialized to False
        mask = torch.zeros(B, num_patches, dtype=torch.bool)
        mask[torch.arange(B).unsqueeze(1), mask_indices] = True
        masked_spec = spectrogram.clone()
        masking_noise = torch.randn_like(masked_spec) * 0.1
        masked_spec[mask] = masking_noise[mask]
        return masked_spec, mask.to(spectrogram.device)

    # THIS CODE IS CLEANED FOR IMPLEMENTING THE SAME BUT DIRECTLY FROM AUDIO
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
        # remove cls first token from logits
        logits = logits[:, 1:]
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
        x, target_tokens = self.vit_tokenization(x)
        # masking
        x, mask = self.random_masking(x)
        x = self.embedding_layer(x)
        x = self.net(x)
        logits = self.linear(x)
        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, mask)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
