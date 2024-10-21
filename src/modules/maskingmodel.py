import math
import os
import random
from collections import defaultdict
from typing import Set

import gin
import torch
import wandb
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
        weight_decay: float,
        representation: nn.Module,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        mask_seconds: float,
        mask_prob: float,
        seed: int,
        diff_input: bool,
        plot_tokens: bool = False,
        input_representation: nn.Module | None = None,
    ):
        super(MaskingModel, self).__init__()

        # global variables
        self.mask_seconds = mask_seconds
        self.mask_prob = mask_prob
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.net = net
        self.representation = representation
        self.lr = lr
        self.seed = seed
        self.plot_tokens = plot_tokens
        self.weight_decay = weight_decay
        self.tokens_coverage = []
        self.first_coverage = True
        self.diff_input = diff_input
        self.input_representation = input_representation

        # downstream evaluation params
        self.downstream_embedding_layer = set([-1])
        self.overlap_ratio = 0.5

        n_reps = 1
        if isinstance(representation, nn.ModuleList):
            n_reps = len(representation)
        self.n_reps = n_reps

        self.linear = nn.Linear(
            self.net.embed_dim, codebook_size * num_codebooks * n_reps
        )

        # debugging variables
        self.tokens_accumulator = defaultdict(list)

        if isinstance(representation, nn.ModuleList):
            self.quantizers = nn.ModuleList()

            for rep in representation:
                if (
                    hasattr(rep, "sr")
                    and hasattr(rep, "hop_len")
                    and hasattr(rep, "rep_dims")
                ):
                    if isinstance(rep, self.input_representation):
                        self.sr = rep.sr
                        self.hop_length = rep.hop_len
                        self.rep_dims = rep.rep_dims
                        self.patch_size = rep.patch_size

                    # Create a ModuleList to hold the quantizers
                    self.quantizers.append(
                        nn.ModuleList(
                            [
                                RandomProjectionQuantizer(
                                    rep.patch_size[0] * rep.patch_size[1],
                                    codebook_dim,
                                    codebook_size,
                                    seed=seed + i,
                                    diff_input=self.diff_input,
                                )
                                for i in range(num_codebooks)
                            ]
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Representation {type(self.representation)} shuold have sr, hop_len and rep_dims attributes."
                    )
        else:
            if (
                hasattr(representation, "sr")
                and hasattr(representation, "hop_len")
                and hasattr(representation, "rep_dims")
            ):
                self.sr = representation.sr
                self.hop_length = representation.hop_len
                self.rep_dims = representation.rep_dims
                # Create a ModuleList to hold the quantizers
                self.quantizers = nn.ModuleList(
                    [
                        RandomProjectionQuantizer(
                            self.patch_size[0] * self.patch_size[1],
                            codebook_dim,
                            codebook_size,
                            seed=seed + i,
                            diff_input=self.diff_input,
                        )
                        for i in range(num_codebooks)
                    ]
                )
            else:
                raise NotImplementedError(
                    f"Representation {type(self.representation)} shuold have sr, hop_len and rep_dims attributes."
                )

        # aux nets
        self.embedding_layer = nn.Linear(
            self.patch_size[0] * self.patch_size[1], self.net.embed_dim
        )

        # loss function
        self.loss = nn.CrossEntropyLoss()

    def pad_spectrogram(self, spectrogram, patch_size=16):
        B, F, T = spectrogram.shape

        # Calculate padding sizes
        pad_f = (patch_size - F % patch_size) % patch_size
        pad_t = (patch_size - T % patch_size) % patch_size

        # Apply padding (only on F and T dimensions)
        padded_spectrogram = torch.nn.functional.pad(
            spectrogram, (0, pad_t, 0, pad_f), mode="constant", value=0
        )
        return padded_spectrogram

    def plot_spectrogram_with_tokens(
        self, spectrogram, num_patches_f, num_patches_t, tokens
    ):
        from matplotlib import pyplot as plt

        plt.figure(figsize=(36, 8), dpi=300)

        plt.imshow(spectrogram, aspect="auto", cmap="viridis", origin="lower")
        plt.colorbar(label="Magnitude")
        plt.title("Spectrogram with Token Numbers")
        plt.xlabel("Time")
        plt.ylabel("Frequency")

        token_index = 0
        for i in range(num_patches_f):
            for j in range(num_patches_t):
                # Calculate the patch boundaries
                start_f = i * self.patch_size[0]
                end_f = start_f + self.patch_size[0]
                start_t = j * self.patch_size[1]
                end_t = start_t + self.patch_size[1]

                # Draw the patch boundary
                plt.plot(
                    [start_t, start_t], [start_f, end_f], color="white", linewidth=1
                )
                plt.plot([end_t, end_t], [start_f, end_f], color="white", linewidth=1)
                plt.plot(
                    [start_t, end_t], [start_f, start_f], color="white", linewidth=1
                )
                plt.plot([start_t, end_t], [end_f, end_f], color="white", linewidth=1)

                # Place the token number in the center of each patch
                center_t = (start_t + end_t) / 2
                center_f = (start_f + end_f) / 2
                plt.text(
                    center_t,
                    center_f,
                    str(tokens[token_index].item()),
                    color="red",
                    fontsize=8,
                    ha="center",
                    va="center",
                    rotation=90,
                )
                token_index += 1
        # save the plot in ../figs as pdf
        randint = random.randint(0, 100000)
        if not os.path.exists("../figs"):
            os.makedirs("figs")
        # save pdf
        plt.savefig(f"figs/spectrogram_with_tokens_{randint}.pdf")
        plt.close()

    def vit_tokenization(self, spectrogram, quantizers, rep):
        B, F, T = spectrogram.shape
        num_patches_f = F // rep.patch_size[0]
        num_patches_t = T // rep.patch_size[1]
        # Reshape spectrogram into patches
        patches = spectrogram.unfold(1, rep.patch_size[0], rep.patch_size[0])
        patches = patches.unfold(2, rep.patch_size[1], rep.patch_size[1])
        # Reshape to (B, num_patches_f * num_patches_t, patch_frames_f, patch_frames_t)
        patches = patches.contiguous().view(
            B, num_patches_f * num_patches_t, rep.patch_size[0], rep.patch_size[1]
        )
        # Flatten patches to tokens
        patches = patches.view(B, num_patches_f * num_patches_t, -1)
        # Return patches and tokens
        tokens = torch.stack([quantizer(patches) for quantizer in quantizers], dim=-1)
        if self.plot_tokens:
            self.plot_spectrogram_with_tokens(
                spectrogram[0].detach().cpu(),
                num_patches_f,
                num_patches_t,
                tokens[0, :, 0].detach().cpu(),
            )
        return patches, tokens

    def random_masking_simple(self, patches):
        B, num_patches, patch_size = patches.shape
        num_masked = int(self.mask_prob * num_patches)
        # we have a windows_random
        # Generate random mask indices
        mask_indices = torch.rand(B, num_patches).argsort(dim=1)[:, :num_masked]
        # Create a mask array with the same shape as tokens, initialized to False
        mask = torch.zeros(B, num_patches, dtype=torch.bool)
        mask[torch.arange(B).unsqueeze(1), mask_indices] = True
        masked_spec = patches.clone()
        masking_noise = torch.randn_like(masked_spec) * 0.1
        masked_spec[mask] = masking_noise[mask]
        return masked_spec, mask.to(patches.device)

    def random_masking(self, patches):
        B, num_patches, patch_size = patches.shape
        mx = patches.clone()

        len_masking_spec_frames = math.ceil(
            self.mask_seconds * self.sr / self.hop_length
        )
        windows_tokens = (
            len_masking_spec_frames
            // self.patch_size[1]
            * (self.rep_dims // self.patch_size[0])
        )

        # Generate random mask indices
        start_indices = (
            torch.rand(B, math.ceil(num_patches / windows_tokens)) < self.mask_prob
        )
        mask = start_indices.repeat_interleave(windows_tokens, dim=1)

        # Trim mask to fit the number of patches
        if mask.size(1) > num_patches:
            mask = mask[:, :num_patches]

        # Mask with random values
        masking_noise = (torch.randn(mx.shape, dtype=patches.dtype) * 0.1).to(
            patches.device
        )  # 0 mean 0.1 std
        # Apply masking in parallel
        mx[mask] = masking_noise[mask]
        # tensor 1 x N repeat to 16 x N
        return mx, mask.to(patches.device)

    def get_loss(self, logits, target_tokens, mask):
        # zeros boolean with the shape of logit_out
        masked_logits = logits[mask]
        masked_tokens = target_tokens[mask]
        # The loss is calculated only for the masked tokens
        losses = self.loss(masked_logits, masked_tokens)
        accuracies = (
            torch.sum(masked_logits.argmax(1) == masked_tokens) / masked_tokens.numel()
        )
        return losses, accuracies

    def forward(self, x):
        x = x[0]
        B = x.shape[0]

        if isinstance(self.representation, nn.ModuleList):
            target_tokens_l = []
            for i, rep in enumerate(self.representation):
                x_rep = rep(x)
                x_rep, target_tokens = self.vit_tokenization(
                    x_rep, self.quantizers[i], rep
                )
                target_tokens_l.append(target_tokens)

                if isinstance(rep, self.input_representation):
                    x_input = x_rep

            # trim to the shortest
            min_len = min([t.shape[1] for t in target_tokens_l])
            max_len = max([t.shape[1] for t in target_tokens_l])
            diff = max_len - min_len
            assert diff < 3, f"diff {diff} is too big"

            target_tokens_l = [t[:, :min_len] for t in target_tokens_l]
            target_tokens = torch.cat(target_tokens_l, dim=-1)
            x_input = x_input[:, :min_len]

            x = x_input

        else:
            x = self.representation(x)
            # get target feature tokens
            x, target_tokens = self.vit_tokenization(
                x, self.quantizers, self.representation
            )  # B x t x (16 x 4)

        # masking

        x, mask = self.random_masking(x)

        x = self.embedding_layer(x)
        x = self.net(x)
        logits = self.linear(x)
        logits = logits.view(
            B, -1, self.codebook_size, self.num_codebooks * self.n_reps
        )
        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, mask)
        return logits, losses, accuracies, target_tokens

    def training_step(self, batch, batch_idx):
        x = batch
        logits, loss, accuracies, target_tokens = self.forward(x)
        # log tokens coverage
        first_coverage_steps = 100

        if self.first_coverage and batch_idx < first_coverage_steps:
            # collapse batch and time axes
            tokens_c = target_tokens.view(-1, self.num_codebooks)

            # log tokens
            for i in range(self.num_codebooks):
                self.tokens_accumulator[f"codebook_{i}"].extend(
                    tokens_c[:, i].cpu().tolist()
                )

        elif self.first_coverage and batch_idx == first_coverage_steps:
            # Print the histogram you can check it in the wandb dashboard (log section)
            for key, value in self.tokens_accumulator.items():
                self.logger.experiment.log({f"{key}_histogram": wandb.Histogram(value)})
            print(
                f"Logged histograms of token counts for the first {first_coverage_steps} steps."
            )
            self.first_coverage = False
            del self.tokens_accumulator

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracies)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        logits, loss, accuracies, target_tokens = self.forward(x)
        self.log("val_loss", loss, prog_bar=True)
        self.log(f"val_acc", accuracies)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        # If the input is all zeros, return None to give signal
        if x is None:
            return None
        return self.extract_embeddings(x.squeeze(0))

    def extract_embeddings(
        self,
        audio: torch.Tensor,
        layers: Set[int] | None = None,
        overlap_ratio: float | None = None,
    ):
        """Extract audio embeddings using the model.

        Parameters:
            audio (torch.Tensor): 1D audio tensor.
            layer (list): List of layer indices to extract embeddings from.
            By default, it extracts embeddings from the last layer.

        Output:
            torch.Tensor: Extracted embeddings.
                Even in the case of aggregation or single layer embeddings,
                the output tensor will have the same shape (L, B, T, C,)
                where L = len(layer), B is the number of chunks
                T is the number of melspec frames the model can accomodate
                C = model output dimension. No aggregation is applied.

        """

        assert audio.ndim == 1, f"audio must be a 1D audio tensor not {audio.ndim}D."

        # If a layer is not provided, use the layer specified at initialization
        if layers is None:
            layers = self.downstream_embedding_layer
        assert isinstance(layers, set), "Layer must be a set."
        if overlap_ratio is None:
            overlap_ratio = self.overlap_ratio
        assert (
            overlap_ratio >= 0 and overlap_ratio < 1
        ), "Overlap ratio must be between 0 and 1."

        # Compute the representation
        x = self.representation(audio)  # (F, Tm)

        # TODO: what if in Frequency axis we need to aggregate?
        assert x.shape[0] == self.patch_size[0], (
            f"Frequency patching is not implemented yet!"
            f"Expected {self.patch_size[0]} but got {x.shape[0]}"
        )

        # Chunk the representation using the model's context length
        chunk_len = self.patch_size[1] * self.net.num_patches
        hop_len = int(chunk_len * (1 - overlap_ratio))

        # Number of chunks
        Nc = max((x.shape[1] - chunk_len) // hop_len + 1, 1)

        # Create the chunks
        x_chunks = torch.zeros(
            (Nc, x.shape[0], chunk_len), device=x.device, dtype=x.dtype
        )
        for i in range(Nc):
            chunk = x[:, i * hop_len : i * hop_len + chunk_len]
            x_chunks[i, :, : chunk.shape[1]] = chunk

        # Embed the representation
        x_chunks, _ = self.vit_tokenization(x_chunks)  # (B, N, P1*P2)
        x_chunks = self.embedding_layer(x_chunks)  # (B, N, Cin)
        # TODO: support multiple layers
        x_chunks = self.net(x_chunks, layers=layers)  # (B, N, Cout)
        x_chunks = x_chunks.unsqueeze(0)  # (L, B, N, Cout)

        return x_chunks

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
