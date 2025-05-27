import gin

import os
import sys
from pathlib import Path
from typing import Tuple
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as L
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))

from ssl_mtg import get_model


@gin.configurable
class CLAP(L.LightningModule):
    """
    Contrastive Language-Audio Pretraining (CLAP) model.
    inspired in

    """

    max_len_s = 30  # s

    def __init__(
        self,
        audio_encoder_name: Path | str,
        text_encoder_name: Path | str,
        audio_encoder_params: dict,
        train_audio_encoder: bool,
        train_text_encoder: bool,
        tokenizers_parallelism: bool,
        proj_size: int,
        temp: float,
        lr: float,
        weight_decay: float,
        seed: int,
        aggregation_type: str = "mean",
        n_pool_att_heads: int = 8,
    ):
        super(CLAP, self).__init__()

        # global variables
        self.seed = seed
        self.text_encoder_name = text_encoder_name
        self.audio_encoder_name = audio_encoder_name
        self.audio_encoder_params = audio_encoder_params
        self.train_audio_encoder = train_audio_encoder
        self.train_text_encoder = train_text_encoder
        self.tokenizers_parallelism = tokenizers_parallelism

        self.lr = lr
        self.weight_decay = weight_decay
        self.proj_size = proj_size
        self.temp = temp

        self.aggregation_type = aggregation_type
        self.n_pool_att_heads = n_pool_att_heads

        self.predict_data = defaultdict(list)

        # TODO: Load text module
        self.text_encoder = SentenceTransformer(
            str(self.text_encoder_name), device=self.device
        )

        for _, param in self.text_encoder.named_parameters():
            param.requires_gradient = self.train_text_encoder

        if self.train_text_encoder:
            self.text_encoder.train()
        else:
            self.text_encoder.eval()

        self.audio_encoder, _ = get_model(
            self.audio_encoder_name,
            device=self.device,
            **self.audio_encoder_params,
        )

        for _, param in self.audio_encoder.named_parameters():
            param.requires_gradient = self.train_audio_encoder

        if self.train_audio_encoder:
            self.audio_encoder.train()
        else:
            self.audio_encoder.eval()

        # aux projection layers
        self.a_z_size = self.audio_encoder.net.embed_dim
        self.proj_a = nn.Linear(self.a_z_size, self.proj_size)

        sr = self.audio_encoder.sr
        patch_size = self.audio_encoder.patch_size

        self.max_timestamps = int(self.max_len_s * sr / patch_size[1])

        if self.aggregation_type == "attention_pooler":
            self.proj_att_query = nn.Linear(self.max_timestamps, 1)
            self.att_pooler = nn.MultiheadAttention(
                embed_dim=self.a_z_size,
                num_heads=self.n_pool_att_heads,
                batch_first=True,
            )

        dummy_text = ""
        self.t_z_size = self.text_encoder.encode(dummy_text).shape[0]
        self.proj_t = nn.Linear(self.t_z_size, self.proj_size)

        # loss function
        self.loss = nn.CrossEntropyLoss()

        if self.tokenizers_parallelism:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def info_nce_loss(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """InfoNCE loss function.

        This function expect features of shape: (2 * batch_size, feat_dim):

        features = [
            F0_1,
            F0_2,
            F0_N,
            F1_1,
            F1_2,
            F1_N,
        ]
        """

        # our implemnentation only works for 2 views
        n_views = 2
        batch_size = features.shape[0] // n_views

        # create two stacked diagonal matrices
        labels = torch.cat(
            [torch.arange(batch_size) for _ in range(n_views)], dim=0
        ).to(self.device)

        # create a matrix of shape (2 * batch_size, 2 * batch_size) with a diagonal and two sub-diagonals
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # normalize features and compute similarity matrix
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # select the positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        # rearange similirities: 1st column are the positives, the rest are the negatives
        logits = torch.cat([positives, negatives], dim=1)

        # create labels as class indices: the target is always the first column (0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # normalize the logits by the temperature
        logits = logits / self.temp

        return logits, labels

    def forward_audio(self, audio):
        x_a = self.audio_encoder.extract_embeddings(audio)  # (B, T, 768)

        x_a.squeeze_(dim=0)

        if self.aggregation_type == "mean":
            x_a = x_a.mean(dim=1)  # (B, 768)
        elif self.aggregation_type == "attention_pooler":
            # Do self attention aggregation
            q_emb = torch.swapaxes(x_a, -2, -1)
            q_emb = self.proj_att_query(q_emb)
            q_emb = torch.swapaxes(q_emb, -2, -1)
            x_a, _ = self.att_pooler(q_emb, x_a, x_a)
            x_a.squeeze_(dim=1)

        return self.proj_a(x_a)

    def forward_text(self, text):
        x_t = self.text_encoder.encode(text, convert_to_tensor=True)
        return self.proj_t(x_t)

    def forward(self, batch):
        a, t = batch

        z_a = self.forward_audio(a)
        z_t = self.forward_text(t)

        z = torch.cat([z_a, z_t], dim=0)

        logits, labels = self.info_nce_loss(z)
        loss = self.loss(logits, labels)

        return z_a, z_t, loss

    def training_step(self, batch, batch_idx):
        _, _, loss = self.forward(batch)
        B = batch[0].shape[0]

        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, loss = self.forward(batch)
        B = batch[0].shape[0]

        self.log("val_loss", loss, prog_bar=True, batch_size=B, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, filenames = batch

        embeddings = self.forward_audio(x).cpu()

        for i in range(len(filenames)):
            self.predict_data[filenames[i]].append(embeddings[i, :])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
