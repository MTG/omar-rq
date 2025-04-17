import gin

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
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

    def __init__(
        self,
        audio_encoder_name: Path | str,
        text_encoder_name: Path | str,
        proj_size: int,
        temp: float,
        lr: float,
        weight_decay: float,
        seed: int,
    ):
        super(CLAP, self).__init__()

        # global variables
        self.seed = seed
        self.text_encoder_name = text_encoder_name
        self.audio_encoder_name = audio_encoder_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.proj_size = proj_size
        self.temp = temp

        # TODO: Load text module
        self.text_encoder = SentenceTransformer(
            str(self.text_encoder_name), device=self.device
        )

        self.audio_encoder, _ = get_model(self.audio_encoder_name, device=self.device)

        # aux projection layers
        self.proj_a = nn.Linear(self.net.embed_dim, self.proj_size)
        self.proj_t = nn.Linear(self.net.embed_dim, self.proj_size)

        # loss function
        self.loss = nn.CrossEntropyLoss()

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

    def forward(self, batch):
        a, t = batch

        x_a = self.audio_encoder(a)  # (B, T, 768)

        # TODO: Do a more clever time agregation
        x_a = x_a.mean(dim=1)  # (B, 768)

        x_t = self.text_encoder(t)  # (B, 768)

        z_a = self.proj_a(x_a)
        z_t = self.proj_t(x_t)
        z = torch.cat([z_a, z_t], dim=0)

        logits, labels = self.info_nce_loss(z)
        loss = self.loss(logits, labels)

        return x_a, x_t, loss

    def training_step(self, batch, batch_idx):
        x = batch
        _, _, loss = self.forward(x)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        _, _, loss = self.forward(x)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
