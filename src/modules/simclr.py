import gin.torch
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

first_run = True


@gin.configurable
class SimCLR(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        representation: nn.Module,
        temperature: float,
        lr: float,
        mixup_alpha: float,
    ):
        super().__init__()

        self.net = net
        self.representation = representation

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.lr = lr
        self.mixup_alpha = mixup_alpha

    def info_nce_loss(self, features):
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
        logits = logits / self.temperature
        return logits, labels

    def training_step(self, batch):
        """Training step for SimCLR."""

        global first_run

        # get the views
        x_0 = batch["view_0"]
        x_1 = batch["view_1"]

        # apply mixup augmentation
        if self.mixup_alpha > 0:
            x_0 = self.mixup(x_0)
            x_1 = self.mixup(x_1)

        if first_run:
            print(f"x_0 shape: {x_0.shape}")

        x_0 = self.representation(x_0)
        x_1 = self.representation(x_1)

        if first_run:
            print(f"x_0 (mels) shape: {x_0.shape}")

        # get the embeddings for the views
        z_0 = self.net(x_0)
        z_1 = self.net(x_1)

        if first_run:
            print(f"z_0 shape: {z_0.shape}")

        if first_run:
            self.logger.log_image(key="representation", images=[x_0[0], x_1[0]])

        # stack embeddings on the batch dimension
        z = torch.cat([z_0, z_1], dim=0)

        # get logits and labels and compute the loss
        y_hat, y = self.info_nce_loss(z)
        loss = self.criterion(y_hat, y)

        self.log("train/loss", loss)

        first_run = False

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def mixup(self, x):
        """Mixup data augmentation in the waveform domain (1D)."""

        batch_size = x.size(0)
        rn_indices = torch.randperm(batch_size)

        lambd = np.random.beta(self.mixup_alpha, self.mixup_alpha, batch_size).astype(np.float32)
        lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)

        lam = torch.FloatTensor(lambd).to(x.device)

        x = x * lam.reshape(batch_size, 1) + \
        x[rn_indices] * (1. - lam.reshape(batch_size, 1))

        return x
