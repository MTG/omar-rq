import gin.torch
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


@gin.configurable
class SimCLR(L.LightningModule):
    def __init__(self, net: nn.Module, temperature: float = 0.1):
        super().__init__()

        self.net = net
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature

    def info_nce_loss(self, features):
        """InfoNCE loss function.

        Expect features of shape: (2 * batch_size, feat_dim):

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
        labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)

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

        # get the views
        # x_0 = batch["view_0"]
        # x_1 = batch["view_1"]
        x_0 = batch[0]
        x_1 = batch[0]

        # get the embeddings and create new axis for stacking them
        z_0 = self.net(x_0)
        z_1 = self.net(x_1)

        # get the logits and labels with the InfoNCE loss
        z = torch.cat([z_0, z_1], dim=0)

        y_hat, y = self.info_nce_loss(z)

        loss = self.criterion(y_hat, y)

        self.log("train/loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
