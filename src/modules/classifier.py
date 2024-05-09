import gin.torch
import pytorch_lightning as L
import torch
import torch.nn as nn


@gin.configurable
class Classifier(L.LightningModule):
    def __init__(self, arch: nn.Module):
        super().__init__()

        self.arch = arch

    def training_step(self, batch):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.arch(x)

        loss = nn.BCEWithLogitsLoss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
