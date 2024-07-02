import gin.torch
import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.nn.functional import one_hot


@gin.configurable
class Classifier(L.LightningModule):
    def __init__(self, net: nn.Module):
        super().__init__()

        self.net = net

    def training_step(self, batch):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)

        # TODO: data prep should happen in the datamodule
        y = one_hot(y, num_classes=10).float()

        y_hat = self.net(x)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat, y)

        self.log("train/loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
