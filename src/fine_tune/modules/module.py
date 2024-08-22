import gin
import torch
import wandb
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAUROC
from torch import nn
import pytorch_lightning as L


@gin.configurable
class MTTProbe(L.LightningModule):
    """Train a probe on top of a pre-trained model to predict the
    tags of the Magnatagatune dataset. The probe is trained for multi-label
    classification. The AUROC and Average Precision are used as metrics.
    """

    def __init__(self, model, in_features, num_labels):
        super(MTTProbe, self).__init__()
        # TODO create the probe with gin?
        # self.model = model
        self.model = nn.Linear(in_features, num_labels, bias=False)
        # TODO sigmoid or not?
        self.criterion = nn.BCEWithLogitsLoss()
        # TODO metrics fixed or with gin?
        self.metrics = [
            MultilabelAUROC(num_labels=num_labels, average="macro"),
            MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
        ]

    def forward(self, x):
        # (B, F) -> (B, num_labels)
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        """X : (n_chunks, n_feat_in), y : (n_chunks, num_labels)
        each chunk may com from another track."""

        x, y_true = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y_true)
        self.log("train_loss", loss)
        return loss

    def predict(self, batch, return_predicted_class=False):
        """Prediction step for a single track. A batch should
        contain all the chunks of a single track.

        # x : (n_chunks, n_feat_in)
        # y_true : (num_labels, ) TODO ??"""

        x, y_true = batch
        assert y_true.ndim == 1, "A batch should contain a single track"
        assert x.ndim == 2, "input should be 2D tensor of chunks"

        # y_true = y_true.unsqueeze(0)  # (1, num_labels) TODO?
        # TODO y_true dtype?

        logits = self.forward(x)  # (n_chunks, num_labels)
        # Aggregate the chunk embeddings
        logits = torch.mean(logits, dim=0)  # (num_labels, )
        # Calculate the loss
        loss = self.criterion(logits, y_true)  # (1,) TODO ?
        self.log("val_loss", loss)
        if return_predicted_class:
            predicted_class = (torch.sigmoid(logits) > 0.5).long()
            return logits, loss, predicted_class
        return logits, loss

    # Calculate the metrics
    # TODO
    def validation_step(self, batch, batch_idx):
        y_true = batch[1]
        logits, loss = self.predict(batch)
        self.log("val_loss", loss)
        # Update all metrics with the current batch
        for metric in self.metrics:
            metric.update(logits, y_true)

    def on_validation_epoch_end(self):
        # Calculate and log the final value for each metric
        for metric in self.metrics:
            self.log(f"val-{metric.name}", metric.compute())

    def test_step(self, batch, batch_idx):
        y_true = batch[1]
        logits, _ = self.predict(batch)
        # Update all metrics with the current batch
        for metric in self.metrics:
            metric.update(logits, y_true)

    def on_test_epoch_end(self):
        # Calculate and log the final value for each metric
        for metric in self.metrics:
            self.log(f"test-{metric.name}", metric.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
