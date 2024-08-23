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

    def __init__(self, num_layers, activation, in_features, num_labels):
        super(MTTProbe, self).__init__()

        self.num_labels = num_labels
        self.activation = activation
        self.in_features = in_features
        self.num_layers = num_layers

        # TODO create the probe with gin
        self.model = []
        for _ in range(num_layers):
            if activation == "relu":
                self.model.append(nn.ReLU())
            elif activation == "sigmoid":
                self.model.append(nn.Sigmoid())
            else:
                # TODO: more later
                raise ValueError(f"Unknown activation function: {activation}")
            self.model.append(nn.Linear(in_features, num_labels, bias=False))
            in_features = num_labels
        self.model = nn.Sequential(*self.model)

        # TODO sigmoid or not?
        self.criterion = nn.BCEWithLogitsLoss()
        # TODO metrics fixed or with gin?
        self.metrics = {
            "AUROC-macro": MultilabelAUROC(num_labels=num_labels, average="macro"),
            "MAP-macro": MultilabelAveragePrecision(
                num_labels=num_labels, average="macro"
            ),
        }

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
        contain all the chunks of a single track."""

        x, y_true = batch
        assert y_true.shape[0] == 1, "A batch should contain a single track"
        assert x.ndim == 2, "input should be 2D tensor of chunks"

        # process each chunk separately
        logits = self.forward(x)  # (n_chunks, num_labels)
        # Aggregate the chunk embeddings
        logits = torch.mean(logits, dim=0, keepdim=True)  # (1, num_labels)
        # Calculate the loss for the track
        loss = self.criterion(logits, y_true)
        self.log("val_loss", loss)
        if return_predicted_class:
            predicted_class = (torch.sigmoid(logits) > 0.5).int()
            return logits, loss, predicted_class
        return logits, loss

    # Calculate the metrics
    def validation_step(self, batch, batch_idx):
        logits, loss = self.predict(batch)
        self.log("val_loss", loss)
        # Update all metrics with the current batch
        y_true = batch[1].int()
        for _, metric in self.metrics.items():
            metric.update(logits, y_true)

    def on_validation_epoch_end(self):
        # Calculate and log the final value for each metric
        for name, metric in self.metrics.items():
            self.log(f"val-{name}", metric.compute())

    def test_step(self, batch, batch_idx):
        logits, _ = self.predict(batch)
        # Update all metrics with the current batch
        y_true = batch[1].int()
        for _, metric in self.metrics.items():
            metric.update(logits, y_true)

    def on_test_epoch_end(self):
        # Calculate and log the final value for each metric
        for name, metric in self.metrics.items():
            self.log(f"test-{name}", metric.compute())

    def configure_optimizers(self):
        # TODO take lr from construction
        return torch.optim.Adam(self.parameters(), lr=1e-4)
