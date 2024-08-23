import gin
import torch
from torch import nn
import pytorch_lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAUROC


@gin.configurable
class MTTProbe(L.LightningModule):
    """Train a probe on top of a pre-trained model to predict the
    tags of the Magnatagatune dataset. The probe is trained for multi-label
    classification. The AUROC and Average Precision are used as metrics.
    """

    def __init__(
        self,
        in_features,
        num_labels,
        hidden_size,
        num_layers,
        activation,
        lr,
    ):
        super(MTTProbe, self).__init__()

        self.in_features = in_features
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.lr = lr

        # TODO create the probe with gin
        layers = []
        for i in range(num_layers):
            if i == num_layers - 1:
                hidden_size = num_labels

            # TODO: bias? WD?
            layers.append(nn.Linear(in_features, hidden_size))

            # Choose the activation
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                # TODO: more later
                raise ValueError(f"Unknown activation function: {activation}")

            in_features = hidden_size
        self.model = nn.Sequential(*layers)

        self.criterion = nn.BCEWithLogitsLoss()  # TODO sigmoid or not?

        # Initialize the metrics
        metrics = MetricCollection(
            [
                MultilabelAUROC(num_labels=num_labels, average="macro"),
                MultilabelAveragePrecision(num_labels=num_labels, average="macro"),
            ],
            postfix="-macro",
        )
        self.val_metrics = metrics.clone(prefix="val-")
        self.test_metrics = metrics.clone(prefix="test-")

    def forward(self, x):
        # (B, F) -> (B, num_labels)
        logits = self.model(x)
        return logits

    # TODO log train metrics just in case
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

    def validation_step(self, batch, batch_idx):
        logits, loss = self.predict(batch)
        self.log("val_loss", loss)
        y_true = batch[1].int()
        # Update all metrics with the current batch
        output = self.val_metrics(logits, y_true)
        self.log_dict(output)

    def on_validation_epoch_end(self):
        # Calculate and log the final value for each metric
        output = self.val_metrics.compute()
        self.log_dict(output)
        self.val_metrics.reset()

    # TODO log test loss just in case
    def test_step(self, batch, batch_idx):
        logits, _ = self.predict(batch)
        y_true = batch[1].int()
        # Update all metrics with the current batch
        output = self.test_metrics(logits, y_true)
        self.log_dict(output)

    def on_test_epoch_end(self):
        # Calculate and log the final value for each metric
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
