from pathlib import Path

import gin
import torch
from torch import nn
import pytorch_lightning as L
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MultilabelConfusionMatrix,
)
import matplotlib.pyplot as plt
import numpy as np
import wandb


@gin.configurable
class SequenceMultiLabelClassificationProbe(L.LightningModule):
    """Train a probe using the embeddings from a pre-trained model to predict the
    labels of a downstream dataset. The probe is trained for multi-label
    classification. The macro AUROC, Mean Average Precision metrics are calculated.
    The confusion matrix is also computed on the test set.
    """

    def __init__(
        self,
        in_features: int,
        num_labels: int,
        hidden_size: int,
        num_layers: int,
        activation: str,
        bias: bool,
        dropout: float,
        lr: float,
        labels: Path = None,
        plot_dir: Path = None,
    ):
        super(SequenceMultiLabelClassificationProbe, self).__init__()

        self.in_features = in_features
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.bias = bias
        self.dropout = dropout
        self.lr = lr
        self.labels = np.load(labels) if labels is not None else None
        self.plot_dir = Path(plot_dir) if plot_dir is not None else None

        # TODO create the probe with gin
        layers = []
        for i in range(num_layers):
            if i == num_layers - 1:
                hidden_size = num_labels

            layers.append(nn.Dropout(dropout))

            # Add the linear layer
            layers.append(nn.Linear(in_features, hidden_size, bias=bias))

            # Choose the activation
            if (i == num_layers - 1) or activation.lower() == "none":
                pass
            elif activation.lower() == "relu":
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
        self.val_metrics = nn.ModuleDict(
            {
                "val-AUROC-macro": MultilabelAUROC(
                    num_labels=num_labels, average="macro"
                ),
                "val-MAP-macro": MultilabelAveragePrecision(
                    num_labels=num_labels, average="macro"
                ),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "test-AUROC-macro": MultilabelAUROC(
                    num_labels=num_labels, average="macro"
                ),
                "test-MAP-macro": MultilabelAveragePrecision(
                    num_labels=num_labels, average="macro"
                ),
            }
        )
        self.test_confusion_matrix = MultilabelConfusionMatrix(num_labels=num_labels)

        self.best_val_metric = {metric: 0.0 for metric in self.val_metrics.keys()}

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

    def validation_step(self, batch, batch_idx):
        logits, loss = self.predict(batch)
        self.log("val_loss", loss)
        # Update all metrics with the current batch
        y_true = batch[1].int()
        for metric in self.val_metrics.values():
            metric.update(logits, y_true)

    def on_validation_epoch_end(self):
        # Calculate and log the final value for each metric
        for name, metric in self.val_metrics.items():
            self.log(name, metric, on_epoch=True)
            # Save the best value
            metric_value = metric.compute().cpu().numpy()
            if metric_value > self.best_val_metric[name]:
                self.best_val_metric[name] = metric_value

    def test_step(self, batch, batch_idx):
        logits, _ = self.predict(batch)
        # Update all metrics with the current batch
        y_true = batch[1].int()
        for metric in self.test_metrics.values():
            metric.update(logits, y_true)
        # Update the confusion matrix
        self.test_confusion_matrix.update(logits, y_true)

    def on_test_epoch_end(self):
        # Calculate and log the final value for each metric
        for name, metric in self.test_metrics.items():
            self.log(name, metric, on_epoch=True)
        # Compute the confusion matrix
        conf_matrix = self.test_confusion_matrix.compute()
        fig = self.plot_confusion_matrix(conf_matrix)
        # Log the figure directly to wandb
        if self.logger:
            self.logger.experiment.log({"test_confusion_matrix": wandb.Image(fig)})
        if self.plot_dir:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.plot_dir / "test_confusion_matrix.png")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def plot_confusion_matrix(self, conf_matrix):
        conf_matrix = conf_matrix.cpu().numpy()
        fig, axes = plt.subplots(
            nrows=10, ncols=5, figsize=(25, 50), constrained_layout=True
        )
        axes = axes.flatten()
        labels = [f"{i+1}" for i in range(50)] if self.labels is None else self.labels
        for ax, cm, label in zip(axes, conf_matrix, labels):
            # Plot the confusion matrix in each subplot
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(label, fontsize=15)
            # Annotation inside the heatmap
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text = ax.text(
                        j, i, cm[i, j], ha="center", va="center", color="red"
                    )

            ax.set_xticks(np.arange(cm.shape[1]))
            ax.set_yticks(np.arange(cm.shape[0]))
            ax.set_xticklabels(["False", "True"])
            ax.set_yticklabels(["False", "True"])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
        return fig
