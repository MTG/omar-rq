from pathlib import Path

import gin
import torch
from torch import nn
import pytorch_lightning as L
from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MultilabelConfusionMatrix,
)
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch.nn.functional as F


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


@gin.configurable
class AggregateMultiClassProbe(L.LightningModule):
    """Train a probe using the embeddings from a pre-trained model to predict the
    labels of a downstream dataset. The probe is trained for multi-class
    classification. The Acc metrics are calculated.
    The confusion matrix is also computed on the test set.
    """

    def __init__(
            self,
            num_classes: int,
            hidden_size: int,
            bias: bool,
            dropout: float,
            lr: float,
            num_aggregations: int,
            in_features: int,
            class_weights: torch.tensor,
    ):
        super(AggregateMultiClassProbe, self).__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.lr = lr
        self.num_aggregations = num_aggregations
        self.avg = nn.AvgPool1d(kernel_size=num_aggregations, stride=num_aggregations)

        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.frame_output = nn.Linear(hidden_size, num_classes, bias=bias)
        self.boundary_output = nn.Linear(hidden_size, 1, bias=bias)
        self.criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        self.criterion_boundaries = nn.BCEWithLogitsLoss()

        # Initialize the metrics
        self.val_metrics = nn.ModuleDict(
            {
                "val-acc": Accuracy(
                    num_classes=num_classes, average="macro", task="multiclass"
                ),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "test-acc": Accuracy(
                    num_classes=num_classes, average="macro", task="multiclass"
                ),
            }
        )

        # Confusion matrix metric
        self.conf_matrix = ConfusionMatrix(num_classes=num_classes, task="multiclass")

    def forward(self, x):
        x = self.avg(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x = self.model(x)
        logits_frame = self.frame_output(x)
        logits_boundaries = self.boundary_output(x)
        return logits_frame, logits_boundaries

    def training_step(self, batch, batch_idx):
        x, y_true, boundaries = batch
        logits_frame, logits_boundaries = self.forward(x)
        # frame loss
        y_true_one_hot = self._one_hot(y_true, x.shape, x.device)
        loss = self.criterion(logits_frame, y_true_one_hot)
        # boundaries loss
        boundaries_smoothed = apply_moving_average(boundaries.unsqueeze(-1), 3)
        loss_boundaries = self.criterion_boundaries(logits_boundaries, boundaries_smoothed)
        self.log("train_loss", loss + loss_boundaries)
        self.log("train_loss_frame", loss)
        self.log("train_loss_boundaries", loss_boundaries)
        return loss + loss_boundaries

    def predict(self, batch, return_predicted_class=False):
        x, y_true, boundaries = batch
        logits, logits_boundaries = self.forward(x)
        # frame
        logits = logits.reshape(-1, logits.shape[-1])
        y_true_one_hot = self._one_hot(y_true.squeeze(0), logits.shape, logits.device)
        loss = self.criterion(logits, y_true_one_hot)
        # boundaries
        logits_boundaries = logits_boundaries.reshape(-1, 1)
        boundaries_smoothed = apply_moving_average(boundaries.unsqueeze(-1), 3).squeeze(0)
        loss_boundaries = self.criterion_boundaries(logits_boundaries, boundaries_smoothed)
        if return_predicted_class:
            predicted_class = torch.argmax(logits, dim=1)
            return logits, loss, predicted_class, loss_boundaries
        return logits, loss, loss_boundaries

    def validation_step(self, batch, batch_idx):
        logits, loss, preds, loss_boundaries = self.predict(batch, return_predicted_class=True)
        self.log("val_loss", loss + loss_boundaries)
        self.log("val_loss_frame", loss)
        self.log("val_loss_boundaries", loss_boundaries)
        y_true = batch[1].int().squeeze(0)
        for metric in self.val_metrics.values():
            metric.update(preds, y_true)

    def on_validation_epoch_end(self):
        for name, metric in self.val_metrics.items():
            self.log(name, metric, on_epoch=True)

    def test_step(self, batch, batch_idx):
        logits, loss, preds, loss_boundaries = self.predict(batch, return_predicted_class=True)
        self.log("test_loss", loss + loss_boundaries)
        y_true = batch[1].int().squeeze(0)
        for metric in self.test_metrics.values():
            metric.update(preds, y_true)

        # Update the confusion matrix metric
        self.conf_matrix.update(preds, y_true)

    def on_test_epoch_end(self):
        for name, metric in self.test_metrics.items():
            self.log(name, metric, on_epoch=True)

        # Compute and log the confusion matrix
        conf_matrix = self.conf_matrix.compute()
        fig = self.plot_confusion_matrix(conf_matrix)
        wandb.log({"confusion_matrix": wandb.Image(fig)})

        # Clear the confusion matrix for potential future test runs
        self.conf_matrix.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def plot_confusion_matrix(self, conf_matrix):
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(conf_matrix.cpu().numpy(), cmap="Blues")
        fig.colorbar(cax)

        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(range(self.num_classes))
        ax.set_yticklabels(range(self.num_classes))

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")

        # Annotate the confusion matrix
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, int(conf_matrix[i, j].item()), ha='center', va='center', color='black')

        return fig

    def _one_hot(self, y_true, shape_embedding, device):
        if len(shape_embedding) == 3:
            # 3D case
            B, T, E = shape_embedding
            y_true_one_hot = torch.zeros(B, T // 3, self.num_classes).to(device)
            y_true_one_hot.scatter_(2, y_true.unsqueeze(2), 1)
            y_true_one_hot = apply_moving_average(y_true_one_hot)
        elif len(shape_embedding) == 2:
            # 2D case
            T3, E = shape_embedding
            y_true_one_hot = torch.zeros(T3, self.num_classes).to(device)
            y_true_one_hot.scatter_(1, y_true.unsqueeze(1), 0)
            y_true_one_hot = apply_moving_average(y_true_one_hot.unsqueeze(0)).squeeze(0)
        else:
            raise ValueError("shape_embedding must be either 2D or 3D")

        return y_true_one_hot


def apply_moving_average(matrix, filter_size=10):
    # Input is expected in shape B x T x C
    B, T, C = matrix.shape

    # We need to transpose to (B x C x T) for conv1d
    matrix_torch = matrix.transpose(1, 2)  # Shape: (B, C, T)

    # Define the moving average filter (simple averaging kernel) for each channel
    filter_kernel = torch.ones(C, 1, filter_size) / filter_size  # Shape: (C, 1, filter_size)
    filter_kernel = filter_kernel.to(matrix_torch.device)

    # Apply the convolution (moving average) along the time axis (T) using grouped convolution
    smoothed_matrix_torch = F.conv1d(matrix_torch, filter_kernel, padding=filter_size // 2, groups=C)

    # Trim the extra time step if the output size is larger than the input
    if smoothed_matrix_torch.shape[2] > T:
        smoothed_matrix_torch = smoothed_matrix_torch[:, :, :T]  # Trim to match original T

    # Transpose back to original shape (B x T x C)
    smoothed_matrix = smoothed_matrix_torch.transpose(1, 2)

    return smoothed_matrix


# Function to plot the heatmap using matplotlib
def plot_labels(matrix):
    matrix = matrix.cpu().transpose(0, 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Display the heatmap using imshow
    ax.imshow(matrix, cmap='Blues', interpolation='nearest')

    # Adding labels
    ax.set_title('Heatmap of Binary Matrix')
    ax.set_xlabel('Class')
    ax.set_ylabel('Time Step')

    # Adding grid lines
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.grid(False)

    plt.show()
