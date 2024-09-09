from pathlib import Path

import gin
import mir_eval
import torch
import torchmetrics
from torch import nn
import pytorch_lightning as L
from torchmetrics import Accuracy, ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch.nn.functional as F


class SegmentDetectionMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.scores_list = []

    def update(self, reference_intervals, estimated_intervals):
        # NOTE IT IS WORKING FOR BATCHSIZE 1
        if isinstance(reference_intervals, torch.Tensor):
            reference_intervals = reference_intervals.detach().cpu().numpy()
        if isinstance(estimated_intervals, torch.Tensor):
            estimated_intervals = estimated_intervals.detach().cpu().numpy()
        scores, _, _ = mir_eval.segment.detection(
            reference_intervals, estimated_intervals, 0.5
        )
        self.scores_list.append(scores)

    def compute(self):
        return torch.mean(torch.tensor(self.scores_list))


@gin.configurable
class StructureClassProbe(L.LightningModule):
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
        save_prediction: bool = False,
    ):
        super(StructureClassProbe, self).__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = dropout
        self.lr = lr
        self.num_aggregations = num_aggregations
        self.avg = nn.AvgPool1d(kernel_size=num_aggregations, stride=num_aggregations)
        self.save_prediction = save_prediction

        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.frame_output = nn.Linear(hidden_size, num_classes, bias=bias)
        self.boundary_output = nn.Linear(hidden_size, 1, bias=bias)
        self.criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        self.criterion_boundaries = nn.BCEWithLogitsLoss(weight=torch.tensor([0.05]))

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
                "test_boundaries": SegmentDetectionMetric(),
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
        # normalize boundaries between 0 and 1
        if torch.sum(boundaries_smoothed) != 0:
            boundaries_smoothed = (boundaries_smoothed - boundaries_smoothed.min()) / (
                boundaries_smoothed.max() - boundaries_smoothed.min()
            )

        # boundaries_smoothed
        loss_boundaries = self.criterion_boundaries(
            logits_boundaries, boundaries_smoothed
        )
        # normalize
        loss_boundaries = loss_boundaries / x.shape[1]
        self.log("train_loss", loss + loss_boundaries)
        self.log("train_loss_frame", loss)
        self.log("train_loss_boundaries", loss_boundaries)
        return 0.1 * loss + 0.9 * loss_boundaries

    def predict(self, batch):
        x, y_true, boundaries, _, _ = batch
        logits, logits_boundaries = self.forward(x[::10])
        # frame
        logits = logits.reshape(-1, logits.shape[-1])
        y_true_one_hot = self._one_hot(y_true.squeeze(0), logits.shape, logits.device)
        loss = self.criterion(logits, y_true_one_hot)
        # boundaries
        logits_boundaries = logits_boundaries.reshape(-1, 1)
        boundaries_smoothed = apply_moving_average(boundaries.unsqueeze(-1), 3).squeeze(
            0
        )
        # normalize boundaries between 0 and 1
        boundaries_smoothed = (boundaries_smoothed - boundaries_smoothed.min()) / (
            boundaries_smoothed.max() - boundaries_smoothed.min()
        )
        loss_boundaries = self.criterion_boundaries(
            logits_boundaries, boundaries_smoothed
        )
        predicted_class = torch.argmax(logits, dim=1)
        return logits, loss, predicted_class, logits_boundaries, loss_boundaries

    def validation_step(self, batch, batch_idx):
        logits, loss, preds, logits_boundaries, loss_boundaries = self.predict(batch)
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
        logits, loss, _, logits_boundaries, loss_boundaries = self.predict(batch)
        self.log("test_loss", loss + loss_boundaries)
        y_true = batch[1].int().squeeze(0)

        if self.save_prediction:
            torch.save(
                {
                    "embedding": batch[0][::10],
                    "logits_frames": logits,
                    "logits_boundaries": logits_boundaries,
                    "y_true": y_true,
                    "boundaries_intervals": batch[3],
                    "path": batch[4],
                },
                f"src/probe/visualize_probe/embedding_structure/{batch[4]}.pt",
            )

        # postprocessing
        logits_boundaries = torch.sigmoid(logits_boundaries)
        peaks = peak_picking(logits_boundaries, 0.064 * self.num_aggregations)
        peaks = thresholding(peaks, logits_boundaries, 0.0)
        normalized_frames, boundary_prediction = normalize_frames_with_peaks(
            peaks, logits, 0.064 * self.num_aggregations
        )
        self.test_metrics["test-acc"].update(normalized_frames, y_true)
        boundary_intervals = batch[3]
        self.test_metrics["test_boundaries"].update(
            np.array(boundary_prediction), np.array(boundary_intervals)
        )
        # Update the confusion matrix metric
        self.conf_matrix.update(normalized_frames, y_true)

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
                ax.text(
                    j,
                    i,
                    int(conf_matrix[i, j].item()),
                    ha="center",
                    va="center",
                    color="black",
                )

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
            y_true_one_hot = apply_moving_average(y_true_one_hot.unsqueeze(0)).squeeze(
                0
            )
        else:
            raise ValueError("shape_embedding must be either 2D or 3D")

        return y_true_one_hot


def apply_moving_average(matrix, filter_size=10):
    # Input is expected in shape B x T x C
    B, T, C = matrix.shape

    # We need to transpose to (B x C x T) for conv1d
    matrix_torch = matrix.transpose(1, 2)  # Shape: (B, C, T)

    # Define the moving average filter (simple averaging kernel) for each channel
    filter_kernel = (
        torch.ones(C, 1, filter_size) / filter_size
    )  # Shape: (C, 1, filter_size)
    filter_kernel = filter_kernel.to(matrix_torch.device)

    # Apply the convolution (moving average) along the time axis (T) using grouped convolution
    smoothed_matrix_torch = F.conv1d(
        matrix_torch, filter_kernel, padding=filter_size // 2, groups=C
    )

    # Trim the extra time step if the output size is larger than the input
    if smoothed_matrix_torch.shape[2] > T:
        smoothed_matrix_torch = smoothed_matrix_torch[
            :, :, :T
        ]  # Trim to match original T

    # Transpose back to original shape (B x T x C)
    smoothed_matrix = smoothed_matrix_torch.transpose(1, 2)

    return smoothed_matrix


def normalize_frames_with_peaks(peaks, function_probabilities, frame_size):
    """
    Assign the function labels with the largest average probability in each segment.

    Parameters:
    - peaks: A list of peaks (indices of boundary frames).
    - function_probabilities: A 2D tensor where each row corresponds to a frame and
                              each column corresponds to the probability of a specific function label.

    Returns:
    - assigned_labels: A list of function labels assigned to each segment.
    """

    num_frames = function_probabilities.size(0)
    num_labels = function_probabilities.size(1)
    assigned_labels = []

    # if not present add a boundary for first segment
    if peaks[0] != 0:
        peaks = [0] + peaks
    # if not present add a boundary add a boundary for the last segment
    if peaks[-1] != num_frames:
        peaks = peaks + [num_frames]

    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]  # The segment is from start to just before the next peak

        # Get the frames for the current segment
        segment_probs = function_probabilities[start:end]

        # Calculate the average probability for each function label
        avg_probs = torch.mean(segment_probs, dim=0)

        # Find the label with the highest average probability
        label = torch.argmax(avg_probs).item()

        # Assign this label to the entire segment
        assigned_labels.append((start, end - 1, label))

    final_labels = []
    for i in range(len(assigned_labels)):
        start, end, label = assigned_labels[i]
        for j in range(start, end + 1):
            final_labels.append(label)

    assert (
        len(final_labels) == num_frames
    ), f"Length of assigned labels {len(final_labels)} != {num_frames}"

    # TODO I AM NOT CLEAR WHEN FINISH THE BOUNDARY IN THE NEW FRAME OR IN THE PREVIOUS FRAME
    boundaries = [
        (start * frame_size, end * frame_size) for start, end, _ in assigned_labels
    ]

    return torch.tensor(final_labels).to(function_probabilities.device), boundaries


def peak_picking(boundary_activation_curve, frame_size):
    """
    3.4 Peak-picking
    At test time, we apply the trained network to each position
    in the spectrogram of the music piece to be segmented, obtaining a boundary probability for each frame. We then
    employ a simple means of peak-picking on this boundary
    activation curve: Every output value that is not surpassed
    within ±6 seconds is a boundary candidate. From each
    candidate value we subtract the average of the activation
    curve in the past 12 and future 6 seconds, to compensate
    for long-term trends. We end up with a list of boundary
    candidates along with strength values that can be thresholded at will. We found that more elaborate peak picking
    methods did not improve results.

    Parameters:
    - boundary_activation_curve: A 1D torch.Tensor containing the boundary activation values for each frame.
    - frame_size: The duration of each frame in seconds.

    Returns:
    - peaks: List of tuples, where each tuple contains (frame_index, peak_strength).
    """

    # Calculate the number of frames corresponding to the time windows
    window_size_past = int(12 / frame_size)  # 12 seconds converted to number of frames
    window_size_future = int(6 / frame_size)  # 6 seconds converted to number of frames

    peaks = []

    for i in range(
        window_size_past, len(boundary_activation_curve) - window_size_future
    ):
        current_value = boundary_activation_curve[i]

        # Every output value that is not surpassed within ±6 seconds is a boundary candidate
        local_max = torch.max(
            boundary_activation_curve[
                i - window_size_future : i + window_size_future + 1
            ]
        )

        # Check if the current value is a local peak
        if current_value == local_max:
            # From each candidate value we subtract the average of the activation curve
            # in the past 12 and future 6 seconds, to compensate for long-term trends
            past_average = torch.mean(
                boundary_activation_curve[max(0, i - window_size_past) : i]
            )
            future_average = torch.mean(
                boundary_activation_curve[i + 1 : i + 1 + window_size_future]
            )

            # Compensate for long-term trends
            compensated_value = current_value - (past_average + future_average) / 2

            # We end up with a list of boundary candidates along with strength values
            peaks.append((i, compensated_value.item()))

    return peaks


def thresholding(peaks, curve, threshold):
    ans = []
    for frame_index, peak_strength in peaks:
        if curve[frame_index] > threshold:
            ans.append(frame_index)
    return ans


# Function to plot the heatmap using matplotlib
def plot_labels(matrix):
    matrix = matrix.cpu().transpose(0, 1)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Display the heatmap using imshow
    ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    # Adding labels
    ax.set_title("Heatmap of Binary Matrix")
    ax.set_xlabel("Class")
    ax.set_ylabel("Time Step")

    # Adding grid lines
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.grid(False)

    plt.show()
