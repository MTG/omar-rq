import math
from pathlib import Path
import random

import numpy as np

import torch
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
import gin.torch


label_to_number = {
    "intro": 0,
    "verse": 1,
    "chorus": 2,
    "bridge": 3,
    "outro": 4,
    "inst": 5,
    "silence": 6,
}


class HarmonixEmbeddingLoadingDataset(Dataset):
    """Dataset for loading embeddings and labels from the Magnatagatune dataset."""

    def __init__(
        self,
        embeddings_dir: Path,
        gt_path: Path,
        filelist: Path,
        mode: str,
        num_frames_aggregate: int,
        overlap: float,
    ):
        """filelist is a text file with one filename per line without extensions."""

        self.embeddings_dir = embeddings_dir
        self.gt_path = gt_path
        self.filelist = filelist
        self.mode = mode
        self.num_frames_aggregate = num_frames_aggregate
        self.overlap = overlap

        # Load the embeddings and labels
        (
            self.embeddings,
            self.labels,
            self.boundaries,
            self.boundary_intervals,
            self.paths,
        ) = ([], [], [], [], [])
        filenames = [p.strip() for p in open(filelist).readlines()]

        for filename in filenames:
            emb_name = Path(filename + ".pt")
            emb_path = self.embeddings_dir / Path(str(emb_name)[:3]) / emb_name
            # If the embedding exists, add it to the filelist
            if emb_path.exists():
                self.embeddings.append(emb_path)
                self.paths.append(filename)

        if self.mode == "train":
            # iterate all the dataset
            class_counts = {label: 0 for label in range(0, 7)}
            for idx in range(len(self.embeddings)):
                emb_path = self.embeddings[idx]
                path = self.paths[idx]
                embeddings = torch.load(emb_path, map_location="cpu")
                _, N, F, D = embeddings.shape
                labels_song = [0 for _ in range(7)]
                for index in range(N):

                    y_true = self._get_labels_train(path, embeddings, index, F)
                    for label in y_true.flatten():
                        labels_song[label.item()] += 1
                labels_song = np.array(labels_song) / N
                class_counts = {label: class_counts[label] + labels_song[label] for label in range(0, 7)}
            print(class_counts)
            self.class_counts = class_counts

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        """Loads the labels and the processed embeddings for a given index."""
        emb_path = self.embeddings[idx]
        path = self.paths[idx]
        embeddings = torch.load(emb_path, map_location="cpu")
        _, N, F, D = embeddings.shape
        embeddings = torch.squeeze(embeddings, 0)

        if self.mode in ["val", "test"]:
            return self._get_val_test_data(path, embeddings, N, F)
        elif self.mode == "train":
            return self._get_train_data(path, embeddings, N, F)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _get_val_test_data(self, path, embeddings, N, F):
        # TODO AQUI HAY UN ERROR
        frames_length = F // self.num_frames_aggregate * math.ceil(N * self.overlap)

        path_structure = self.gt_path / Path(path + ".txt")

        labels = self.prepare_structure_class_annotations(path_structure, frames_length, 1)
        boundaries, boundary_intervals = self.prepare_boundary_class_annotations(path_structure, frames_length, 1)

        labels = torch.tensor(labels)
        boundaries = torch.tensor(boundaries).float()

        return embeddings, labels, boundaries, boundary_intervals, path

    def _get_train_data(self, path, embeddings, N, F):
        frames_length = embeddings.shape[1] * embeddings.shape[2] // self.num_frames_aggregate

        random_int = random.randint(0, N - 1)
        embeddings = embeddings[random_int]
        fragment_size = F // 3

        path_structure = self.gt_path / Path(path + ".txt")

        labels = self.prepare_structure_class_annotations(path_structure, frames_length, self.overlap,
                                                          random_int, fragment_size)
        boundaries, boundary_intervals = self.prepare_boundary_class_annotations(path_structure, frames_length,
                                                                                 self.overlap, random_int,
                                                                                 fragment_size)

        labels = torch.tensor(labels)
        boundaries = torch.tensor(boundaries).float()

        return embeddings, labels, boundaries

    def _get_labels_train(self, path, embeddings, index, F):
        frames_length = int(embeddings.shape[1] * embeddings.shape[2] // self.num_frames_aggregate // self.overlap)

        fragment_size = F // 3
        path_structure = self.gt_path / Path(path + ".txt")
        # if index == 20:
        #     print()
        labels = self.prepare_structure_class_annotations(path_structure, frames_length, self.overlap,
                                                          index, fragment_size)
        labels = torch.tensor(labels)
        return labels

    def prepare_structure_class_annotations(self, file_path, output_length, overlap, start=0, fragment_size=None):
        timestamps, labels = self._read_annotation_file(file_path)
        output_labels = self._generate_output_labels(timestamps, labels, output_length, overlap, start, fragment_size)
        return output_labels

    def prepare_boundary_class_annotations(self, file_path, output_length, overlap, start=0, fragment_size=None):
        timestamps, labels = self._read_annotation_file(file_path)
        output_boundaries = self._generate_output_boundaries(timestamps, labels, output_length, overlap, start,
                                                             fragment_size)
        boundary_intervals = self._generate_boundary_intervals(timestamps, output_length)
        return output_boundaries, boundary_intervals

    def _read_annotation_file(self, file_path):
        timestamps = []
        labels = []
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    timestamps.append(float(parts[0]))
                    labels.append(parts[1])
        return timestamps, labels

    def _generate_output_labels(self, timestamps, labels, output_length, overlap, start, fragment_size):
        # if start == 20:
        #     print()
        # Output list for labels
        output_labels = []

        # Calculate the initial time based on the start, overlap, and frame rate (30 frames per second)
        initial_time = start * overlap * 30

        # Find the starting index for labels based on the timestamp
        label_index = 0 if start == 0 else next(
            (i for i, timestamp in enumerate(timestamps) if timestamp >= initial_time), len(timestamps)) - 1

        # Determine the range of steps (how many labels to generate)
        range_end = start + (fragment_size if fragment_size is not None else output_length)

        # Loop through each step to generate the labels
        for step in range(range_end - start):
            # Calculate the current time for the step
            current_time = initial_time + (step * 0.064 * self.num_frames_aggregate)

            # Check if we need to move to the next label based on the current time
            if label_index < len(timestamps) and current_time >= timestamps[label_index]:
                current_label = labels[label_index]
                label_index += 1
            else:
                # Use the previous label if no new timestamp is reached
                current_label = labels[label_index - 1] if label_index > 0 else labels[0]

            # Convert the current label to a number, using "silence" if no match is found
            label_number = label_to_number.get(current_label, label_to_number["silence"])

            # Add the label number to the output list
            output_labels.append(label_number)

        return output_labels

    def _generate_output_boundaries(self, timestamps, labels, output_length, overlap, start, fragment_size):
        output_boundaries = []
        initial_time = start * 30 * overlap
        range_end = start + (fragment_size if fragment_size is not None else output_length)
        label_index = 0 if start == 0 else next(
            (i for i, timestamp in enumerate(timestamps) if timestamp >= initial_time), len(timestamps)) - 1
        previous_label = labels[label_index]

        for step in range(range_end - start):
            current_time = initial_time + (step * 0.064 * self.num_frames_aggregate)
            if label_index < len(timestamps) and current_time >= timestamps[label_index]:
                current_label = labels[label_index]
                label_index += 1
            else:
                current_label = labels[label_index - 1] if label_index > 0 else labels[0]

            output_boundaries.append(1 if current_label != previous_label else 0)
            previous_label = current_label

        return output_boundaries

    def _generate_boundary_intervals(self, timestamps, output_length):
        boundary_intervals = [(0, timestamps[0])] if timestamps[0] != 0 else []
        for i in range(1, len(timestamps)):
            boundary_intervals.append((timestamps[i - 1], timestamps[i]))
        boundary_intervals.append((timestamps[-1], timestamps[-1] + output_length * 0.064 * self.num_frames_aggregate))
        return boundary_intervals


def collate_fn_val_test(items):
    """Collate function to pack embeddings and labels for validation and testing."""
    assert len(items) == 1, "Validation and testing should have one track at a time."
    embeddings, labels, boundaries, boundary_intervals, path = zip(*items)
    return (
        embeddings[0],
        labels[0].unsqueeze(0),
        boundaries[0].unsqueeze(0),
        boundary_intervals[0],
        path[0],
    )


@gin.configurable
class HarmonixEmbeddingLoadingDataModule(L.LightningDataModule):
    """DataModule for loading embeddings and labels from the Magnatagatune dataset."""

    def __init__(
        self,
        embeddings_dir: Path,
        gt_path: Path,
        train_filelist: Path,
        val_filelist: Path,
        test_filelist: Path,
        batch_size: int,
        num_workers: int,
        num_frames_aggregate: int,
        overlap: float,
    ):
        super().__init__()
        self.embeddings_dir = embeddings_dir
        self.gt_path = gt_path
        self.train_filelist = train_filelist
        self.val_filelist = val_filelist
        self.test_filelist = test_filelist
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_frames_aggregate = num_frames_aggregate
        self.num_classes = 8  # this number is not going to be modified
        self.overlap = overlap

        # Load one embedding to get the dimension
        # NOTE: I tried doing this inside self.setup() but those are
        # called when the trainer is used.

        emb_name = Path(open(train_filelist).readlines()[0].strip() + ".pt")
        emb_path = self.embeddings_dir / Path(str(emb_name)[:3]) / emb_name
        embedding = torch.load(emb_path, map_location="cpu")
        self.embedding_dimension = embedding.shape[-1]

    def setup(self, stage: str):
        if stage == "fit":
            print("\nSetting up Train dataset...")
            self.train_dataset = HarmonixEmbeddingLoadingDataset(
                self.embeddings_dir,
                self.gt_path,
                self.train_filelist,
                num_frames_aggregate=self.num_frames_aggregate,
                mode="train",
                overlap=self.overlap,
            )
            print("\nSetting up Validation dataset...")
            self.val_dataset = HarmonixEmbeddingLoadingDataset(
                self.embeddings_dir,
                self.gt_path,
                self.val_filelist,
                num_frames_aggregate=self.num_frames_aggregate,
                overlap=self.overlap,
                mode="val",
            )
        if stage == "test":
            print("Setting up the Test dataset...")
            self.test_dataset = HarmonixEmbeddingLoadingDataset(
                self.embeddings_dir,
                self.gt_path,
                self.test_filelist,
                overlap=self.overlap,
                num_frames_aggregate=self.num_frames_aggregate,
                mode="test",
            )

    @property
    def class_weights(self):
        self.setup("fit")
        weights = []
        for key in self.train_dataset.class_counts.keys():
            weights.append(1 / (self.train_dataset.class_counts[key] + 1))
        return torch.tensor(weights) * 1e6

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_val_test,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_val_test,
        )
