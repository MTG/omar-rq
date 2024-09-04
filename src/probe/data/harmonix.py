import math
from pathlib import Path
import random

import numpy as np

import torch
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader
import gin.torch


label_to_number = {
    'intro': 0,
    'verse': 1,
    'chorus': 2,
    'bridge': 3,
    'outro': 4,
    'inst': 5,
    'silence': 6
}


class HarmonixEmbeddingLoadingDataset(Dataset):
    """Dataset for loading embeddings and labels from the Magnatagatune dataset."""

    def __init__(
        self,
        embeddings_dir: Path,
        gt_path: Path,
        filelist: Path,
        mode: str,
        num_frames_aggregate: int
    ):
        """filelist is a text file with one filename per line without extensions."""

        self.embeddings_dir = embeddings_dir
        self.gt_path = gt_path
        self.filelist = filelist
        self.mode = mode
        self.num_frames_aggregate = num_frames_aggregate

        # Load the embeddings and labels
        self.embeddings, self.labels = [], []
        filenames = [p.strip() for p in open(filelist).readlines()]

        for filename in filenames:
            emb_name =  Path(filename + ".pt")
            emb_path = self.embeddings_dir / Path(str(emb_name)[:3]) / emb_name
            # If the embedding exists, add it to the filelist
            if emb_path.exists():
                # shape embeddings: (1, N, F, D)
                embedding = torch.load(emb_path, map_location="cpu")
                _, N, F, D = embedding.shape
                frames_length = embedding.shape[1] * (embedding.shape[2] // num_frames_aggregate)
                embedding = torch.squeeze(embedding, 0)
                self.embeddings.append(embedding)
                path_structure =  gt_path / Path(filename + ".txt")
                label = self.prepare_structure_class_annotations(path_structure, output_length=frames_length)
                #assert N*F//3 == len(label), f"{N * F // 3} != {len(label)}"
                label = torch.tensor(label)
                self.labels.append(label)

        class_counts = {label: 0 for label in range(0, 7)}
        for y_true in self.labels:
            for label in y_true.flatten():
                class_counts[label.item()] += 1
        print(class_counts)
        self.class_counts = class_counts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Loads the labels and the processed embeddings for a given index."""
        embeddings = self.embeddings[idx]
        labels = self.labels[idx]# (N, F)
        if self.mode == "train":  # If training, get a random chunk
            N, F, E = embeddings.shape
            random_int = random.randint(0, N - 1)
            embeddings = embeddings[random_int]
            random_fragment_idx = random_int * (F // self.num_frames_aggregate)
            random_fragment_jdx = random_fragment_idx + (F // 3)
            labels = labels[random_fragment_idx:random_fragment_jdx]
        # if labels is an empty list
        if len(labels) == 0:
            print()
        return embeddings, labels

    def prepare_structure_class_annotations(self, file_path, output_length):
        timestamps = []
        labels = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    timestamp = float(parts[0])
                    label = parts[1]
                    timestamps.append(timestamp)
                    labels.append(label)

        output_labels = []
        label_index = 0

        for step in range(output_length):
            current_time = step * 0.064 * self.num_frames_aggregate
            if label_index < len(timestamps) and current_time >= timestamps[label_index]:
                current_label = labels[label_index]
                label_index += 1
            else:
                current_label = labels[label_index - 1] if label_index > 0 else labels[0]

            label_number = label_to_number.get(current_label, label_to_number['silence'])
            output_labels.append(label_number)
        return output_labels


def collate_fn_val_test(items):
    """Collate function to pack embeddings and labels for validation and testing."""
    assert len(items) == 1, "Validation and testing should have one track at a time."
    embeddings, labels = zip(*items)
    return embeddings[0], labels[0].unsqueeze(0)


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
        num_frames_aggregate: int
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
        self.num_classes = 8 # this number is not going to be modified

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
            )
            print("\nSetting up Validation dataset...")
            self.val_dataset = HarmonixEmbeddingLoadingDataset(
                self.embeddings_dir,
                self.gt_path,
                self.val_filelist,
                num_frames_aggregate=self.num_frames_aggregate,
                mode="val",
            )
        if stage == "test":
            print("Setting up the Test dataset...")
            self.test_dataset = HarmonixEmbeddingLoadingDataset(
                self.embeddings_dir,
                self.gt_path,
                self.test_filelist,
                num_frames_aggregate=self.num_frames_aggregate,
                mode="test",
            )

    @property
    def class_weights(self):
        self.setup("test")
        weights = []
        for key in self.test_dataset.class_counts.keys():
            weights.append(1/(self.test_dataset.class_counts[key]+1))
        return torch.tensor(weights)

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
