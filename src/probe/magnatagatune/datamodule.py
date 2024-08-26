from pathlib import Path

import numpy as np

import torch
import pytorch_lightning as L
from torch.utils.data import Dataset, DataLoader


class MTTEmbeddingLoadingDataset(Dataset):
    """Dataset for loading embeddings and labels from the Magnatagatune dataset."""

    def __init__(
        self,
        embeddings_dir: Path,
        gt_path: Path,
        filelist: Path,
        layer_aggregation: str,
        granularity: str,
        time_aggregation: str,
        mode: str,
    ):
        """filelist is a text file with one filename per line without extensions."""
        # TODO more docs

        # Assertions
        assert mode in ["train", "val", "test"], "Mode not recognized."
        assert layer_aggregation in [
            "mean",
            "max",
            "concat",
            "sum",
            "none",
        ], "Layer aggregation not recognized."
        assert granularity in ["frame", "chunk", "clip"], "Granularity not recognized."
        if mode == "train":
            assert granularity == "chunk", "Training mode should use chunk granularity."
        assert time_aggregation in [
            "mean",
            "max",
        ], "Time aggregation not recognized."

        self.embeddings_dir = embeddings_dir
        self.gt_path = gt_path
        self.filelist = filelist
        self.layer_aggregation = layer_aggregation
        self.granularity = granularity
        self.time_aggregation = time_aggregation
        self.mode = mode
        # self.normalize = normalize # TODO?

        # Load the filelist of the partition and the binarized labels from Minz et al. 2020
        filenames = np.load(filelist)
        full_dataset_labels = np.load(gt_path)

        # Load the embeddings and labels
        self.embeddings, self.labels = [], []
        for filename in filenames:
            ix, fn = filename.split("\t")
            emb_name = fn.split("/")[1].replace(".mp3", ".pt")
            emb_path = self.embeddings_dir / emb_name[:3] / emb_name
            # If the embedding exists, add it to the filelist
            if emb_path.exists():
                embedding = torch.load(emb_path)
                self.embeddings.append(embedding)
                binary_label = full_dataset_labels[int(ix)]
                self.labels.append(torch.tensor(binary_label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Loads the labels and the processed embeddings for a given index."""

        embeddings = self.embeddings[idx]
        embeddings = self.prepare_embedding(embeddings)  # (N, F)
        if self.mode == "train":  # If training, get a random chunk
            N = embeddings.size(0)
            embeddings = embeddings[torch.randint(0, N, ())]  # (F, )
        labels = self.labels[idx]  # (C, )

        return embeddings, labels

    def prepare_embedding(self, embeddings):
        """Prepare embeddings for training. Expects the embeddings to be 4D (L, N, T, F)."""

        assert embeddings.ndim == 4, "Embeddings should be 4D."
        L, N, T, F = embeddings.shape

        # Aggregate embeddings through layers (L, N, T, F) -> (N, T, F)
        if self.layer_aggregation == "mean":
            embeddings = embeddings.mean(dim=0)
        elif self.layer_aggregation == "max":
            embeddings = embeddings.max(dim=0)
        elif self.layer_aggregation == "concat":
            embeddings = embeddings.permute(1, 2, 0, 3)  # (N, T, L, F)
            embeddings = embeddings.reshape(N, T, -1)  # (N, T, L*F)
        elif self.layer_aggregation == "sum":
            embeddings = embeddings.sum(dim=0)
        else:
            assert L == 1
            embeddings = embeddings.squeeze(0)

        # Aggregate embeddings through time (N, T, F) -> (N', F)
        if self.granularity == "frame":
            embeddings = embeddings.view(-1, F)  # (N*T, F)
        elif self.granularity == "chunk":
            if self.time_aggregation == "mean":
                embeddings = embeddings.mean(dim=1)  # (N, F)
            elif self.time_aggregation == "max":
                embeddings = embeddings.max(dim=1)  # (N, F)
        else:
            if self.time_aggregation == "mean":
                embeddings = embeddings.mean(dim=(0, 1)).unsqueeze(0)  # (1, F)
            elif self.time_aggregation == "max":
                embeddings = embeddings.max(dim=(0, 1)).unsqueeze(0)  # (1, F)

        return embeddings


def collate_fn_val_test(items):
    """Collate function to pack embeddings and labels for validation and testing."""
    assert len(items) == 1, "Validation and testing should have one track at a time."
    embeddings, labels = zip(*items)
    return embeddings[0], labels[0].unsqueeze(0)


class MTTEmbeddingLoadingDataModule(L.LightningDataModule):
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
        layer_aggregation: str,
        granularity: str,
        time_aggregation: str,
    ):
        super().__init__()
        self.embeddings_dir = embeddings_dir
        self.gt_path = gt_path
        self.train_filelist = train_filelist
        self.val_filelist = val_filelist
        self.test_filelist = test_filelist
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.layer_aggregation = layer_aggregation
        self.granularity = granularity
        self.time_aggregation = time_aggregation

        # Load one embedding to get the dimension
        # NOTE: I tried doing this inside self.setup() but those are
        # called when the trainer is used.
        filename = np.load(train_filelist)[0]
        _, filename = filename.split("\t")
        emb_name = filename.split("/")[1].replace(".mp3", ".pt")
        emb_path = self.embeddings_dir / emb_name[:3] / emb_name
        embedding = torch.load(emb_path)
        self.embedding_dimension = embedding.shape[-1]

    def setup(self, stage: str):
        if stage == "fit":
            print("\nSetting up Train dataset...")
            self.train_dataset = MTTEmbeddingLoadingDataset(
                self.embeddings_dir,
                self.gt_path,
                self.train_filelist,
                self.layer_aggregation,
                self.granularity,
                self.time_aggregation,
                mode="train",
            )
            print("\nSetting up Validation dataset...")
            self.val_dataset = MTTEmbeddingLoadingDataset(
                self.embeddings_dir,
                self.gt_path,
                self.val_filelist,
                self.layer_aggregation,
                self.granularity,
                self.time_aggregation,
                mode="val",
            )
        if stage == "test":
            print("Setting up the Test dataset...")
            self.test_dataset = MTTEmbeddingLoadingDataset(
                self.embeddings_dir,
                self.gt_path,
                self.test_filelist,
                self.layer_aggregation,
                self.granularity,
                self.time_aggregation,
                mode="test",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            multiprocessing_context="spawn",  # TODO?
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_val_test,
            multiprocessing_context="spawn",  # TODO?
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_val_test,
            multiprocessing_context="spawn",  # TODO?
            persistent_workers=True,
        )
