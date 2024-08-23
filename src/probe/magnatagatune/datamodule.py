import csv
from pathlib import Path

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
        ], "Layer aggregation not recognized."
        assert granularity in ["frame", "chunk", "clip"], "Granularity not recognized."
        if mode == "train":
            assert granularity == "chunk", "Training mode should use chunk granularity."
        assert time_aggregation in [
            "mean",
            "max",
        ], "Time aggregation not recognized."

        self.mode = mode
        self.embeddings_dir = embeddings_dir
        self.annotations_path = gt_path
        self.layer_aggregation = layer_aggregation
        self.granularity = granularity
        self.time_aggregation = time_aggregation
        # self.normalize = normalize # TODO?

        # Load the filelist of the partition
        with open(filelist, "r") as in_f:
            self.filelist = [
                self.embeddings_dir / line[:3] / f"{line.strip()}.pt" for line in in_f
            ]
        assert len(self.filelist) > 0, "No files found in the filelist."
        print(f"{len(self.filelist):,} files specified in the filelist.")

        print("Checking if embeddings exist...")
        self.filelist = [filepath for filepath in self.filelist if filepath.exists()]
        assert len(self.filelist) > 0, "No embeddings found."
        print(f"{len(self.filelist):,} embeddings found.")
        file_names = set([filepath.stem for filepath in self.filelist])

        # Load labels and filter out rows that do not have embeddings
        print("Reading the labels...")
        annotations_clean = []
        with open(self.annotations_path) as in_f:
            labels = csv.reader(in_f, delimiter="\t")
            tags = next(labels)[1:-1]
            for row in labels:
                if row[-1].split("/")[-1].split(".")[0] in file_names:
                    row = row[1:-1]  # skip track id and track path
                    row = torch.tensor(
                        [float(i) for i in row]
                    )  # convert str to float tensor
                    annotations_clean.append(row)
        self.labels = torch.stack(annotations_clean)

        # Keep only the Top50 labels
        print("Keeping only the Top50 labels...")
        _, indices = torch.topk(self.labels.sum(dim=0), 50, largest=True)
        self.labels = self.labels[:, indices]
        self.tags = [tags[i] for i in indices]
        print(f"Top50 labels: {self.tags}")

        # If an example's labels are all zeros, exclude it
        print("Excluding examples without any labels...")
        mask = self.labels.sum(dim=1) > 0
        self.labels = self.labels[mask]
        self.filelist = [self.filelist[i] for i in range(len(self.filelist)) if mask[i]]

        # Load all embeddings to memory
        print("Loading the embeddings to memory...")
        self.embeddings = torch.stack(
            [torch.load(filepath) for filepath in self.filelist]
        )
        assert len(self.labels) == len(
            self.embeddings
        ), "Labels and embeddings do not match."

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        """Load embeddings and labels for a given index. Expects the embeddings
        to be 4D (L, N, T, F) and labels to be 1D."""

        # Load embeddings
        embeddings = self.embeddings[idx]
        assert embeddings.ndim == 4, "Embeddings should be 4D."
        L, N, T, F = embeddings.shape

        # Load labels
        labels = self.labels[idx]  # (C, )

        # Aggregate embeddings through layers (L, N, T, F) -> (N, T, F)
        if self.layer_aggregation == "mean":
            embeddings = embeddings.mean(dim=0)
        elif self.layer_aggregation == "max":
            embeddings = embeddings.max(dim=0)
        elif self.layer_aggregation == "concat":
            embeddings = embeddings.permute(1, 2, 0, 3)  # (N, T, L, F)
            embeddings = embeddings.reshape(N, T, -1)  # (N, T, L*F)
        else:
            embeddings = embeddings.sum(dim=0)

        # Aggregate embeddings through time (N, T, F) -> (N', F)
        if self.granularity == "frame":
            embeddings = embeddings.view(-1, F)  # (N*T, F)
        elif self.granularity == "chunk":
            if self.time_aggregation == "mean":
                embeddings = embeddings.mean(dim=1)  # (N, F)
            elif self.time_aggregation == "max":
                embeddings = embeddings.max(dim=1)  # (N, F)
            # If training, get a random chunk
            if self.mode == "train":
                embeddings = embeddings[torch.randint(0, N, ())]  # (F, )
        else:
            if self.time_aggregation == "mean":
                embeddings = embeddings.mean(dim=(0, 1)).unsqueeze(0)  # (1, F)
            elif self.time_aggregation == "max":
                embeddings = embeddings.max(dim=(0, 1)).unsqueeze(0)  # (1, F)

        return embeddings, labels


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
            persistent_workers=True,  # TODO?
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_val_test,
            multiprocessing_context="spawn",  # TODO?
            persistent_workers=True,  # TODO?
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn_val_test,
            multiprocessing_context="spawn",  # TODO?
            persistent_workers=True,  # TODO?
        )
