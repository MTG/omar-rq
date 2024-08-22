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

        # TODO: i can not do it more elegantly for some linux reason
        # line.strip() returns empty string
        self.filelist = []
        with open(filelist, "r") as f:
            for line in f:
                self.filelist.append(
                    self.embeddings_dir / line[:3] / f"{line.replace('\n', '')}.pt"
                )
        # print(self.filelist[0]) # TODO: this does not work on linux
        assert len(self.filelist) > 0, "No files found in the filelist."
        print(f"{len(self.filelist):,} files specified in the filelist.")

        print("Checking if embeddings exist...")
        self.filelist = [filepath for filepath in self.filelist if filepath.exists()]
        assert len(self.filelist) > 0, "No embeddings found."
        print(f"{len(self.filelist):,} embeddings found.")
        file_names = set([filepath.stem for filepath in self.filelist])

        # Load labels and filter out rows that do not have embeddings
        annotations_clean = []
        with open(self.annotations_path) as in_f:
            labels = csv.reader(in_f, delimiter="\t")
            next(labels)  # skip header
            for row in labels:
                if row[-1].split("/")[-1].split(".")[0] in file_names:
                    row = row[1:-1]  # skip track id and track path
                    row = torch.tensor(
                        [float(i) for i in row]
                    )  # convert str to float tensor
                    annotations_clean.append(row)
        self.labels = annotations_clean
        assert len(self.labels) == len(
            self.filelist
        ), "Labels and embeddings do not match."

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        """Load embeddings and labels for a given index.
        Expects the embeddings to be 4D (L, N, T, F) and
        labels to be 50D. TODO 50?
        """

        # Load embeddings
        embeddings = torch.load(self.filelist[idx])
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
                embeddings = embeddings[torch.randint(0, N, (1,))]  # (1, F)
        else:
            if self.time_aggregation == "mean":
                embeddings = embeddings.mean(dim=(0, 1)).unsqueeze(0)  # (1, F)
            elif self.time_aggregation == "max":
                embeddings = embeddings.max(dim=(0, 1)).unsqueeze(0)  # (1, F)

        # Load labels
        labels = self.labels[idx]

        return embeddings, labels

    @staticmethod
    def collate_fn_train(items):
        """Collate function to pack embeddings and labels for training."""
        embeddings, labels = zip(*items)
        embeddings = torch.cat(embeddings)
        labels = torch.stack(labels)
        return embeddings, labels

    @staticmethod
    def collate_fn_val_test(items):
        """Collate function to pack embeddings and labels for validation and testing."""
        embeddings, labels = zip(*items)
        assert (
            len(embeddings) == 1
        ), "Validation and testing should have one track at a time."
        assert (
            len(labels) == 1
        ), "Validation and testing should have one track at a time."
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
            collate_fn=self.train_dataset.collate_fn_train,
            multiprocessing_context="spawn",  # TODO?
            persistent_workers=True,
            # pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # TODO??
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn_val_test,
            multiprocessing_context="spawn",  # TODO?
            persistent_workers=True,
            # pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # TODO??
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn_val_test,
            multiprocessing_context="spawn",  # TODO?
            persistent_workers=True,
            # pin_memory=True,
        )
