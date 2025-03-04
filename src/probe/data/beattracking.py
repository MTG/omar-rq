import pickle as pk
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import gin.torch
import numpy as np
import pytorch_lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class BeatTrackingEmbeddingLoadingDataset(Dataset):
    """Dataset for loading embeddings and labels from the Magnatagatune dataset."""

    def __init__(
        self,
        embeddings_dir: Path,
        filelist: Path,
        gt_path: Path,
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
        assert time_aggregation in [
            "mean",
            "max",
            "none",
        ], "Time aggregation not recognized."

        self.embeddings_dir = embeddings_dir
        self.filelist = filelist
        self.layer_aggregation = layer_aggregation
        self.granularity = granularity
        self.time_aggregation = time_aggregation
        self.mode = mode

        # Load the groundtruth
        with open(gt_path, "rb") as f:
            self.groundtruth = pk.load(f)

        with open(filelist, "r") as f:
            filenames = [line.strip() for line in f.readlines()]

        self.embeddings, self.beats, self.filenames = self.parallel_loading(filenames)

        print(f"Loaded {len(self.embeddings)} embeddings and labels.")

        self.track_map = []
        for i, emb in enumerate(self.embeddings):
            self.track_map.extend([(i, j) for j in range(emb.shape[0])])

    def process_file(self, fn):
        fn = Path(fn)
        emb_name = fn.stem + ".pt"
        emb_path = self.embeddings_dir / emb_name[:3] / emb_name

        emb_key = emb_name[:3] + "/" + fn.stem
        beats = self.groundtruth[emb_key]

        # If the embedding exists, load and process it
        try:
            if emb_path.exists():
                embedding = torch.load(emb_path, map_location="cpu")
                embedding = self.prepare_embedding(
                    embedding
                )  # Assuming this is an instance method

                return embedding, beats, str(fn)
        except Exception as e:
            print(f"Error processing {fn}: {e}")

        return None, None, None

    def parallel_loading(self, filenames):
        embeddings, beats, fns = [], [], []

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.process_file, fn): fn for fn in filenames}

            for future in tqdm(as_completed(futures), total=len(filenames)):
                embedding, beat, fn = future.result()
                if embedding is not None:
                    embeddings.append(embedding)
                    beats.append(beat)
                    fns.append(fn)

        print(f"Loaded {len(embeddings)} embeddings and labels.")
        return embeddings, beats, fns

    def __len__(self):
        return len(self.track_map)

    def __getitem__(self, idx):
        """Loads the labels and the processed embeddings for a given index."""

        track_id, chunk_id = self.track_map[idx]

        # print(f"Processing index {idx}.")

        embeddings = self.embeddings[track_id][chunk_id]  # (T, F)
        beats = self.beats[track_id]

        T, _ = embeddings.shape
        labels = torch.zeros(T, dtype=torch.float32, device=embeddings.device)

        sr = 24000
        chunk_hop = T // 2
        frame_hop = 320
        n_frames = 4

        timestamps = torch.arange(T) + chunk_id * chunk_hop
        timestamps = timestamps * frame_hop * n_frames / sr

        t_min = timestamps[0]
        t_max = timestamps[-1]
        # print(f"t_min: {t_min:.4f}, t_max: {t_max:.4f}")

        for beat in beats:
            if beat < t_min:
                continue
            if beat > t_max:
                break

            dist = np.abs(beat - timestamps)
            idx = np.argmin(dist)

            labels[idx] = 1

            # if idx > 0:
            #     labels[idx - 1] = 1
            # if idx < T - 1:
            #     labels[idx + 1] = 1

            # if dist[idx] > thres:
            #     print(f"Beat {beat} is far from the closest timestamp.")
            #     print(f"Closest timestamp: {timestamps[idx]:.4f}")
            #     print(f"Distance: {dist[idx]:.4f}")

        # embeddings = embeddings.flatten()  # (T*F, )

        labels.unsqueeze_(1)  # (T, 1)

        return embeddings, labels, self.filenames[track_id]

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
            # embeddings = embeddings.view(-1, F)  # (N*T, F)
            pass  # (N, T, F)
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


@gin.configurable
class BeatTrackingEmbeddingLoadingDataModule(L.LightningDataModule):
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
        with open(train_filelist, "r") as f:
            filename = f.readline().strip()
            filename = Path(filename)

        emb_sub = filename.stem[:3]
        emb_fn = filename.stem + ".pt"
        emb_path = self.embeddings_dir / emb_sub / emb_fn

        embedding = torch.load(emb_path, map_location="cpu")

        # Get the dimensions of the embeddings
        self.timestamps = embedding.shape[2]
        self.embedding_dimension = embedding.shape[3]

        print(f"Number of timestamps: {self.timestamps}")
        print(f"Embedding dimension: {self.embedding_dimension}")

        print("\nSetting up Train dataset...")
        self.train_dataset = BeatTrackingEmbeddingLoadingDataset(
            self.embeddings_dir,
            self.train_filelist,
            self.gt_path,
            self.layer_aggregation,
            self.granularity,
            self.time_aggregation,
            mode="train",
        )
        print("\nSetting up Validation dataset...")
        self.val_dataset = BeatTrackingEmbeddingLoadingDataset(
            self.embeddings_dir,
            self.val_filelist,
            self.gt_path,
            self.layer_aggregation,
            self.granularity,
            self.time_aggregation,
            mode="val",
        )

        print("Setting up the Test dataset...")
        self.test_dataset = BeatTrackingEmbeddingLoadingDataset(
            self.embeddings_dir,
            self.test_filelist,
            self.gt_path,
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
