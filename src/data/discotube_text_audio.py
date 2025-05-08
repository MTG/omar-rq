import gin
import json
import random
import traceback
import warnings
from pathlib import Path
from typing import Union

import torch
import pytorch_lightning as L
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader


from .data_utils import AudioDataset


@gin.configurable
class DiscotubeTextAudioDataset(AudioDataset):
    """Generic audio dataset."""

    def __init__(
        self,
        data_dir: Path,
        filelist: Path,
        metadata_youtube: dict,
        metadata_discogs: dict,
        metadata_id_map: dict,
        metadata_dropout: float,
        is_training: bool,
        frame_offset: Union[int, str] = "random",
    ):
        super().__init__(
            data_dir=data_dir,
            filelist=filelist,
            frame_offset=frame_offset,
        )

        self.metadata_youtube = metadata_youtube
        self.metadata_discogs = metadata_discogs
        self.metadata_id_map = metadata_id_map

        self.metadata_dropout = metadata_dropout
        self.is_training = is_training

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def get_audio_path(youtube_id: str) -> Path:
        return Path(youtube_id[:2], youtube_id).with_suffix(".mmap")

    def __getitem__(self, idx):
        try:
            file_path = self.filelist[idx]
            file_path = self.data_dir / file_path

            id_yt = file_path.stem

            # load audio
            audio = self.load_audio(file_path, frame_offset=self.frame_offset)

            # load YouTube metadata
            meta_youtube = self.metadata_youtube[id_yt]

            # load discogs metadata
            ids_discogs = self.metadata_id_map[id_yt]

            # sample randonly among available releases
            id_discogs = random.choice(ids_discogs)
            meta_discogs = self.metadata_discogs[id_discogs]

            # process metadata
            text = self.preprocess_text(
                {"youtube_metadata": meta_youtube, "discogs_metadata": meta_discogs}
            )
        except Exception:
            print(f"Error loading {self.filelist[idx]}")
            print(traceback.format_exc())
            return [None, None]

        return [audio, text]

    def preprocess_text(self, metadata: dict) -> str:
        """Text preprocessing"""

        # Process YouTube metadata
        fields_to_keep = ["description", "categories", "tags", "view_count"]
        youtube_metadata = metadata["youtube_metadata"]
        new_youtube_metadata = {
            field: youtube_metadata[field]
            for field in fields_to_keep
            if field in youtube_metadata and youtube_metadata[field] != ""
        }

        # Process Discogs metadata

        fields_to_keep = ["labels", "genres", "styles", "country", "released"]
        dicogs_metadata = metadata["discogs_metadata"]
        new_discogs_metadata = {
            field: dicogs_metadata[field]
            for field in fields_to_keep
            if field in dicogs_metadata and dicogs_metadata[field] != ""
        }

        # Fetch artist description
        # TODO: Get this too

        metadata = {
            "youtube_metadata": new_youtube_metadata,
            "discogs_metadata": new_discogs_metadata,
        }

        # Discard one random metadata field
        if random.random() < self.metadata_dropout and self.is_training:
            metadata_keys = list(metadata.keys())
            remove_key = random.choice(metadata_keys)
            metadata.pop(remove_key)

        # format as YAML
        yaml_text = yaml.dump(metadata, sort_keys=False)
        return yaml_text


@gin.configurable
class DiscotubeTextAudioDataModule(L.LightningDataModule):
    """AudioDataModule for the Discogs dataset."""

    def __init__(
        self,
        batch_size: int,
        data_dir: Path,
        filelist_train: Path,
        filelist_val: Path,
        metadata_youtube_file: Path,
        metadata_discogs_file: Path,
        metadata_id_map_file: Path,
        num_workers: int,
    ):
        super().__init__()

        self.batch_size = batch_size

        self.data_dir = Path(data_dir)
        self.filelist_train = Path(filelist_train)
        self.filelist_val = Path(filelist_val)

        self.num_workers = num_workers

        self.metadata_youtube_file = metadata_youtube_file
        self.metadata_discogs_file = metadata_discogs_file
        self.metadata_id_map_file = metadata_id_map_file

    def setup(self, stage: str):
        # load YouTube metadata from jsonl (one json object per line)
        self.metadata_youtube = dict()
        with open(self.metadata_youtube_file, "r") as f:
            for line in tqdm(f.readlines(), desc="Loading YouTube metadata"):
                line = json.loads(line)
                self.metadata_youtube[line["id"]] = line

        # load Discogs metadata from jsonl (one json object per line)
        self.metadata_discogs = dict()
        with open(self.metadata_discogs_file, "r") as f:
            for line in tqdm(f.readlines(), desc="Loading Discogs metadata"):
                line = json.loads(line)
                self.metadata_discogs[line["id"]] = line

        # load the id map from jsonl (one json object per line)
        self.metadata_id_map = dict()
        with open(self.metadata_id_map_file, "r") as f:
            for line in tqdm(f.readlines(), desc="Loading ID map"):
                line = json.loads(line)
                for k, v in line.items():
                    self.metadata_id_map[k] = v

        self.dataset_train = DiscotubeTextAudioDataset(
            self.data_dir,
            filelist=self.filelist_train,
            metadata_youtube=self.metadata_youtube,
            metadata_discogs=self.metadata_discogs,
            metadata_id_map=self.metadata_id_map,
            is_training=True,
        )
        self.dataset_val = DiscotubeTextAudioDataset(
            self.data_dir,
            filelist=self.filelist_val,
            metadata_youtube=self.metadata_youtube,
            metadata_discogs=self.metadata_discogs,
            metadata_id_map=self.metadata_id_map,
            is_training=False,
        )

    def collate_fn(self, batch):
        """Custom collate function to skip data loader errors"""

        audio_in, text_in = zip(*batch)

        audio = [i for i in audio_in if i is not None]
        text = [i for i in text_in if i is not None]

        audio = torch.stack(audio)

        if len(audio) < len(audio_in) // 2:
            warnings.warn(
                f"Skipping {len(audio_in) - len(audio)} samples out if {len(audio_in)} in collate_fn "
            )

        return audio, text

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
