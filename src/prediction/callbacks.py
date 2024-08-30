from pathlib import Path
import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class EmbeddingWriter(BasePredictionWriter):

    def __init__(self, output_dir: Path, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):

        # Get the audio and audio path
        _, audio_path = batch
        audio_name = audio_path.stem
        _output_dir = self.output_dir / audio_name[:3]
        _output_dir.mkdir(parents=True, exist_ok=True)
        output_path = _output_dir / f"{audio_name}.pt"

        # If the prediction is a tensor, write it to a file
        if prediction is not None:
            # NOTE: this will remember the device the tensor was on
            # and will restore it when loading the tensor
            # maybe we should move to CPU before saving
            torch.save(prediction, output_path)
