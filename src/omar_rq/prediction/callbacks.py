from pathlib import Path
import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class EmbeddingWriter(BasePredictionWriter):
    def __init__(self, output_dir: Path, write_interval: str = "epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ):
        # Get the audio and audio path
        for audio_path, embeddings in pl_module.predict_data.items():
            audio_name = Path(audio_path).stem
            _output_dir = self.output_dir / audio_name[:3]
            _output_dir.mkdir(parents=True, exist_ok=True)
            output_path = _output_dir / f"{audio_name}.pt"

            # If the prediction is a tensor, write it to a file
            if embeddings is not None:
                try:
                    prediction = torch.stack(embeddings, dim=1)
                    torch.save(prediction, output_path)
                except Exception as e:
                    print(f"Error saving embeddings for {audio_name}: {e}")
