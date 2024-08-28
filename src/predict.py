import yaml
from argparse import ArgumentParser
from pathlib import Path
import traceback

import gin.torch
import pytorch_lightning as L

from data import DATASETS
from modules import MODULES
from nets import NETS
from utils import build_module
from prediction.callbacks import EmbeddingWriter
from prediction.dataset import AudioEmbeddingDataModule

# Register all modules, datasets and networs with gin
for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)


if __name__ == "__main__":
    parser = ArgumentParser("Evaluate SSL models using gin config")
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the model config of a trained model.",
    )
    parser.add_argument(
        "predict_config",
        type=Path,
        help="Path to the config file of the downstream task's dataset.",
    )

    args = parser.parse_args()

    try:

        # Load the test config
        with open(args.predict_config, "r") as f:
            predict_config = yaml.safe_load(f)

        # TODO: parse multiple gin configs. Or use YAML for test config
        # Convert the config string to a gin config
        gin.parse_config_file(args.train_config, skip_unknown=True)

        # Set the output directory with model name and dataset name
        ckpt_path = Path(gin.query_parameter("build_module.ckpt_path"))
        ssl_model_id = ckpt_path.parent.parent.name
        output_dir = (
            Path(predict_config["embeddings_dir"])
            / ssl_model_id
            / predict_config["dataset_name"]
        )

        # Writer callback
        callbacks = [EmbeddingWriter(output_dir)]

        # Need to use a trainer for model initialization
        trainer = L.Trainer(callbacks=callbacks, **predict_config["predict"]["device"])

        # Build the module and load the weights
        module, _ = build_module(trainer=trainer)

        gin.finalize()

        # Get the data module
        audio_config = {**predict_config["predict"]["audio"]}
        audio_config["data_dir"] = predict_config["data_dir"]
        data_module = AudioEmbeddingDataModule(**audio_config)

        trainer.predict(
            module,
            data_module,
        )

        print("Embedding extraction completed successfully.")

    except Exception:
        traceback.print_exc()
