import os
from pathlib import Path

import gin.torch
from pytorch_lightning.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

import gin.torch

from nets.transformer import Transformer


class GinConfigSaverCallback(Callback):
    """Callback to save the gin config file to the checkpoints directory.
    It is not the most elegant way of using gin, but I could not find a better
    solution."""

    def __init__(self, train_config_path: Path):
        """Initialize the callback with the path to the training gin config file."""

        super().__init__()

        # Store the path to the gin config file
        self.train_config_path = train_config_path

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Initialize the checkpoint directory and the new gin config path."""

        # This is where wandb logger saves the checkpoint
        # Needs the training to be started
        self.ckpt_dir = Path(
            os.path.join(
                trainer.logger.save_dir,
                trainer.logger.name,
                trainer.logger.version,
                "checkpoints",
            )
        ).resolve()  # convert it to a full path
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # The training gin config will be saved here
        self.new_train_config_path = self.ckpt_dir / self.train_config_path.name

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """Update the gin config and write it next to the latest checkpoint."""

        # Create the full path to the latest checkpoint
        ckpt_name = f"epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
        ckpt_path = self.ckpt_dir / ckpt_name
        print(f"Saving gin config with checkpoint path: {ckpt_path}")

        # Update the model config
        with gin.unlock_config():
            # Add the current checkpoint path
            gin.bind_parameter(
                "build_module.ckpt_path",
                str(ckpt_path),
            )

            # If the model is a transformer, add the number of patches to the gin config
            if isinstance(pl_module.net, Transformer):
                gin.bind_parameter(
                    "nets.transformer.Transformer.num_patches",
                    pl_module.net.num_patches,
                )

        # Save the config
        # NOTE: We do not use the operative_config_str() method as the parameters
        # we want to write are not *operative*
        with open(self.new_train_config_path, "w") as f:
            f.write(gin.config_str())
