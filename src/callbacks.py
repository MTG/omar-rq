import os

from pytorch_lightning.callbacks import Callback

from nets.transformer import Transformer


class GinConfigSaverCallback(Callback):
    """Callback to save the gin config file in the checkpoints directory."""

    def __init__(self, gin_config_path):
        super().__init__()

        # Store the path to the gin config file
        self.gin_config_path = gin_config_path

    def on_train_start(self, trainer, pl_module):
        """Read the gin config file when the training starts once."""

        # This is where wandb logger saves the checkpoint
        self.ckpt_dir = os.path.join(
            trainer.logger.save_dir,
            trainer.logger.name,
            trainer.logger.version,
            "checkpoints",
        )
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Read the gin config file once when the training starts
        with open(self.gin_config_path, "r") as f:
            self.gin_config = f.read()

    def on_train_epoch_end(self, trainer, pl_module):
        """Save the gin config file in the checkpoints directory,
        appending the current checkpoint path."""

        # Get the full path to the checkpoint
        ckpt_name = f"epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
        ckpt_path = os.path.abspath(os.path.join(self.ckpt_dir, ckpt_name))

        # Update the gin config file with the current checkpoint path
        gin_config = (
            "# Model checkpoint path\n"
            + f"checkpoint_path = '{ckpt_path}'\n\n"
            + self.gin_config
        )

        # If the model is a transformer, add the number of patches to the gin config
        if isinstance(pl_module.net, Transformer):
            gin_config += f"\nnets.transformer.Transformer.num_patches = {pl_module.net.num_patches}\n"

        # Save the updated gin config file
        with open(
            os.path.join(self.ckpt_dir, os.path.basename(self.gin_config_path)), "w"
        ) as f:
            f.write(gin_config)
