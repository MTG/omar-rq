import os

from pytorch_lightning.callbacks import Callback

from nets.transformer import Transformer


class GinConfigSaverCallback(Callback):
    """Callback to save the gin config file to the checkpoints directory.
    It is not the most elegant way of using gin, but I could not find a better
    solution."""

    def __init__(self, train_config_path):
        """Initialize the callback with the path to the training gin config file.
        This config file must have the model_config.gin path in the 2nd line of
        the file.
        """

        super().__init__()

        # Store the path to the gin config file
        self.train_config_path = train_config_path

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

        # Read the training gin config file once when the training starts
        with open(self.train_config_path, "r") as f:
            train_config = f.read()

        # Get the model config path from the gin config
        train_config_lines = train_config.split("\n")
        model_config_path = (
            train_config_lines[1]
            .replace("include ", "")
            .replace(" ", "")
            .replace("'", "")
        )
        with open(model_config_path, "r") as f:
            self.model_config = f.read()

        # Save the model config to ckpt_dir
        self.new_model_config_path = os.path.abspath(
            os.path.join(self.ckpt_dir, os.path.basename(model_config_path))
        )
        with open(self.new_model_config_path, "w") as f:
            f.write(self.model_config)

        # Remove the model config path from the train config
        self.train_config = "\n".join(train_config_lines[3:]) + "\n"

        # The training gin config will be saved here at the end of each epoch
        self.new_train_config_path = os.path.abspath(
            os.path.join(self.ckpt_dir, os.path.basename(self.train_config_path))
        )

    def on_train_epoch_end(self, trainer, pl_module):
        """Save the gin config file in the checkpoints directory,
        appending the current checkpoint path."""

        # Create the full path to the checkpoint
        ckpt_name = f"epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
        ckpt_path = os.path.abspath(os.path.join(self.ckpt_dir, ckpt_name))

        # Update the train config with the current checkpoint path and the abspath
        # to the model config
        train_config = (
            "# Model checkpoint path\n"
            + f"checkpoint_path = '{ckpt_path}'\n\n"
            + "# Load the model configuration. Do NOT change the location of these lines!\n"
            + f"include '{self.new_model_config_path}'\n\n"
            + self.train_config
        )
        # Save the updated train config
        with open(self.new_train_config_path, "w") as f:
            f.write(train_config)

        # If the model is a transformer, add the number of patches to the gin config
        if isinstance(pl_module.net, Transformer):
            model_config = (
                self.model_config
                + f"\nnets.transformer.Transformer.num_patches = {pl_module.net.num_patches}\n"
            )
            # Save the updated model config
            with open(self.new_model_config_path, "w") as f:
                f.write(model_config)
