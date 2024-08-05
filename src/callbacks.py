import os
from pathlib import Path

from pytorch_lightning.callbacks import Callback

from nets.transformer import Transformer


class GinConfigSaverCallback(Callback):
    """Callback to save the gin config file to the checkpoints directory.
    It is not the most elegant way of using gin, but I could not find a better
    solution."""

    def __init__(self, train_config_path: Path):
        """Initialize the callback with the path to the training gin config file.
        This config file must have the model_config.gin path in the 2nd line of
        the file.
        """

        super().__init__()

        # Store the path to the gin config file
        self.train_config_path = train_config_path

        # Read the training gin config file once when the training starts
        with open(self.train_config_path, "r") as f:
            self.train_config = f.read()

        # Get the model config path from the train config
        #  NOTE: IT MUST BE STORED IN THE 2ND LINE OF THE TRAIN CONFIG
        train_config_lines = self.train_config.split("\n")
        self.model_config_path = Path(
            train_config_lines[1]
            .replace("include ", "")
            .replace(" ", "") # just in case
            .replace("'", "") # required
        )

        # Remove the model config path from the train config since it is relative
        train_config_lines = train_config_lines[3:]
        self.train_config = "\n".join(train_config_lines) + "\n"

        # Read the model config file as text
        with open(self.model_config_path, "r") as f:
            self.model_config = f.read()

        # If the model config contains a ckpt path remove it
        # otherwise there will be 2 build_module.ckpt_path lines
        if "build_module.ckpt_path" in  self.model_config:
            model_config_lines = self.model_config.split("\n")
            self.model_config = "\n".join(model_config_lines[3:]) + "\n"

        # If nets.transformer.Transformer.num_patches in model config, remove it
        # otherwise there will be 2 lines
        if "nets.transformer.Transformer.num_patches" in self.model_config:
            model_config_lines = self.model_config.split("\n")
            # Find the lines and remove them
            for i, line in enumerate(model_config_lines):
                if "nets.transformer.Transformer.num_patches" in line:
                    model_config_lines[i] = ""
            # Create the new config as a string
            self.model_config = "\n".join(model_config_lines) + "\n"

    def on_train_start(self, trainer, pl_module):
        """Read the gin config file when the training starts once."""

        # This is where wandb logger saves the checkpoint
        # Needs the training to be started
        self.ckpt_dir = Path(os.path.join(
            trainer.logger.save_dir,
            trainer.logger.name,
            trainer.logger.version,
            "checkpoints",
        )).resolve() # convert it to a full path
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Create the new model config's path
        self.new_model_config_path = self.ckpt_dir / self.model_config_path.name

        # The training gin config will be saved here
        self.new_train_config_path = self.ckpt_dir / self.train_config_path.name

        # Update the train config with the abspath of the new model config
        train_config = (
            "# Load the model configuration. Do NOT change the location of this line!\n"
            + f"include '{self.new_model_config_path}'\n\n"
            + self.train_config
        )
        # Save the updated train config
        with open(self.new_train_config_path, "w") as f:
            f.write(train_config)

    def on_train_epoch_end(self, trainer, pl_module):
        """Save the gin config file in the checkpoints directory, appending the current 
        checkpoint path."""

        # Create the full path to the latest checkpoint
        ckpt_name = f"epoch={trainer.current_epoch}-step={trainer.global_step}.ckpt"
        ckpt_path = self.ckpt_dir / ckpt_name

        # Update the model config with the current checkpoint path
        model_config = (
            "# Model checkpoint path. Do NOT change the location of this line!\n"
            + f"build_module.ckpt_path = '{ckpt_path}'\n\n"
            + self.model_config
        )

        # If the model is a transformer, add the number of patches to the gin config
        if isinstance(pl_module.net, Transformer):
            # Clean the old num_patches lines if they exist
            if "nets.transformer.Transformer.num_patches" in model_config:
                model_config_lines = model_config.split("\n")
                for i, line in enumerate(model_config_lines):
                    if "nets.transformer.Transformer.num_patches" in line:
                        model_config_lines[i] = ""
                model_config = "\n".join(model_config_lines)
            # Add the new num_patches line
            model_config = (
                model_config
                + f"\nnets.transformer.Transformer.num_patches = {pl_module.net.num_patches}\n"
            )

        # Save the updated model config
        with open(self.new_model_config_path, "w") as f:
            f.write(model_config)
