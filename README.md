# ssl-mtg: MTG self-supervised learning models

A repository of models and training code for several SSL aproaches and architectures.

## Install

```
pip install -e .[dev]
```

## Cluster setup 

We have to save ou wand key
```bash
export "WANDB_API_KEY"="YOUR_API_KEY" >> ~/.bashrc
```

To run an interactiver job
```bash
srun --partition=interactive --account=upf97 --qos=acc_interactive --gres=gpu:1  srun --partition=interactive --account=upf97 --qos=acc_interactive --gres=gpu:1 --cpus-per-task=20 --time=02:00:00 --pty /bin/bash
```



## Configure

All the experiment configuration is controle with gin-config.
Check the default [cofngi file](cfg/config.gin).


## Experiment tracking

We use [wandb](https://docs.wandb.ai/) for the experiment tracking.
Make sure you are [registered](https://docs.wandb.ai/quickstart#2-log-in-to-wb) to log the experiment online.


## Prepare the data experiment
In the `config_file` modify these parameters:

- `DiscotubeMultiViewAudioDataModule.data_dir = "/scratch/palonso/data/discotube-2023-03/"` -> This should point to your base data folder.
- `DiscotubeMultiViewAudioDataModule.filelist_train = "data/discotube/train_v1.txt"` -> This should point to a filelist of training audio paths relative to the `data_dir` (one audio file per line).
- `DiscotubeMultiViewAudioDataModule.filelist_val = "data/discotube/val_v1.txt"` -> Same. Wih the tracks for the validation split.


## Run the experiment

- `python src/train.py`
