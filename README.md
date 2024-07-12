# ssl-mtg: MTG self-supervised learning models

A repository of models and training code for several SSL aproaches and architectures.

## Install

```
pip install -e .[dev]
```

## Cluster setup 

wandb without internet
```bash
echo "export WANDB_MODE=offline" >> .bashrc 
```

To run an interactiver job
```bash
srun --partition=interactive --account=upf97 --qos=acc_interactive --gres=gpu:1  srun --partition=interactive --account=upf97 --qos=acc_interactive --gres=gpu:1 --cpus-per-task=20 --time=02:00:00 --pty /bin/bash
```

### if you are saving a new environment

Let's say you have a conda environment called `mtg-bsc` and you want to use this environment in the cluster. First you need to
save the environment in local
```bash
conda pack -n mtg-bsc
```

load the eviroment at bsc

```bash
module load anaconda
mkdir -p mtg-bsc
tar -xzf mtg-bsc.tar.gz -C mtg-bsc
source mtg-bsc/bin/activate
python
```
After that you save the eviroment in /gfs/projects/upf97/envs/mtg-bsc
    
```bash
cp -r mtg-bsc /gpfs/projects/upf97/envs/
```
### if you are using a eviroment already saved in the cluster

The first time you use the environment you need to copy it from the shared folder to your home folder.
```bash
cp -r /gpfs/projects/upf97/envs/mtg-bsc ~/ssl-mtg/
```

From that instant you can run it with the following command from `~/ssl-mtg/`

```bash
module load anaconda
source mtg-bsc/bin/activate
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

