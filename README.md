
A repository of models and training code for several SSL aproaches and architectures.

## Install

```
pip install -e .[dev]
```

## Cluster setup 

To run an interactiver job
```bash
srun --partition=interactive --account=upf97 --qos=acc_interactive --gres=gpu:1 --cpus-per-task=20 --time=02:00:00 --pty /bin/bash
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

### Using wandb in the BSC cluster

BSC nodes do not have internet access so wandb logs cannot be updated directly.
To address this we will use our local machine as a proxy to log the experiments.

To do this follow the following steps in your local machine:
1. Install wandb `pip install wandb`
2. Login to wandb `wandb login`
3. Run the following command to sync the cluster logs to your local machine and sync them to wandb `rsync -a alonging1:/gpfs/scratch/upf97/wandb/logs/ .; wandb sync .`.
Additionally, you can create a background process to keep the logs updated `watch -n 10 "rsync -a alonging1:/gpfs/scratch/upf97/wandb/logs/ .; wandb sync ."`.



## Prepare the data experiment
In the `config_file` modify these parameters:

- `DiscotubeMultiViewAudioDataModule.data_dir = "/scratch/palonso/data/discotube-2023-03/"` -> This should point to your base data folder.
- `DiscotubeMultiViewAudioDataModule.filelist_train = "data/discotube/train_v1.txt"` -> This should point to a filelist of training audio paths relative to the `data_dir` (one audio file per line).
- `DiscotubeMultiViewAudioDataModule.filelist_val = "data/discotube/val_v1.txt"` -> Same. Wih the tracks for the validation split.


## Run the experiment

- `python src/train.py`

