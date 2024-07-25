
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
After that, save the environment in `/gpfs/projects/upf97/envs/`
    
```bash
cp -r mtg-bsc /gpfs/projects/upf97/envs/
```
### if you are using an environment already saved in the cluster


To activate the environment in the cluster:

```bash
module load anaconda
source /gpfs/projects/upf97/envs/mtg-bsc/mtg-bsc/bin/activate
```

Or optionally copy it to your home folder

```bash
cp -r /gpfs/projects/upf97/envs/mtg-bsc ~/ssl-mtg/
```

And activated it from `~/ssl-mtg/` like this:

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

BSC nodes do not have internet access so wandb's logs cannot be updated directly.
To address this we will use our local machine as a proxy to log the experiments.

To do this follow the following steps in your local machine:
1. Install wandb `pip install wandb`
2. Login to wandb `wandb login`
3. Assuming that `alogin1` is an ssh alias set up with passwordless access to the cluster, run the following command to copy the cluster logs to your local machine and sync them to wandb `rsync -avz alogin1:/gpfs/projects/upf97/logs/wandb . && wandb sync --sync-all wandb/`.
Additionally, you can create a background process to keep the logs updated `watch -n 20 "rsync -avz alogin1:/gpfs/projects/upf97/logs/wandb . && wandb sync --sync-all wandb/"`.



## Prepare the data experiment
In the `config_file` modify these parameters:

- `DiscotubeMultiViewAudioDataModule.data_dir = "/scratch/palonso/data/discotube-2023-03/"` -> This should point to your base data folder.
- `DiscotubeMultiViewAudioDataModule.filelist_train = "data/discotube/train_v1.txt"` -> This should point to a filelist of training audio paths relative to the `data_dir` (one audio file per line).
- `DiscotubeMultiViewAudioDataModule.filelist_val = "data/discotube/val_v1.txt"` -> Same. Wih the tracks for the validation split.


## Run the experiment

- `python src/train.py`

