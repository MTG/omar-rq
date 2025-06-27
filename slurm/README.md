
# BSC MareNostrum5 SLURM cluster setup

This sections contains instructions to run the experiments in the [BSC MareNostrum](https://www.bsc.es/supportkc/docs/MareNostrum5/intro) cluster.

## Running interactive jobs

To run an interactive job

```bash
srun --partition=interactive --account=upf97 --qos=acc_interactive --gres=gpu:1 --cpus-per-task=20 --time=02:00:00 --pty /bin/bash
```

## Conda environments

Since the BSC cluster does not have internet access, we operate by creating and transferring Conda environments manually.

### Creating a Conda environment

Let's say you have a Conda environment called `mtg-bsc` and you want to use it in the cluster.

1. Save the local environment

    ```bash
    # This will create `mtg-bsc.tar.gz`
    conda pack -n mtg-bsc
    ```

1. Transfer this packed environment to the cluster

    ```bash
    scp mtg-bsc.tar.gz <USERID>@transfer1.bsc.es:~/
    ```

1. Load the environment at BSC.

    ```bash
    mkdir -p mtg-bsc
    tar -xzf mtg-bsc.tar.gz -C mtg-bsc
    module load anaconda
    source mtg-bsc/bin/activate
    python
    ```

1. Finally, save the environment to `/gpfs/projects/upf97/envs/`

    ```bash
    cp -r mtg-bsc /gpfs/projects/upf97/envs/
    ```

### Using an environment already available in the cluster

To activate the environment `mtg-bsc` in the cluster:

```bash
module load anaconda
source /gpfs/projects/upf97/envs/mtg-bsc/bin/activate
```

Or optionally copy it to your home folder:

```bash
cp -r /gpfs/projects/upf97/envs/mtg-bsc ~/ssl-mtg/
```

And activate it from `~/ssl-mtg/` like this:

```bash
module load anaconda
source mtg-bsc/bin/activate
```

## Experiment tracking

We use [wandb](https://docs.wandb.ai/) for the experiment tracking.
Make sure you are [registered](https://docs.wandb.ai/quickstart#2-log-in-to-wb) to log the experiment online.

### Using wandb in the BSC cluster

BSC nodes do not have internet access so wandb's logs cannot be updated directly.
To address this we will use our local machine as a proxy to log the experiments.

To do this follow these steps in your local machine:

1. Install wandb `pip install wandb`
1. Login to wandb `wandb login`
1. Assuming that `alogin1` is an ssh alias set up with passwordless access to the cluster, run the following command to copy the cluster logs to your local machine and sync them to wandb `rsync -avz alogin1:/gpfs/projects/upf97/logs/wandb . && wandb sync --sync-all wandb/`.
Additionally, you can create a background process to keep the logs updated `watch -n 20 "rsync -avz alogin1:/gpfs/projects/upf97/logs/wandb . && wandb sync --sync-all wandb/"`.
