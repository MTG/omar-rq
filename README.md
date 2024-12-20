# ssl-mtg: MTG self-supervised learning models

A repository of models and training code for several SSL aproaches and architectures.

## Install

For embedding extraction or downstream tasks:
```bash
pip install -e .

```

For development including pre-training models:
```bash
pip install -e .[dev]
```

## Inference

This repository provides a simple interface to extract embeddings given a model configuration file (`.gin`).

Embedding extraction example:
```python 
x = torch.randn(1, 16000 * 4).cpu()
config_file = "my_config_file.gin"

model, eps = get_model(config_file, device="cpu")

embeddings = model.extract_embeddings(x, layers=(6))

timestamps = torch.arange(embeddings.shape[2]) / eps
```

`get_model` reference:
```
Returns the model from the provided config file.

Args:
    config_file (Path): Path to the config file of a trained model.
    device (str): Device to use for the model. Defaults to "cpu".

Output:
    module: The model from the provided config file.
    eps (float): Embeddings per second.
        e.g., torch.arange(T) / eps gives the timestamps of the embeddings.
```

`extract_embeddings` reference
```

Args:
    audio (torch.Tensor): 1D audio tensor.
    layers (set): Set of layer indices to extract embeddings from.
        By default, it extracts embeddings from the last layer.

Output:
    torch.Tensor: Extracted embeddings.
        Even in the case of aggregation or single layer embeddings,
        the output tensor will have the same shape (L, B, T, C,)
        where L = len(layer), B is the number of chunks
        T is the number of melspec frames the model can accomodate
        C = model output dimension. No aggregation is applied.
        audio: torch.Tensor of shape (batch_size, num_samples)
        embeddings: torch.Tensor of shape (lbatch_size, timestamps, embedding_size)
```

## Cluster setup

To run an interactive job

```bash
srun --partition=interactive --account=upf97 --qos=acc_interactive --gres=gpu:1 --cpus-per-task=20 --time=02:00:00 --pty /bin/bash
```

### If you are saving a new environment

Let's say you have a Conda environment called `mtg-bsc` and you want to use this environment in the cluster.

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

### If you are using an environment already saved in the cluster

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

## Configure

All the experiment configuration is controled with gin-config.
Check the default [config file](cfg/config.gin).

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

## Prepare the data experiment

In the `config_file` modify these parameters:

- `DiscotubeMultiViewAudioDataModule.data_dir = "/scratch/palonso/data/discotube-2023-03/"` -> This should point to your base data folder.
- `DiscotubeMultiViewAudioDataModule.filelist_train = "data/discotube/train_v1.txt"` -> This should point to a filelist of training audio paths relative to the `data_dir` (one audio file per line).
- `DiscotubeMultiViewAudioDataModule.filelist_val = "data/discotube/val_v1.txt"` -> Same. With the tracks for the validation split.

> [!NOTE]
> In the context of the BSC MareNostrum cluster experiments, new filelists should be located in the common project folder so that every user can access them: `/gpfs/projects/upf97/data/`.

## Run the experiment

- `python src/train.py`
