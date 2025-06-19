# OMAR-RQ: Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction

This repository contains training, validation, and inference code for various SSL approaches and architectures.

## Install

For embedding extraction or downstream tasks:

```bash
pip install .

```

For development including pre-training models:

```bash
pip install -e .[dev]
```

## Inference

This repository provides a simple interface to extract embeddings given a model configuration file (`.gin`).

Embedding extraction example:

```python
import torch
from omar_rq import get_model

x = torch.randn(1, 16000 * 4).cpu()
model_id = "mtg-upf/omar-rq-multifeature-25hz-fsq"

model = get_model(model_id=model_id, device="cpu")

embeddings = model.extract_embeddings(x, layers=[6])

timestamps = torch.arange(embeddings.shape[2]) model.eps
```

`get_model` reference:

```
Returns the model from the provided config file.

Args:
    config_file (Path): Path to the config file of a trained model.
    device (str): Device to use for the model. Defaults to "cpu".
    encodec_weights_path (str): Path to the EnCodec weights. When set, it will
        override the value in the config file. Note that it can be a local path
        or or a Hugging Face model ID. This parameter only affects to models
        that use EnCodec as representation. Defaults to None.

        https://huggingface.co/docs/transformers/en/model_doc/encodec

Output:
    module: The model from the provided config file.
    eps (float): Embeddings per second.
        e.g., torch.arange(T) / eps gives the timestamps of the embeddings.
```

`extract_embeddings` reference:

```
Extract embeddings from an input audio batch.

Args:
    audio (torch.Tensor): 2D mono audio tensor (B, T'). Where B is 
        the batch size and T' is the number of samples.
    layers (set): Set of layer indices to extract embeddings from.
        By default, it extracts embeddings from the last layer (logits).

Output:
    torch.Tensor: Extracted embeddings. The output tensor has shape 
        (L, B, T, C,) where L = len(layers), B is the batch size, T is
        the number of output timestamps, and C = embedding dimension.
```

## Relevant pre-trained models

- mtg-upf/omar-rq-base
- mtg-upf/omar-rq-multicodebook
- mtg-upf/omar-rq-multifeature
- mtg-upf/omar-rq-multifeature-25hz
- mtg-upf/omar-rq-multifeature-25hz-fsq

## Development list of models (left here for reference)

### Baseline Discogs23 models

| ID   | Arch      | Size  | Input | Target | WandB ID   | Steps  | MTAT  | Beattracking | CONFIG FILE PATH                                      |
|------|-----------|-------|-------|--------|------------|--------|-------|--------------|------------------------------------------------------|
| b0   | Conformer | Small | Mel   | Mel    | c4urat3s   | 400000 | 0.469 | 0.824        | c4urat3s/checkpoints/config_conformer.gin            |
| b1   | Conformer | Small | CQT   | CQT    | 8avrux47   | 305670 | 0.419 | 0.852        | 8avrux47/checkpoints/config_masking_conformer_small_cqt.gin |
| b3   | Conformer | Small | Enc   | Enc    | molbhb3i   | 326048 |       | 0.895        | molbhb3i/checkpoints/config_conformer_encodec.gin    |

### Multi-view Discogs23 models

| ID  | Arch       | Size  | Input  | Target        | WandB ID   | Steps  | MTAT  | Nsynth | Beattracking | CONFIG FILE PATH |
|-----|-----------|-------|--------|--------------|------------|--------|------|--------|--------------|------------------|
| s1  | Conformer | Small | audio  | au/mel/cqt   | hgu9kgyl   | 193591 | 0.440 | 0.910  |              | hgu9kgyl/checkpoints/config_masking_conformer_multiview_small.gin |
| s3  | Conformer | Small | enc    | enc          | adlpqsh3   | 366804 | 0.411 | 0.878  |              | adlpqsh3/checkpoints/config_masking_conformer_multiview_enc_to_enc_small.gin |
| s4  | Conformer | Small | enc    | enc/mel      | 6a8dzz68   | 366804 | 0.445 | 0.898  |              | 6a8dzz68/checkpoints/config_masking_conformer_multiview_enc_to_all_small.gin |
| s5  | Conformer | Small | enc    | enc/cqt      | lfc02r16   | 366804 | 0.433 | 0.892  |              | lfc02r16/checkpoints/config_masking_conformer_multiview_enc_to_cqt_small.gin |
| s6  | Conformer | Small | enc    | au/enc/cqt   | 9sn3yi5h   | 366804 | 0.433 | 0.889  |              | 9sn3yi5h/checkpoints/config_masking_conformer_multiview_enc_to_auenccqt_small.gin |
| s7  | Conformer | Small | enc    | au/enc/mel   | izet8ved   | 366804 | 0.444 | 0.895  |              | izet8ved/checkpoints/config_masking_conformer_multiview_enc_to_auencmel_small.gin |
| s8  | Conformer | Small | enc    | enc/mel/cqt  | bm23z5le   | 366804 | 0.446 | 0.900  | 0.750        | bm23z5le/checkpoints/config_masking_conformer_multiview_enc_to_encmelcqt_small.gin |
| l3  | Conformer | Large | enc    | enc          | z6opk2rz   | 366804 | NaN   | NaN    |              | z6opk2rz/checkpoints/config_masking_conformer_multiview_enc_to_enc_large.gin |
| l5  | Conformer | Large | enc    | enc/cqt      | mbgq9od4   | 366804 | 0.464 | 0.921  |              | mbgq9od4/checkpoints/config_masking_conformer_multiview_enc_to_cqt_large.gin |
| l6  | Conformer | Large | enc    | au/enc/cqt   | ldtuk0yo   | 366804 | 0.461 | 0.911  |              | ldtuk0yo/checkpoints/config_masking_conformer_multiview_enc_to_auenccqt_large.gin |
| l7  | Conformer | Large | enc    | au/enc/mel   | yc10xacz   | 366804 | NaN   | NaN    |              | yc10xacz/checkpoints/config_masking_conformer_multiview_enc_to_auencmel_large.gin |
| l8  | Conformer | Large | enc    | enc/mel/cqt  | 8bi35b82   | 366804 | NaN   | NaN    |              | 8bi35b82/checkpoints/config_masking_conformer_multiview_enc_to_encmelcqt_large.gin |

### FreeSound models

| ID   | Arch          | Size  | Input | Target | WandB ID   | Steps  | MTAT  | Nsynth | CONFIG FILE PATH                                      |
|------|---------------|-------|-------|--------|------------|--------|-------|--------------|------------------------------------------------------|
| f0   | Conformer     | Small | Mel   | Mel    | i2h5dqb8   | 343540 | 0.440 | 0.838        | i2h5dqb8/checkpoints/fs_config_masking_conformer_small.gin |
| f1   | Conformer     | Large | Mel   | Mel    | msesipur   | 364695 | 0.434 | 0.859        | msesipur/checkpoints/fs_config_masking_conformer_large.gin |

>[!NOTE] Most of these models are not available in the Hugging Face Hub.

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

## Licensing information

The code in this repository is available under [AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.en.html) license.
The model weights are available under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license for non-commercial applications.
[Contact us](https://www.upf.edu/web/mtg/contact) for more information.
