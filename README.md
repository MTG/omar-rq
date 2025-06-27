# OMAR-RQ: Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction

This repository contains training, validation, and inference code for various SSL approaches and architectures.

## Install

For embedding extraction or fine-tuning:

```bash
pip install .

```

For development including pre-training your own models:

```bash
pip install -e .[dev]
```

## Inference

Load a model based on its [[#Pre-trained models| Pre-trained model ID]]:

```python
import torch
from omar_rq import get_model

# Embedding extraction example
x = torch.randn(1, 16000 * 4).cpu()
model_id = "mtg-upf/omar-rq-multifeature-25hz-fsq"

model = get_model(model_id=model_id, device="cpu")

embeddings = model.extract_embeddings(x, layers=[6])

timestamps = torch.arange(embeddings.shape[2]) / model.eps
```

`get_model` reference:

```
Returns an OMAR-RQ Module from the provided  model_id or config_file.

Args:
    model_id (str): Hugging Face's Model ID or local path to the model
    config_file (Path): Path to the model config of a trained model.
    device (str): Device to use for the model. Defaults to "cpu".
    quantization_targets (bool): If True, it will create the quantization
        targets for SSL pre-training of the model. Defaults to False.

Output:
    module: The model from the provided config file.


Module usage:

Args:
    audio (torch.Tensor): 2D mono audio tensor (B, T'). Where B is
        the batch size and T' is the number of samples.
    layers (set): Set of layer indices to extract embeddings from.
        By default, it extracts embeddings from the last layer (logits).

Output:
    torch.Tensor: Extracted embeddings. The output tensor has shape
        (L, B, T, C,) where L = len(layers), B is the batch size, T is
        the number of output timestamps, and C = embedding dimension.


Example:

>>> x = torch.randn(1, 16000 * 4).cpu()
>>>
>>> model = get_model(config_file, device="cpu")
>>>
>>> embeddings = model.extract_embeddings(x, layers=(6))
>>>
>>> # use the `eps` field to compute timestamps
>>> timestamps = torch.arange(embeddings.shape[2]) / model.eps



>> NOTE: The model's embedding rate depends on the model's configuration.
    For example, the melspectrogram model has an embedding rate of 16ms.
    audio should be a sequence with a sample rate as inditacted in the
    config file and up to 30s.
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

## Available models

| Model                    | Input | Rate   | Tagging | Difficulty | Pitch   | Chord   | Beat   | Structure |
|--------------------------|--------|--------|---------|------------|---------|---------|--------|-----------|
|                          |        | Hz     | *mAP*   | *MSE*      | *acc.*  | *acc.*  | *F1*   | *acc.*    |
| **base**                 | mel    | 15.63 | .482    | **1.65**   | .892    | .657    | .783   | **.647**  |
| **multicodebook**        | mel    | 15.63 | **.488**    | 1.66       | .897    | .675    | .775   | .639      |
| **multifeature**         | audio  | 18.75  | .467    | 1.76       | .938    | .734    | .833   | .623      |
| **multifeature-25hz**    | audio  | 25     | .463    | 1.79       | .932    | .728    | .848   | .628      |
| **multifeature-25hz-fsq**| audio  | 25     | .463    | 1.71       | **.940**| **.749**| **.855**   | .628      |

### Hugging Face Model IDs

- [mtg-upf/omar-rq-base](https://huggingface.co/mtg-upf/omar-rq-base)
- [mtg-upf/omar-rq-multicodebook](https://huggingface.co/mtg-upf/omar-rq-multicodebook)
- [mtg-upf/omar-rq-multifeature](https://huggingface.co/mtg-upf/omar-rq-multifeature)
- [mtg-upf/omar-rq-multifeature-25hz](https://huggingface.co/mtg-upf/omar-rq-multifeature-25hz)
- [mtg-upf/omar-rq-multifeature-25hz-fsq](https://huggingface.co/mtg-upf/omar-rq-multifeature-25hz-fsq)

## Pre-training OMAR-RQ models

1. Install development dependencies:

```bash
pip install -e .[dev]
```

2. Prepare the experiment data

We downsample our data to 16 kHz mono and store it as 16-bit raw bytes ([numpy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) files).
Check our [data preparation scripts](data/).

3. Configuration

Our experiment configuration is controlled with [gin-config](https://github.com/google/gin-config).
Check the default [config file](../cfg/rq_single_view/config.gin) to see the different parameters that can be modified.

At least the following parameters should be modified:

- `DiscotubeMultiViewAudioDataModule.data_dir` -> This should point to your base data folder.
- `DiscotubeMultiViewAudioDataModule.filelist_train` -> This should point to a filelist of training audio paths relative to the `data_dir` (one audio file per line).
- `DiscotubeMultiViewAudioDataModule.filelist_val` -> Same for the tracks on the validation split.

4. Run the experiment

```bash
python src/train.py cfg/rq_single_view/config.gin
```

## Licensing information

The code in this repository is available under [AGPL-3.0 license](https://www.gnu.org/licenses/agpl-3.0.en.html) license.
The model weights are available under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license for non-commercial applications.
[Contact us](https://www.upf.edu/web/mtg/contact) for more information.
