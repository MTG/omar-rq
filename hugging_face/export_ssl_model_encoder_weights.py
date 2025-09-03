from pathlib import Path
from shutil import copyfile

import torch

from omar_rq import get_model

torch.manual_seed(2)


for model_id in [
    "omar-rq-multifeature-25hz-fsq",
    "omar-rq-base",
    "omar-rq-multifeature",
    "omar-rq-multicodebook",
    "omar-rq-multifeature-25hz",
]:
    print(f"Processing model {model_id}...")

    cfg_file = f"weights/{model_id}/config.gin"
    model = get_model(config_file=cfg_file)

    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_state_dict[k] = v
        elif k.startswith("embedding_layer."):
            new_state_dict[k] = v
        else:
            continue

    print("Loaded state dict with", len(new_state_dict), "keys.")

    weights_out = f"weights_light/{model_id}/model.ckpt"
    weights_out = Path(weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, weights_out)

    copyfile(cfg_file, weights_out.parent / "config.gin")
    print("ok!")
