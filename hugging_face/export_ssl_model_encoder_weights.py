from pathlib import Path
from shutil import copyfile

import torch

from omar_rq import get_model

torch.manual_seed(2)

model_dirs = {
    "omar-rq-base-freesound-small": "omar-rq-base-freesound-small/checkpoints/"
}

for model_id, model_dir in model_dirs.items():
    print(f"Processing model {model_dir}...")

    # ure glob to get the file with .gin extention
    cfg_file = list(Path(model_dir).glob("*.gin"))[0]
    ckpt_file = list(Path(model_dir).glob("*.ckpt"))[0]

    state_dict = torch.load(ckpt_file, map_location="cpu")["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_state_dict[k] = v
        elif k.startswith("embedding_layer."):
            new_state_dict[k] = v
        else:
            continue

    print("Loaded state dict with", len(new_state_dict), "keys.")

    weights_out = f"output_models/{model_id}/model.ckpt"
    weights_out = Path(weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, weights_out)

    new_cfg_file = weights_out.parent / "config.gin"
    copyfile(cfg_file, new_cfg_file)
    print("weights saved!")

    print("Loading model to test...")
    model = get_model(config_file=new_cfg_file)
    print("OK!")
