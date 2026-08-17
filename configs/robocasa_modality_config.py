"""Register the robocasa_panda_omron modality config, read FROM the checkpoint.

`launch_finetune.py --modality-config-path` takes a PYTHON file that registers a
config at import time. One is needed here because Isaac-GR00T n1.6.1-release
does not ship this embodiment in any of its config tables -- the same gap shows
up three times:

    gr00t.data.stats.MODALITY_CONFIGS            -> relative stats fail
    gr00t.configs.data.embodiment_configs        -> server falls back to model
    FinetuneConfig.data.modality_configs         -> training dies in validate()

all of which list libero_panda, gr1, unitree_g1, oxe_* and behavior_r1_pro, and
none of which list robocasa_panda_omron. The pretrained GR00T-N1.6-3B checkpoint
was nevertheless trained on it and evaluates zero-shot at 66.22% -- the config
just lives in the model rather than the codebase.

SO IT IS READ FROM THE MODEL, NOT WRITTEN BY HAND.
`processor_config.json` inside the checkpoint carries the exact spec: modality
keys, delta_indices (action horizon 16), and one ActionConfig per action head.
Retyping that by hand would be a guess dressed as a config, and a wrong
`delta_indices` or a flipped RELATIVE/ABSOLUTE trains cleanly and evaluates as
noise -- the failure mode this whole pipeline has been built to avoid. Reading
it means the fine-tune uses the same contract the checkpoint was pretrained
with, by construction, and keeps doing so if the checkpoint is updated.

Usage:
  bash examples/finetune.sh ... --modality-config-path <this file>
"""
import glob
import json
import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (ActionConfig, ActionFormat, ActionRepresentation,
                              ActionType, ModalityConfig)

TAG = "robocasa_panda_omron"
#: Overridable so a local checkpoint directory can be used instead of the cache.
CKPT = os.environ.get("GROOT_CKPT_DIR") or next(iter(sorted(glob.glob(
    os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/"
                       "snapshots/*")))), "")


def _action_config(d):
    return ActionConfig(
        rep=ActionRepresentation[d["rep"]],
        type=ActionType[d["type"]],
        format=ActionFormat[d["format"]],
        **({"state_key": d["state_key"]} if d.get("state_key") else {}),
    )


def _modality_config(d):
    acts = d.get("action_configs")
    return ModalityConfig(
        delta_indices=list(d["delta_indices"]),
        modality_keys=list(d["modality_keys"]),
        **({"action_configs": [_action_config(a) for a in acts]} if acts
           else {}),
    )


def load():
    """Checkpoint's processor_config.json -> {modality: ModalityConfig}."""
    path = os.path.join(CKPT, "processor_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no processor_config.json at {path!r}. Set GROOT_CKPT_DIR to the "
            "GR00T-N1.6-3B checkpoint directory.")
    with open(path) as f:
        cfgs = json.load(f)["processor_kwargs"]["modality_configs"]
    if TAG not in cfgs:
        raise KeyError(f"{TAG} not in the checkpoint's modality_configs; "
                       f"have {sorted(cfgs)}")
    return {k: _modality_config(v) for k, v in cfgs[TAG].items()}


CONFIG = load()
register_modality_config(CONFIG, embodiment_tag=EmbodimentTag.ROBOCASA_PANDA_OMRON)
print(f"registered {TAG} from {CKPT}: "
      + ", ".join(f"{k}({len(v.modality_keys)} keys)" for k, v in CONFIG.items()))
