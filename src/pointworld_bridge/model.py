"""Build the released PointWorld `BaseModel` for inference, from its checkpoint.

Why `BaseModel` and not `DynamicsPredictor`
-------------------------------------------
`DynamicsPredictor` is the PTv3 trunk plus two heads. It is NOT the model.
Everything that makes its inputs and outputs mean anything lives one level up,
in `BaseModel.forward` (`pointworld/base.py:434`):

  * `normalize_fn` / `unnormalize_fn`, and the final
    `pred = self.unnormalize(pred_norm)` at base.py:484. Without these the
    trunk is fed raw metres and its normalized output is read as metres.
  * `robot_proj` + `time_embed` + `robot_type_emb`, which turn the 16 raw robot
    channels into the trunk's action encoding. The robot point flow IS the
    action, so a zeroed robot feature is a zeroed action.
  * `scene_feature_encoder`, which runs DINOv3, projects it to the points, and
    fuses it with the 31 raw scene channels through `scene_encoder_norm`,
    `scene_raw_feat_proj` and `scene_proj`.

Hand-assembling those pieces is how the first harness reported 553 mm, then
21.86 mm, on a 0.8 m scene. Driving `BaseModel` removes the whole class of
error rather than one instance of it.

DINOv3 comes out of `model-best.pt` itself -- see
`scripts/extract_dinov3_from_checkpoint.py`. Nothing is downloaded.
"""

import copy
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PW = ROOT / "third_party" / "PointWorld"
# Selectable, because the three released checkpoints differ in ways that reach
# all the way into feature assembly (16/31 channels single-arm vs 17/42
# bimanual) and normalisation (one domain vs two).
CKPT_NAME = os.environ.get("POINTWORLD_CKPT", "large-droid+behavior")
CKPT = PW / "pretrained_checkpoints" / CKPT_NAME / "model-best.pt"
DINOV3 = PW / "third_party" / "dinov3" / "checkpoints" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def add_pointworld_to_path():
    """PointWorld is a clone, not an install, and imports its own top level."""
    for p in (str(PW), str(PW / "third_party" / "dinov3")):
        if p not in sys.path:
            sys.path.insert(0, p)


def inference_args(ck, device="cuda"):
    """The checkpoint's own training args, adjusted for single-process inference.

    Only the things that are untrue outside the training job are touched; every
    architectural field is left exactly as trained.
    """
    args = copy.deepcopy(ck["args"])
    args.distributed = False          # no process group exists here
    args.device = device
    args.disable_compile = True       # torch.compile on a one-shot run is pure overhead
    args.rank = 0
    # 'stats/droid' is relative to the PointWorld checkout, not to our cwd.
    if not Path(args.norm_stats_path).is_absolute():
        args.norm_stats_path = str(PW / args.norm_stats_path)
    return args


def data_info_from_checkpoint(ck):
    """Raw feature widths, read off the projections rather than assumed.

    Same rule the upstream trainer uses in inference-only mode
    (`training/trainer.py:101`). For this checkpoint it gives 31 scene channels
    and 16 robot channels.
    """
    state = ck["model"]
    return {
        "scene_features_dim": int(state["scene_feature_encoder.scene_raw_feat_proj.weight"].shape[1]),
        "robot_features_dim": int(state["robot_proj.fc1.weight"].shape[1]),
    }


def load_base_model(device="cuda", ckpt=CKPT, verbose=True):
    """Returns (model, args, ck). The model is on `device`, in eval mode."""
    add_pointworld_to_path()
    if not DINOV3.exists():
        raise FileNotFoundError(
            f"DINOv3 weights missing at {DINOV3}. Run "
            "scripts/extract_dinov3_from_checkpoint.py -- they are inside "
            "model-best.pt and need no download."
        )

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = inference_args(ck, device)
    info = data_info_from_checkpoint(ck)

    from pointworld.base import BaseModel

    # The feature layout the checkpoint expects, carried on args so
    # `build_data_dict` assembles the matching number of channels. 17 robot /
    # 42 scene means BIMANUAL: gripper state occupies two slots, not one.
    args._bimanual = info["robot_features_dim"] == 17
    model = BaseModel(args, info, rank=0).to(device)
    # strict: the norm-stat buffers, DINOv3, and every projection must all come
    # from the checkpoint. A missing key here is a silent change of experiment.
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    if verbose:
        n = sum(v.numel() for v in ck["model"].values())
        print(f"model   : BaseModel, {len(ck['model'])} tensors / {n / 1e6:.0f}M params, "
              f"strict load OK")
        print(f"          scene_features {info['scene_features_dim']}ch, "
              f"robot_features {info['robot_features_dim']}ch, "
              f"domains {args.domains}, T={model.T}")
    return model, args, ck
