"""Write PointWorld's own DINOv3 weights out of `model-best.pt`.

The checkpoint carries the full backbone under
`scene_feature_encoder.scene_encoder.dinov3.*`, in Meta's original naming, so
the exact weights used during training are already on disk. Downloading from
HuggingFace was never necessary (and the HF port needed a LayerNorm fix that
this path avoids entirely -- see NOTES.md).

`SceneEncoder2D` loads the backbone through `torch.hub.load(<dinov3 submodule>,
weights=<path>)`, and Meta's loader parses the 8-character hash out of the
FILENAME, so the name below is not cosmetic.

    CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 \
        .venv-pw/bin/python scripts/extract_dinov3_from_checkpoint.py
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PW = ROOT / "third_party" / "PointWorld"
CKPT = PW / "pretrained_checkpoints" / "small-droid" / "model-best.pt"
PREFIX = "scene_feature_encoder.scene_encoder.dinov3."
# Meta's hub loader does `re.findall(r"-(.{8}).pth", weights)` and switches
# architecture flags on the hash, so this filename must be exact.
OUT = PW / "third_party" / "dinov3" / "checkpoints" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def main():
    if OUT.exists() and "--force" not in sys.argv:
        print(f"already present: {OUT} ({OUT.stat().st_size / 2**20:.0f} MiB)")
        return 0
    if not CKPT.exists():
        print(f"checkpoint not found: {CKPT}")
        return 1

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k[len(PREFIX):]: v for k, v in ck["model"].items() if k.startswith(PREFIX)}
    if not sd:
        print(f"no keys under {PREFIX} -- wrong checkpoint?")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, OUT)
    n = sum(v.numel() for v in sd.values())
    print(f"{len(sd)} tensors, {n / 1e6:.1f}M params -> {OUT}")

    # Prove it loads strictly into Meta's own module rather than trusting the
    # key names: a silent mismatch here would surface as confident nonsense.
    sys.path.insert(0, str(PW / "third_party" / "dinov3"))
    model = torch.hub.load(str(PW / "third_party" / "dinov3"), "dinov3_vitl16",
                           source="local", weights=str(OUT), trust_repo=True)
    print(f"strict load OK: {type(model).__name__}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    return 0


if __name__ == "__main__":
    sys.exit(main())
