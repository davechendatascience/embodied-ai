#!/usr/bin/env python
"""Cache SigLIP patch tokens from the existing raw-frame cache.

Reads cache/libero_spatial_img rather than the hdf5, so the frame selection is
identical to every earlier run by construction rather than by reimplementing the
same predicate a third time.

Encoding goes through screwhead.siglip_backbone, the same path a live run would
use, so a cached run and a fine-tuned run cannot silently differ in
preprocessing.

8x8 tokens at dim 1152 is 17.8 GB for the suite. The full 27x27 grid would be
203 GB, and the source frames are 128x128 upsampled to 384 -- so 27x27
oversamples at ~4.7 native pixels per patch, while 8x8 leaves ~16 and still
carries 64x more spatial information than the single pooled vector that CLIP
gave us.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screwhead.siglip_backbone import MODEL_ID, SiglipBackbone  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cache/libero_spatial_img")
    ap.add_argument("--out", default="cache/libero_spatial_siglip")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    back = SiglipBackbone(grid=args.grid, device=args.device)

    from transformers import AutoTokenizer, SiglipModel
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    full = SiglipModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(args.device).eval()

    t0 = time.time()
    for meta in sorted(src.glob("task*_meta.npz")):
        ti = int(meta.stem.split("_")[0][4:])
        d = dict(np.load(meta, allow_pickle=True))
        with torch.no_grad():
            ids = tok([str(d["instruction"])], padding="max_length",
                      return_tensors="pt").to(args.device)
            d["text"] = full.get_text_features(**ids)[0].to(torch.float16).cpu().numpy()
        d["grid"] = args.grid
        for cam in ("agent", "wrist"):
            frames = np.load(src / f"task{ti:02d}_{cam}.npy", mmap_mode="r")
            chunks = []
            with torch.no_grad():
                for i in range(0, len(frames), args.batch):
                    chunks.append(back.encode(np.asarray(frames[i:i + args.batch]))
                                  .to(torch.float16).cpu().numpy())
            np.save(out / f"task{ti:02d}_{cam}.npy", np.concatenate(chunks))
            del frames, chunks
        np.savez_compressed(out / f"task{ti:02d}_meta.npz", **d)
        n = len(np.load(out / f"task{ti:02d}_agent.npy", mmap_mode="r"))
        print(f"  [{ti}] {n:6d} frames  ({time.time()-t0:.0f}s)", flush=True)
    print("done ->", out, f"({shutil.disk_usage('.').free/1e9:.0f} GB free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
