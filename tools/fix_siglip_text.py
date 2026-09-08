#!/usr/bin/env python
"""Recompute the pooled text embedding in an existing SigLIP cache.

The cache was written by a version that called get_text_features(...)[0], which
returns BaseModelOutputWithPooling -- so [0] selected last_hidden_state and
stored (1, 64, 1152) token vectors where one pooled sentence embedding belongs.
The image tokens are unaffected, and text is ten forward passes, so this repairs
the metadata rather than rebuilding 26 GB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screwhead.siglip_backbone import MODEL_ID  # noqa: E402


def main() -> int:
    cache = Path(sys.argv[1] if len(sys.argv) > 1 else "cache/libero_spatial_siglip")
    from transformers import AutoTokenizer, SiglipModel
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    m = SiglipModel.from_pretrained(MODEL_ID, dtype=torch.float16).cuda().eval()
    for meta in sorted(cache.glob("task*_meta.npz")):
        # Materialise and CLOSE before writing: savez to a path still open for
        # reading is asking for a truncated file, and this metadata is the only
        # copy of the twists and states for that task.
        with np.load(meta, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        before = np.asarray(d["text"]).shape
        ids = tok([str(d["instruction"])], padding="max_length", return_tensors="pt").to("cuda")
        with torch.no_grad():
            d["text"] = m.text_model(**ids).pooler_output[0].to(torch.float16).cpu().numpy()
        # Must end in .npz: savez_compressed appends the extension when it is
        # absent, so a ".npz.tmp" name becomes ".npz.tmp.npz" and the rename
        # then targets a path that was never written.
        tmp = meta.with_name(meta.stem + ".tmp.npz")
        np.savez_compressed(tmp, **d)
        tmp.replace(meta)                      # atomic swap
        print(f"{meta.name}: {before} -> {d['text'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
