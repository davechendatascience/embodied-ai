"""Meta's DINOv3 backbone API, backed by the HuggingFace weights.

PointWorld loads DINOv3 through `torch.hub` from Meta's repo and expects the
original `.pth` state_dict. The publicly downloadable weights on HuggingFace
(`facebook/dinov3-vitl16-pretrain-lvd1689m`) are the same model in Transformers
packaging, with different parameter names.

Rather than remap 415 parameter names -- which would load cleanly and could
silently produce subtly wrong features -- this exposes the ONE method
`scene_featurizer.py:185` actually calls, `get_intermediate_layers`, over the HF
model. The forward pass is then Meta's own reference implementation as ported by
HF, and there is no key mapping of mine to get wrong.

TWO DETAILS THAT MATTER, both read from
`third_party/dinov3/dinov3/models/vision_transformer.py` rather than assumed:

1. INDEXING. `_get_intermediate_layers_not_chunked` appends `x` AFTER running
   block `i` (`x = blk(...)` then `if i in blocks_to_take`). HF's
   `hidden_states` is `[embeddings, after_block_0, ..., after_block_23]`, so
   "after block i" is `hidden_states[i + 1]`. Off by one here silently shifts
   every tap by a layer.

2. NORMALISATION. `get_intermediate_layers` defaults to `norm=True` and applies
   the model's FINAL LayerNorm to each intermediate output. HF's
   `hidden_states` are raw block outputs. `feat_proj` was trained on normed
   features, so skipping this feeds it a different distribution -- the exact
   class of bug that produces plausible, wrong numbers.

Register/CLS tokens are deliberately NOT stripped here: the caller already does
that (`patch_tokens[:, -n_patches:, :]`), and doing it twice would silently drop
real patches.
"""

import torch
import torch.nn as nn

MODEL_DIR = "third_party/dinov3_hf"


class HFDinov3Backbone(nn.Module):
    """Adapter presenting the HF DINOv3 through Meta's backbone interface."""

    def __init__(self, model_dir=MODEL_DIR, dtype=None):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_dir)
        if dtype is not None:
            self.model = self.model.to(dtype)
        self.model.eval()
        cfg = self.model.config
        self.embed_dim = cfg.hidden_size
        self.n_blocks = cfg.num_hidden_layers
        self.patch_size = cfg.patch_size
        # 1 CLS + N register tokens precede the patch tokens
        self.n_prefix = 1 + getattr(cfg, "num_register_tokens", 0)
        self._norm = self._resolve_final_norm()

    def _resolve_final_norm(self):
        """The final LayerNorm, whatever HF happens to call it."""
        for attr in ("layernorm", "norm", "final_layernorm", "ln_post"):
            mod = getattr(self.model, attr, None)
            if isinstance(mod, nn.Module):
                return mod
        # Fall back to identity rather than guessing wrong, but say so loudly:
        # silently skipping the norm is precisely the failure this module exists
        # to avoid.
        raise AttributeError(
            "could not find DINOv3's final LayerNorm on the HF model; "
            "get_intermediate_layers(norm=True) cannot be reproduced faithfully")

    @torch.no_grad()
    def get_intermediate_layers(self, x, *, n=1, reshape=False,
                                return_class_token=False, return_extra_tokens=False,
                                norm=True, **_):
        hidden = self.model(pixel_values=x, output_hidden_states=True).hidden_states
        n_blocks = len(hidden) - 1                      # hidden[0] is the embedding
        blocks = list(n) if not isinstance(n, int) else range(n_blocks - n, n_blocks)
        outs = [hidden[i + 1] for i in blocks]          # see NOTE 1
        if norm:
            outs = [self._norm(o) for o in outs]        # see NOTE 2

        if reshape:
            h = w = int(round((outs[0].shape[1] - self.n_prefix) ** 0.5))
            outs = [o[:, self.n_prefix:].reshape(o.shape[0], h, w, -1).permute(0, 3, 1, 2)
                    for o in outs]
        if return_class_token:
            return tuple(outs), tuple(o[:, 0] for o in outs)
        return tuple(outs)


def install(featurizer, model_dir=MODEL_DIR, dtype=None):
    """Replace a `SceneFeaturizer`'s backbone with the HF-backed adapter.

    Used instead of letting `_load_backbone` run, so PointWorld never needs
    Meta's `.pth`. Everything downstream -- the projection of 3D points into
    camera views, the multi-layer concatenation, `feat_proj` -- is unchanged
    upstream code.
    """
    featurizer.dinov3 = HFDinov3Backbone(model_dir, dtype=dtype)
    for p in featurizer.dinov3.parameters():
        p.requires_grad_(False)
    return featurizer
