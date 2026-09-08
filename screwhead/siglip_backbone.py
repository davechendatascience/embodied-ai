"""SigLIP vision tower, run live, returning PATCH tokens rather than a pool.

The measurement this exists to answer: on pooled CLIP features a linear probe
reaches 0.58 correlation on tool angular velocity and a 4.26M head reaches 0.54.
The head extracts no more than a linear map, so the features are the ceiling --
while translation and gripper sit at 0.88-0.93. Orientation is the binding
constraint, and orientation is spatial, which is precisely what pooling to a
single vector destroys.

So the tokens are kept. 27x27 patches at dim 1152 are average-pooled to a coarse
grid: the source frames are 128x128 upsampled to 384, so 27x27 oversamples at
~4.7 native pixels per patch, and an 8x8 grid still leaves 64x more spatial
information than one vector.

Run live rather than cached: at full resolution the cache would be 203 GB, and
caching also forecloses fine-tuning, which is the next lever if patches alone do
not lift wx/wy.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

MODEL_ID = "google/siglip-so400m-patch14-384"
DIM = 1152


class SiglipBackbone(nn.Module):
    def __init__(self, grid: int = 8, lora_r: int = 0, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        from transformers import AutoImageProcessor, SiglipVisionModel
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.vis = SiglipVisionModel.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
        self.grid, self.device, self.dtype = grid, device, dtype
        self.lora = lora_r > 0
        if self.lora:
            from peft import LoraConfig, get_peft_model
            self.vis = get_peft_model(self.vis, LoraConfig(
                r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
        else:
            for p in self.vis.parameters():
                p.requires_grad_(False)

    # Normalisation constants are read from the processor rather than hardcoded,
    # and the resize runs on GPU: PIL round-tripping every frame was the single
    # largest cost in the CLIP cache.
    def _pixels(self, frames: np.ndarray) -> Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
        x = x.permute(0, 3, 1, 2).to(self.dtype) / 255.0
        size = self.proc.size["height"]
        x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
        mean = torch.tensor(self.proc.image_mean, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.proc.image_std, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def encode(self, frames: np.ndarray) -> Tensor:
        """(B,H,W,3) uint8 -> (B, grid*grid, DIM)."""
        h = self.vis(pixel_values=self._pixels(frames)).last_hidden_state
        b, n, d = h.shape
        g = int(round(n ** 0.5))
        h = h.transpose(1, 2).reshape(b, d, g, g)
        h = torch.nn.functional.adaptive_avg_pool2d(h, self.grid)
        return h.flatten(2).transpose(1, 2)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.vis.parameters() if p.requires_grad)
