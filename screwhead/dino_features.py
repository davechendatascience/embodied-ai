"""Frozen DINOv2 patch tokens: the VLA's visual input.

Chosen by measurement (tools/feature_bakeoff.py). A probe from each candidate's
features to the privileged grasp position, near the grasp, in the states the VLA
itself drives into (median error):

  CLIP ViT-B/32 pooled @128 (previous input)   16.8 mm
  CLIP patches @224                            16.7 mm
  SigLIP so400m patches @224                   14.0 mm
  DINOv2-base patches @128 / @224              10.6-13.0 / 13.0-14.3 mm

and error kept falling with more VLA-driven data (8 -> 60 episodes: 19.3 -> 12.4 mm),
so the limit is coverage more than the features. Resolution did not matter, so the
policy's own 128 px renders are used, upsampled to the encoder's 224.

Output per frame: an 8x8 grid of 768-d tokens (DINOv2's 16x16 average-pooled), float16.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

MODEL_ID = "facebook/dinov2-base"
GRID = 8
DIM = 768
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class DinoFeatures:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel
        self.device = device
        self.model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, frames) -> torch.Tensor:
        """frames: (N, H, W, 3) uint8 array or list of arrays -> (N, 64, 768) float16 on device."""
        x = torch.from_numpy(np.ascontiguousarray(np.stack(frames) if isinstance(frames, (list, tuple)) else frames))
        x = x.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).half()
        h = self.model(pixel_values=x).last_hidden_state[:, 1:]                  # drop CLS: 16x16 patches
        side = int(round(h.shape[1] ** 0.5))
        g = h.float().transpose(1, 2).reshape(len(h), DIM, side, side)
        g = nn.functional.adaptive_avg_pool2d(g, GRID)
        return g.flatten(2).transpose(1, 2).half()

    def encode_array(self, frames: np.ndarray, batch: int = 128) -> np.ndarray:
        out = np.empty((len(frames), GRID * GRID, DIM), np.float16)
        for k in range(0, len(frames), batch):
            out[k:k + batch] = self(frames[k:k + batch]).cpu().numpy()
        return out
