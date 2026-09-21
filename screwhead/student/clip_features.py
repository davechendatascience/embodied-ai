"""Frozen CLIP ViT-B/32: pooled image features (the first students' visual input) and
the instruction embedding every student reads (512-d text projection)."""
from __future__ import annotations

import numpy as np
import torch


def clip_encoder(device):
    from transformers import CLIPModel, CLIPTokenizer
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def images(*frames):
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(device) / 255.0
        x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        with torch.no_grad():
            return m.vision_model(pixel_values=(x - mean) / std).pooler_output

    def text(s):
        ids = tok([s], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            return m.text_projection(m.text_model(**ids).pooler_output)[0]
    return images, text
