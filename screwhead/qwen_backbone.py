"""Qwen2-VL as a jointly-grounded scene+language encoder, LoRA-tuned.

Why a VLM rather than a stronger vision encoder: with CLIP the image and the
instruction are embedded SEPARATELY and concatenated, so the model never sees
them together. "the black bowl BETWEEN THE PLATE AND THE RAMEKIN" cannot be
resolved that way even in principle -- it requires the text to attend over image
regions. Every task in this suite is phrased like that. A VLM does it natively.

Only the adapters train. The 2.2B backbone stays frozen in bf16, so a checkpoint
is a few MB of LoRA weights plus the action head rather than 4.4 GB, and the
optimiser state stays small enough to matter on one device.

What is deliberately NOT changed: the state encoding (tool pose, embodiment
free), the action channels (standardised, because the gripper otherwise carries
99.8% of the loss), and the twist/joint-delta split between the two heads. Those
were each established by measurement earlier and are not the variable under test
here -- the backbone is.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
HIDDEN = 1536


class QwenBackbone(nn.Module):
    """Two camera frames plus an instruction -> one feature vector.

    The feature is the last position's hidden state. Qwen2-VL is causal, so that
    position has attended over both images and the whole instruction; it is the
    one place a single vector summarises the joint grounding.
    """

    def __init__(self, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, train_vision: bool = False,
                 device: str = "cuda"):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=device)
        base.config.use_cache = False

        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if train_vision:
            # The vision tower's projections are named the same; restricting to
            # the language model is the cheaper default because the tower is
            # already a good encoder and the grounding is what needs adapting.
            targets += ["gate_proj", "up_proj", "down_proj"]
        self.model = get_peft_model(base, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=targets))
        self.device = device

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def encode(self, agent: np.ndarray, wrist: np.ndarray,
               instructions: list[str]) -> Tensor:
        """(B,H,W,3) uint8 x2 + B instructions -> (B, HIDDEN) bf16.

        Both cameras go in as separate images in one conversation, so the model
        can relate the scene view to the wrist view rather than seeing them as
        unrelated inputs.
        """
        from PIL import Image
        texts, images = [], []
        for i, instr in enumerate(instructions):
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"}, {"type": "text", "text": instr}]}]
            texts.append(self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            images += [Image.fromarray(agent[i]), Image.fromarray(wrist[i])]
        batch = self.processor(text=texts, images=images,
                               padding=True, return_tensors="pt").to(self.device)
        out = self.model(**batch, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[-1]
        # Last NON-PAD position per sequence. The processor left-pads a batch, so
        # taking h[:, -1] unconditionally would read a pad token for every
        # sequence shorter than the longest one.
        mask = batch["attention_mask"]
        idx = mask.sum(dim=1) - 1
        return h[torch.arange(h.shape[0], device=h.device), idx]

    def save_adapter(self, path) -> None:
        """LoRA weights only -- a few MB, not the 4.4 GB backbone."""
        self.model.save_pretrained(str(path))
