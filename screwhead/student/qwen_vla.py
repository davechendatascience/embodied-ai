"""The Panda VLA on Qwen3-VL-2B (docs/design_VLA_qwen.md, BRN-vla-sees-and-acts-as-trained).

The backbone reads both cameras and the instruction; after them come one proprioception token and H action
query tokens. Their embeddings are ours -- a projection of the proprioceptive vector and H learned vectors --
written over two embedding rows the tokenizer never produces, so the backbone's own forward pass places the
images and their positions. The action queries' final hidden states give a chunk of H actions: a normalized
twist (L1-regressed) and a gripper logit whose sign is the open/close command. The vision tower is frozen and
the language model adapted by LoRA.

Everything the model's inputs and outputs are made or read with -- the normalization constants fitted on its
training labels, H and the k actions executed per query, the image size -- is stored with its weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from .libero_data import PROPRIO_DIM

BACKBONE = "Qwen/Qwen3-VL-2B-Instruct"
# The VLA's execution (BRN-vla-decodes-twists-exactly), one definition for its evaluation and its replay gate: the
# skill teacher's servo and controller, the free-motion acceleration bound kept while holding, the IK iterated to
# 1e-6 mm / 1e-6 rad in at most 30 iterations, the null space pulled to the start posture, the simulator's own
# gripper command, the read after each step's physics. Keyword arguments of screwhead.sim.sim_arm.Execution.
VLA_EXECUTION = dict(gripper_mode="command", max_lin_acc_holding=None, servo_iters=30, servo_tol=(1e-9, 1e-6),
                     posture_start=True, observe_end=True)
PROPRIO_TOKEN = 151900     # rows past len(tokenizer) (151669): never produced by the tokenizer
ACTION_TOKEN = 151901
LORA_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


@dataclass
class VLAConfig:
    chunk: int = 8                     # H: actions predicted per query
    execute: int = 4                   # k: actions executed before the model is queried again
    lora_rank: int = 32
    lora_alpha: int = 64
    blind: bool = False                # the blind twin: camera images replaced by zeros, in training and evaluation
    image_px: int = 128                # as LIBERO stores them
    twist_mean: list = field(default_factory=lambda: [0.0] * 6)   # fitted on the training labels
    twist_std: list = field(default_factory=lambda: [1.0] * 6)
    proprio_mean: list = field(default_factory=lambda: [0.0] * PROPRIO_DIM)
    proprio_std: list = field(default_factory=lambda: [1.0] * PROPRIO_DIM)


class QwenVLA(nn.Module):
    def __init__(self, cfg: VLAConfig, backbone: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        d = backbone.config.text_config.hidden_size
        self.proprio_in = nn.Sequential(nn.Linear(PROPRIO_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.queries = nn.Parameter(torch.randn(cfg.chunk, d) * 0.02)
        self.head = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, 7))
        self._proprio: torch.Tensor | None = None
        backbone.get_input_embeddings().register_forward_hook(self._write_embeddings)

    def _write_embeddings(self, _module, inputs, out):
        """The proprioception and action-query rows of the embedded sequence, replaced by ours."""
        ids = inputs[0]
        out = out.clone()
        p = ids == PROPRIO_TOKEN
        if p.any():
            out[p] = self.proprio_in(self._proprio.float()).to(out.dtype)
        a = ids == ACTION_TOKEN
        if a.any():
            out[a] = self.queries.to(out.dtype).repeat(int(a.sum()) // self.cfg.chunk, 1)
        return out

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, H, 6) normalized twists and (B, H) gripper logits. `batch`: input_ids, attention_mask and
        mm_token_type_ids (left-padded, ending in the proprioception token and H action queries),
        pixel_values, image_grid_thw, and proprio already normalized."""
        self._proprio = batch["proprio"]
        out = self.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            mm_token_type_ids=batch["mm_token_type_ids"],
                            pixel_values=batch["pixel_values"], image_grid_thw=batch["image_grid_thw"],
                            output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1][:, -self.cfg.chunk:]
        y = self.head(h.float())
        return y[..., :6], y[..., 6]

    # -- the weights and everything that goes with them -------------------------------------
    def trainable_state(self) -> dict:
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: t.detach().cpu() for n, t in self.state_dict().items() if n in keep}

    def save(self, path: str) -> None:
        torch.save({"config": asdict(self.cfg), "backbone": BACKBONE, "weights": self.trainable_state()}, path)


def build(cfg: VLAConfig, device: str = "cuda") -> tuple[QwenVLA, object]:
    """The model (backbone frozen but for LoRA on the language model) and the backbone's processor."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(BACKBONE)
    base = AutoModelForImageTextToText.from_pretrained(BACKBONE, dtype=torch.bfloat16)
    for p in base.parameters():
        p.requires_grad_(False)
    base = get_peft_model(base, LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                                           target_modules=LORA_TARGETS))
    model = QwenVLA(cfg, base).to(device)
    model.proprio_in.float(); model.head.float()
    return model, proc


def load(path: str, device: str = "cuda") -> tuple[QwenVLA, object]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, proc = build(VLAConfig(**ck["config"]), device)
    missing = set(ck["weights"]) - set(model.state_dict())
    if missing:
        raise KeyError(f"{path}: weights the model does not have: {sorted(missing)[:3]}")
    model.load_state_dict(ck["weights"], strict=False)
    return model.eval(), proc


def prompt_ids(proc, instruction: str, images: list[np.ndarray], cfg: VLAConfig) -> dict:
    """One sample's tokens and pixels: the two upright camera images and the instruction, then the
    proprioception token and H action queries. Blind: the images are zeros of the same size."""
    if cfg.blind:
        images = [np.zeros_like(im) for im in images]
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": instruction}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=False)
    enc = proc(text=[text], images=list(images), return_tensors="pt")
    extra = torch.tensor([PROPRIO_TOKEN] + [ACTION_TOKEN] * cfg.chunk)
    ids = torch.cat([enc["input_ids"][0], extra])
    kinds = torch.cat([enc["mm_token_type_ids"][0], torch.zeros_like(extra)])     # the appended tokens are text
    return {"input_ids": ids, "mm_token_type_ids": kinds, "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"]}


def collate(samples: list[dict], pad_id: int) -> dict:
    """Left-padded: every row ends in its action queries."""
    n = max(len(s["input_ids"]) for s in samples)
    ids = torch.full((len(samples), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(samples), n), dtype=torch.long)
    kinds = torch.zeros((len(samples), n), dtype=torch.long)
    for i, s in enumerate(samples):
        k = len(s["input_ids"])
        ids[i, n - k:] = s["input_ids"]
        mask[i, n - k:] = 1
        kinds[i, n - k:] = s["mm_token_type_ids"]
    out = {"input_ids": ids, "attention_mask": mask, "mm_token_type_ids": kinds,
           "pixel_values": torch.cat([s["pixel_values"] for s in samples]),
           "image_grid_thw": torch.cat([s["image_grid_thw"] for s in samples])}
    for k in ("proprio", "twist", "gripper", "valid"):
        if k in samples[0]:
            out[k] = torch.stack([torch.as_tensor(s[k]) for s in samples])
    return out
