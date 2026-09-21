"""VLA action head over visual patch tokens.

The previous head saw one pooled vector per camera; it could not locate a 3 mm
bowl wall. This one keeps an 8x8 grid per camera and lets learned action queries
attend over it, together with language, tool pose and the robot-description
tokens that condition the twist head on the arm.

  tokens: 64 agentview + 64 wrist (768 -> d, + camera and grid-position embeddings)
        + 1 language (512 -> d) + 1 tool pose (10 -> d) + robot-spec tokens (-> d)
  queries: n_query learned tokens, prepended
  encoder: L layers of pre-norm self-attention over [queries, tokens]
  output: mean of the query outputs -> MLP -> chunk x 7 (6 twist + gripper)
"""
from __future__ import annotations

import torch
from torch import nn

from .dino_features import DIM, GRID
from .spec import TOKEN_DIM
from ..geometry.state import STATE_DIM


class TokenHead(nn.Module):
    def __init__(self, d: int = 256, layers: int = 4, heads: int = 4, n_query: int = 4, chunk: int = 1,
                 state_dim: int = STATE_DIM, legacy_spec_mask: bool = False, out_dim: int = 7):
        super().__init__()
        self.chunk = chunk
        # 7 = twist (6) + regressed gripper; 6 + K = twist + logits over K gripper apertures
        self.out_dim = out_dim
        # Spec.padded() marks REAL joints True; attention's key padding mask ignores True.
        # Checkpoints trained before 2026-09-15 passed the mask through uninverted, so for a
        # 7-joint Panda every spec token was ignored. They load with legacy_spec_mask=True
        # to keep the computation they were trained with.
        self.legacy_spec_mask = legacy_spec_mask
        n = GRID * GRID
        self.img = nn.Linear(DIM, d)
        self.cam = nn.Parameter(torch.zeros(2, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.txt = nn.Linear(512, d)
        self.state = nn.Sequential(nn.Linear(state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.spec = nn.Linear(TOKEN_DIM, d)
        self.query = nn.Parameter(torch.randn(1, n_query, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, chunk * out_dim))

    def forward(self, agent, wrist, text, state, spec_tokens, spec_mask):
        """agent, wrist: (B, 64, 768); text (B, 512); state (B, state_dim); spec (B, J, TOKEN_DIM), mask (B, J) True=real joint."""
        b = agent.shape[0]
        a = self.img(agent.float()) + self.cam[0] + self.pos
        w = self.img(wrist.float()) + self.cam[1] + self.pos
        t = self.txt(text.float())[:, None]
        s = self.state(state.float())[:, None]
        sp = self.spec(spec_tokens.float())
        q = self.query.expand(b, -1, -1)
        x = torch.cat([q, a, w, t, s, sp], 1)
        pad = torch.zeros(b, x.shape[1], dtype=torch.bool, device=x.device)
        pad[:, -sp.shape[1]:] = spec_mask if self.legacy_spec_mask else ~spec_mask
        h = self.encoder(x, src_key_padding_mask=pad)
        return self.out(self.norm(h[:, : q.shape[1]].mean(1))).view(b, self.chunk, self.out_dim)
