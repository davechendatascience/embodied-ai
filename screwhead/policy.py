"""Two action heads over the same frozen features.

They differ in exactly one thing -- what they predict -- so a difference in
transfer is attributable to the action representation and not to perception,
capacity, data, or optimisation.

  BaselineHead    delta-q in joint space, padded to MAX_DOF. This is how pi0
                  and GR00T handle embodiment: a fixed-width vector, sliced per
                  robot, with the kinematics never reaching the model. On an
                  arm it was not trained on there is nothing to do but slice,
                  which is the charitable reading of zero-shot transfer for
                  those systems.

  ScrewHead       a body twist in the tool frame plus grasp intent, decoded by
                  the arm's own Jacobian. The learned output carries no joint
                  count, and the spec tokens tell it which arm it is driving.

Both emit an action CHUNK. Chunking is near-universal in this literature and
buys temporal consistency; it also means the twist decoder is exercised over a
horizon, where null-space drift would show up.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .spec import TOKEN_DIM
from .state import STATE_DIM

FEAT_DIM = 768          # CLIP ViT-B/32 vision pooler
TEXT_DIM = 512          # CLIP text projection
MAX_DOF = 7


class Trunk(nn.Module):
    """Shared perception-to-latent stack. Identical in both policies."""

    def __init__(self, width: int = 512, state_dim: int = STATE_DIM, depth: int = 2):
        super().__init__()
        self.img = nn.Linear(2 * FEAT_DIM, width)
        self.txt = nn.Linear(TEXT_DIM, width)
        self.state = nn.Linear(state_dim, width)
        layers = []
        for _ in range(depth):
            layers += [nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                       nn.Linear(width * 2, width)]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(width)

    def forward(self, agent: Tensor, wrist: Tensor, text: Tensor, state: Tensor) -> Tensor:
        h = self.img(torch.cat([agent, wrist], -1)) + self.txt(text) + self.state(state)
        return self.norm(h + self.mlp(h))


class BaselineHead(nn.Module):
    """Padded joint-space delta, conditioned on a learned embodiment EMBEDDING.

    Deliberately not a weaker straw man. It carries the same trunk, the same
    attention block and within 2% of the same parameter count as ScrewHead, and
    it conditions on embodiment exactly the way GR00T does -- an integer id
    selecting learned weights (CategorySpecificMLP over max_num_embodiments=32).
    The single difference is WHAT the conditioning carries: an opaque id here, a
    screw axis per joint there. That is the variable under test.

    An id has no value for an arm that was never trained, which is the real
    limitation of the approach rather than an artifact of this implementation.
    """

    def __init__(self, width: int = 512, chunk: int = 8, heads: int = 4,
                 num_embodiments: int = 32):
        super().__init__()
        self.chunk = chunk
        self.trunk = Trunk(width)
        self.embodiment = nn.Embedding(num_embodiments, width)
        self.emb_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.emb_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, chunk * (MAX_DOF + 1))

    def forward(self, agent, wrist, text, state, embodiment_id) -> Tensor:
        h = self.trunk(agent, wrist, text, state)
        tok = self.embodiment(embodiment_id)[:, None]
        attended, _ = self.emb_attn(h[:, None], tok, tok, need_weights=False)
        h = self.emb_norm(h + attended[:, 0])
        return self.out(h).view(-1, self.chunk, MAX_DOF + 1)


class ScrewHead(nn.Module):
    """Body twist + grasp intent, conditioned on variable-length spec tokens.

    The spec is read by cross-attention over one token per joint, with a mask
    rather than zero padding: a zero screw axis is a meaningful and wrong value,
    and an unmasked pad would be learned from.
    """

    def __init__(self, width: int = 512, chunk: int = 8, heads: int = 4):
        super().__init__()
        self.chunk = chunk
        self.trunk = Trunk(width)
        self.spec_in = nn.Linear(TOKEN_DIM, width)
        self.spec_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.spec_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, chunk * 7)          # 6 twist + 1 grasp

    def forward(self, agent, wrist, text, state, spec_tokens, spec_mask) -> Tensor:
        h = self.trunk(agent, wrist, text, state)
        tok = self.spec_in(spec_tokens)
        attended, _ = self.spec_attn(h[:, None], tok, tok,
                                     key_padding_mask=~spec_mask, need_weights=False)
        h = self.spec_norm(h + attended[:, 0])
        return self.out(h).view(-1, self.chunk, 7)
