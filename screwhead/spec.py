"""The robot specification, as tokens a policy can read.

One token per joint, variable length. This replaces GR00T's integer
`embodiment_id` into 32 slabs of per-robot weights and pi0's zero padding into a
fixed 32-vector: both are learned encodings that need data for every new arm,
and neither reads the kinematics that fully determine the decode.

Scale is handled explicitly. A screw axis (w, v) has a unit angular part but a
linear part with units of length -- v = -w x q, so it grows with the moment arm.
Feeding raw v means a 1.5 m arm and a 0.4 m arm with identical geometry produce
different tokens, and the policy would have to learn the conversion. The linear
part is normalised by the arm's reach and the reach is supplied separately, so
the tokens carry shape and the global carries size.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .kinematics import fk
from .poe import Chain

TOKEN_DIM = 10          # 6 screw + 1 type + 2 limits + 1 normalised index
GLOBAL_DIM = 8          # reach + dof + redundancy + log(M) as a 6-vector? see below


@dataclass
class Spec:
    tokens: Tensor      # (n, TOKEN_DIM)
    globals_: Tensor    # (GLOBAL_DIM,)
    n: int

    def padded(self, max_n: int) -> tuple[Tensor, Tensor]:
        """Pad to a fixed width and return the mask.

        Masked, not zero-filled. A 6-DoF arm in a 7-wide batch must be absent
        from attention, not present as a zero token -- a zero screw axis is a
        meaningful and wrong value, and the model would learn from it.
        """
        pad = torch.zeros(max_n - self.n, TOKEN_DIM, dtype=self.tokens.dtype)
        mask = torch.zeros(max_n, dtype=torch.bool)
        mask[: self.n] = True
        return torch.cat([self.tokens, pad]), mask


def reach(chain: Chain, samples: int = 512, seed: int = 0) -> float:
    """Radius of the tool's reachable cloud -- the arm's characteristic length."""
    g = torch.Generator().manual_seed(seed)
    p = fk(chain, chain.sample(samples, g))[:, :3, 3]
    return float(torch.linalg.norm(p - p.mean(0), dim=-1).max())


def encode(chain: Chain, scale: float | None = None) -> Spec:
    """Chain -> tokens. Body-form axes, so an embodiment swap leaves the
    policy's output frame untouched (ch.4 sec.4.3)."""
    L = reach(chain) if scale is None else scale
    B = chain.B.clone()
    B[:, 3:] = B[:, 3:] / max(L, 1e-9)                  # linear part is a length
    kind = torch.tensor([[0.0 if t == "revolute" else 1.0] for t in chain.joint_types],
                        dtype=B.dtype)
    lim = chain.limits.clone().to(B.dtype) / torch.pi   # radians -> O(1)
    idx = (torch.arange(chain.n, dtype=B.dtype) / max(chain.n - 1, 1))[:, None]
    tokens = torch.cat([B, kind, lim, idx], dim=-1)

    M = chain.M
    globals_ = torch.cat([
        torch.tensor([L, chain.n / 10.0, float(chain.n > 6)], dtype=B.dtype),
        M[:3, 3] / max(L, 1e-9),
        torch.tensor([float(torch.linalg.det(M[:3, :3])), 0.0], dtype=B.dtype),
    ])
    return Spec(tokens=tokens, globals_=globals_, n=chain.n)


def perturb(chain: Chain, magnitude: float, seed: int) -> Chain:
    """A geometrically valid variation of an arm: the randomisation the design
    depends on, and the probe that shows whether the tokens are read at all.

    Link offsets move, so screw-axis moment arms change; axis directions tilt.
    Both keep the chain a legal open chain, so the decode stays exact for the
    perturbed arm -- which is what makes it a fair test.
    """
    g = torch.Generator().manual_seed(seed)
    S = chain.S.clone()
    w = S[:, :3]
    tilt = torch.randn(chain.n, 3, generator=g) * magnitude
    w_new = w + tilt
    w_new = w_new / torch.clamp(torch.linalg.norm(w_new, dim=-1, keepdim=True), min=1e-9)
    # Recover a point on each axis, jitter it, and rebuild v = -w x q.
    q = torch.linalg.cross(w, S[:, 3:])
    q = q + torch.randn(chain.n, 3, generator=g) * magnitude
    revolute = torch.tensor([t == "revolute" for t in chain.joint_types])
    S_new = S.clone()
    S_new[revolute, :3] = w_new[revolute]
    S_new[revolute, 3:] = -torch.linalg.cross(w_new, q)[revolute]
    S_new[~revolute, 3:] = w_new[~revolute]

    M = chain.M.clone()
    M[:3, 3] = M[:3, 3] + torch.randn(3, generator=g) * magnitude
    return Chain(
        name=f"{chain.name}-perturbed", joint_names=list(chain.joint_names),
        joint_types=list(chain.joint_types), S=S_new, M=M,
        limits=chain.limits.clone(), base_frame=chain.base_frame,
        tool_frame=chain.tool_frame, source=f"{chain.source}#perturb{magnitude}",
    )
