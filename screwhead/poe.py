"""The embodiment, reduced to what forward kinematics actually needs.

A Chain is (M, {S_i}) plus joint types and limits -- ch.4 sec.4.5. That is the
entire embodiment encoding: a home pose and one screw axis per joint. It is
variable length by construction, which is the property a padded 29- or 32-dim
action vector does not have.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .se3 import adjoint, inverse


@dataclass
class Chain:
    name: str
    joint_names: list[str]
    joint_types: list[str]          # "revolute" | "prismatic"
    S: Tensor                       # (n, 6) screw axes in the reference frame, at zero
    M: Tensor                       # (4, 4) tool pose at zero, same reference frame
    limits: Tensor                  # (n, 2) lower, upper
    base_frame: str
    tool_frame: str
    source: str                     # where the numbers came from, for the trial record

    @property
    def n(self) -> int:
        return len(self.joint_names)

    @property
    def dof(self) -> int:
        return self.n

    @property
    def B(self) -> Tensor:
        """Body-form axes, seen from the tool: B = Ad(M^-1) S (ch.4 sec.4.3).

        The body form is what the action head wants -- an embodiment swap then
        changes only M and {B_i} and leaves the policy's output frame alone.
        """
        return (adjoint(inverse(self.M)) @ self.S.T).T

    def sample(self, k: int, generator: torch.Generator | None = None) -> Tensor:
        """Uniform configurations inside the declared limits."""
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        u = torch.rand(k, self.n, dtype=self.S.dtype, generator=generator)
        return lo + u * (hi - lo)
