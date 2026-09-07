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

    def with_tool(self, offset: Tensor, name: str | None = None) -> "Chain":
        """Move the tool frame by a fixed transform: M' = M @ offset.

        The screw axes are untouched -- they live in the reference frame and do
        not know where the tool is. Only M moves.

        This matters more than it looks. LIBERO records ee_pos at the gripper's
        grip_site, which is 97 mm beyond the right_hand body, so FK measured to
        right_hand disagrees with every recorded pose by that offset while
        looking entirely plausible (FM-tool-frame-drift).

        Note what is NOT needed here: the arm's base placement. A body twist
        log(T^-1 T') is invariant to left multiplication, so where the robot
        stands cancels out of every retargeted twist. Only the tool offset,
        which multiplies on the right, changes the answer.
        """
        return Chain(
            name=name or f"{self.name}+tool", joint_names=list(self.joint_names),
            joint_types=list(self.joint_types), S=self.S.clone(),
            M=self.M @ offset.to(self.M.dtype), limits=self.limits.clone(),
            base_frame=self.base_frame, tool_frame=f"{self.tool_frame}+offset",
            source=self.source,
        )

    def sample(self, k: int, generator: torch.Generator | None = None) -> Tensor:
        """Uniform configurations inside the declared limits."""
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        u = torch.rand(k, self.n, dtype=self.S.dtype, generator=generator)
        return lo + u * (hi - lo)
