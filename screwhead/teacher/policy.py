"""pi_theta: the optimization teacher's network, and the features it reads (BRN-teacher-policy).

The search of BRN-optimization-teacher starts each control step from a distribution centred on
pi_theta's plan. This module holds that network, the feature vector it reads, and the callable the
search takes as `policy=`. Weights are frozen whenever pi_theta labels or is searched around:
Policy runs under no_grad with the module in eval mode, and the checkpoint carries the feature
layout and the search settings the labels were produced under, so a plan is never decoded against
a layout other than the one it was fitted on.

The feature vector is the teacher state (screwhead/teacher/state.py) followed by the arm's
specification and its feasibility at this state:

  spec          the arm's body screw axes, joint limits and reach -- constant per arm, carried so
                the input is the same shape of thing on another arm rather than a Panda-shaped one
  feasibility   the smallest singular value of the body Jacobian and each joint's distance to its
                nearest limit, both from the decode (screwhead/geometry/ik.py)

The output is a plan in the parameterization BRN-screw-search declares: per segment, a normalized
body twist (6, moment first) bounded to the +-1 box, and logits over the student's snap levels.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..geometry import ik
from ..geometry.kinematics import body_jacobian
from .search import LEVELS, Settings
from .state import TeacherState


class TeacherNet(nn.Module):
    """One plan per state: a twist and a level per segment, from a plain MLP."""

    def __init__(self, in_dim: int, segments: int, levels: int = len(LEVELS),
                 width: int = 512, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.GELU()]
            d = width
        self.trunk = nn.Sequential(*layers)
        self.twist = nn.Linear(width, segments * 6)
        self.level = nn.Linear(width, segments * levels)
        self.segments, self.levels, self.in_dim = segments, levels, in_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        twist = torch.tanh(self.twist(h)).view(-1, self.segments, 6)
        logits = self.level(h).view(-1, self.segments, self.levels)
        return twist, logits


class Features:
    """The state pi_theta reads, assembled the same way when labelling and when fitting."""

    def __init__(self, env, verdicts):
        self.env, self.v = env, verdicts
        self.state = TeacherState(env, verdicts)
        chain = env.chain
        limits = chain.limits.detach().numpy().astype(float)
        self.limits = limits
        reach = float(np.abs(chain.M.detach().numpy()[:3, 3]).sum())
        self.spec = np.concatenate([
            chain.B.detach().numpy().astype(float).ravel(),      # body screw axes, one per joint
            limits.ravel(),
            [reach, float(chain.n)],
        ]).astype(np.float32)
        self.dim = int(self.state.dim + self.spec.size + 1 + chain.n)

    def feasibility(self) -> np.ndarray:
        """What the decode knows about this configuration: conditioning and limit margins."""
        q = np.asarray(self.env.scene.raw()[1].qpos[self.env.robot._ref_joint_pos_indexes], float)
        theta = torch.tensor(q, dtype=torch.float64)[None]
        sigma = float(ik.sigma_min(self.env.chain, theta)[0])
        margin = np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q)
        return np.concatenate([[sigma], margin]).astype(np.float32)

    def vector(self, watch, start_reference: dict) -> np.ndarray:
        return np.concatenate([self.state.vector(watch, start_reference), self.spec,
                               self.feasibility()]).astype(np.float32)


class Policy:
    """pi_theta as the search takes it: the plan it starts from, with the weights frozen.

    Called by Search._start once per control step (BRN-optimization-teacher): it returns the mean
    twist per segment and a categorical over snap levels, floored so every level keeps positive
    probability -- the floor is the search's, applied there.
    """

    def __init__(self, net: TeacherNet, features: Features, settings: Settings, device: str = "cpu"):
        self.net, self.features, self.s = net.to(device).eval(), features, settings
        self.device = device
        for p in self.net.parameters():
            p.requires_grad_(False)

    def __call__(self, env, verdicts, watch, start_reference: dict) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(self.features.vector(watch, start_reference)).to(self.device)[None]
        with torch.no_grad():
            twist, logits = self.net(x)
            probs = torch.softmax(logits, dim=-1)
        return twist[0].cpu().numpy().astype(float), probs[0].cpu().numpy().astype(float)


def save(path: str | Path, net: TeacherNet, features: Features, settings: Settings,
         suite: str, task: int, rounds: int, kept: int) -> None:
    """The weights with everything needed to decode them: the feature layout they were fitted
    against, the plan shape, the snap levels, and the search settings the labels came from."""
    torch.save({
        "state_dict": net.state_dict(),
        "in_dim": net.in_dim, "segments": net.segments, "levels": net.levels,
        "state_dim": int(features.state.dim), "spec": features.spec, "levels_m": list(LEVELS),
        "settings": asdict(settings), "suite": suite, "task": task,
        "rounds": rounds, "kept_episodes": kept,
    }, Path(path))


def load(path: str | Path, env, verdicts, device: str = "cpu") -> tuple[Policy, dict]:
    """Rebuild pi_theta against this environment, refusing a checkpoint whose feature layout or
    plan shape does not match the one here -- a plan decoded against another layout is not a plan."""
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    features = Features(env, verdicts)
    if int(blob["in_dim"]) != features.dim:
        raise SystemExit(f"{path}: fitted on {blob['in_dim']} features, this task builds {features.dim}")
    if list(blob["levels_m"]) != list(LEVELS):
        raise SystemExit(f"{path}: fitted on snap levels {blob['levels_m']}, this build uses {list(LEVELS)}")
    settings = Settings(**blob["settings"])
    if blob["segments"] != settings.segments:
        raise SystemExit(f"{path}: {blob['segments']} segments in the weights, {settings.segments} in its settings")
    net = TeacherNet(features.dim, settings.segments, blob["levels"])
    net.load_state_dict(blob["state_dict"])
    return Policy(net, features, settings, device), blob
