"""LIBERO demonstrations, as chains and trajectories.

The demonstrations are Panda, OSC_POSE, 20 Hz, and their env_args confirm the
ActionSpec defaults independently:

    control_freq 20, output_max [0.05]*3 + [0.5]*3, control_delta true,
    uncouple_pos_ori true

Two frame facts, both established by measurement against the recordings rather
than assumed:

  TOOL OFFSET. LIBERO records ee_pos at the gripper's grip_site, 97 mm beyond
  the right_hand body that FK measures to. Fitting the offset over 6079 frames
  gives 0.0972 m against the 0.097 in panda_gripper.xml. Ignoring it leaves
  every pose wrong by 97 mm while looking entirely plausible.

  BASE PLACEMENT. The arm sits at roughly (-0.660, 0, 0.912) in LIBERO's world.
  Retargeting does not need it: a body twist log(T^-1 T') is invariant to left
  multiplication, so where the robot stands cancels out.

A residual of ~0.33 mm remains between FK(joint_states) and the recorded ee_pos.
It is NOT a constant offset -- calibrating the offset does not reduce it -- and
it correlates with tool speed at r = 0.86, rising from 0.14 mm at rest to 0.37
mm at 0.3 m/s. That is about 1.2 ms of lag, half a physics step: the two
quantities are sampled across a step boundary. It is a recording artifact, not
a kinematics error, and retargeting reads joint_states directly so it never
propagates.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
from torch import Tensor

from .mjcf import from_mjcf
from .poe import Chain

ROBOSUITE_ROBOTS = Path(os.environ.get(
    "ROBOT_ASSETS",
    "/home/edge-host/Documents/GitHub/vla_jepa/.venv/lib/python3.12/site-packages/robosuite/models/assets/robots",
))
DATASETS = Path(os.environ.get(
    "LIBERO_DATASETS",
    "/home/edge-host/Documents/GitHub/embodied_ai/third_party/LIBERO/libero/datasets",
))
GRIP_SITE_Z = 0.0972          # calibrated over 6079 frames; 0.097 in the XML


def panda_chain(tool_z: float = GRIP_SITE_Z) -> Chain:
    """The arm LIBERO actually demonstrates on, tool frame at the grip site."""
    base = from_mjcf(ROBOSUITE_ROBOTS / "panda" / "robot.xml", angle="radian", name="panda")
    off = torch.eye(4, dtype=base.M.dtype)
    off[2, 3] = tool_z
    return base.with_tool(off, name="libero-panda")


def task_files(suite: str = "libero_spatial") -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(DATASETS / suite / "*.hdf5")))


def demos(path: Path, limit: int | None = None):
    """Yield (joint_states, actions, gripper_states) per demonstration."""
    import h5py
    with h5py.File(path, "r") as f:
        keys = list(f["data"].keys())
        if limit is not None:
            keys = keys[:limit]
        for k in keys:
            g = f["data"][k]
            yield (
                torch.tensor(g["obs"]["joint_states"][:], dtype=torch.float64),
                torch.tensor(g["actions"][:], dtype=torch.float64),
                torch.tensor(g["obs"]["gripper_states"][:], dtype=torch.float64),
            )
