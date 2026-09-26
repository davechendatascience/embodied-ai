"""LIBERO's human demonstrations as the Panda VLA's training pairs (BRN-vla-learns-the-recorded-motion), and
the one preprocessing its inputs go through in training and in evaluation (BRN-vla-sees-and-acts-as-trained).

Nothing is re-simulated or re-rendered. The observation stored at step j shows the recorded state j+1
(AXM-libero-demo-observations-follow-their-actions), so its label is the motion from state j+1 to state j+2:
the constant body twist that carries the one tool pose to the other in one control period (the SE(3)
logarithm of their relative pose over the period; forward kinematics only), and the gripper command the
demonstration recorded at step j+1 (+1 closes, -1 opens; AXM-libero-demos-record-their-actions).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry import kin_np
from ..geometry.interface import ActionSpec

DATASETS = Path(__file__).resolve().parents[2] / "third_party/LIBERO/libero/datasets"
ARM_NQ = 7                 # the Panda's arm joints, first in a recorded state after its time entry
PROPRIO_DIM = 10           # tool position (3), the first two columns of its rotation (6), finger opening (1)
SPEC = ActionSpec()        # the control period and the action range the twist is divided by
TWIST_RANGE = np.array([SPEC.max_angular_speed] * 3 + [SPEC.max_linear_speed] * 3)   # moment first


def panda_chain() -> kin_np.NpChain:
    from ..sim.libero_env import PANDA_GRIP_SITE_Z, build_chain
    return kin_np.NpChain.of(build_chain("panda", PANDA_GRIP_SITE_Z))


def upright(image: np.ndarray) -> np.ndarray:
    """robosuite returns images in OpenGL's convention, bottom row first -- the stored ones and the
    evaluation's alike."""
    return np.ascontiguousarray(image[::-1])


def proprio(chain: kin_np.NpChain, joints: np.ndarray, gripper_qpos: np.ndarray) -> np.ndarray:
    """(..., 10): the tool pose in the robot's base frame and the finger opening -- a tool pose rather than
    joint angles, so that another arm's input means the same thing."""
    joints = np.asarray(joints, float)
    T = kin_np.fk(chain, joints.reshape(-1, joints.shape[-1])).reshape(*joints.shape[:-1], 4, 4)
    g = np.asarray(gripper_qpos, float)
    rot6 = np.concatenate([T[..., :3, 0], T[..., :3, 1]], -1)
    return np.concatenate([T[..., :3, 3], rot6, (g[..., :1] - g[..., 1:2])], -1).astype(np.float32)


def recorded_twists(chain: kin_np.NpChain, states: np.ndarray) -> np.ndarray:
    """(T-1, 6): the constant body twist (moment first) from each recorded state's tool pose to the next's
    over one control period, divided by the action range."""
    T = kin_np.fk(chain, np.asarray(states[:, 1:1 + ARM_NQ], float))
    V = kin_np.log_se3(kin_np.inverse(T[:-1]) @ T[1:]) / SPEC.dt
    return (V / TWIST_RANGE).astype(np.float32)


@dataclass(frozen=True)
class Demo:
    """One demonstration's pairs: observation j with the motion from state j+1 to j+2."""
    agentview: np.ndarray      # (N, 128, 128, 3) uint8, upright
    wrist: np.ndarray
    proprio: np.ndarray        # (N, PROPRIO_DIM)
    twist: np.ndarray          # (N, 6) divided by the action range
    gripper: np.ndarray        # (N,) +1 close, -1 open


def load_demo(chain: kin_np.NpChain, group) -> Demo:
    """The pairs of one stored demonstration (an h5py group): N = its length minus two."""
    S, A = group["states"][()], group["actions"][()]
    obs = group["obs"]
    n = len(S) - 2
    twists = recorded_twists(chain, S)
    return Demo(agentview=obs["agentview_rgb"][:n][:, ::-1].copy(), wrist=obs["eye_in_hand_rgb"][:n][:, ::-1].copy(),
                proprio=proprio(chain, obs["joint_states"][:n], obs["gripper_states"][:n]),
                twist=twists[1:n + 1], gripper=A[1:n + 1, -1].astype(np.float32))


def task_file(suite: str, task_name: str) -> Path:
    return DATASETS / suite / f"{task_name}_demo.hdf5"
