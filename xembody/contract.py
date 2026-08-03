"""Make the frame errors unrepresentable instead of debuggable.

A goal is not a 4x4 matrix. A 4x4 matrix means nothing without the frame it is
expressed in, and every expensive failure in this project has been a matrix
handed across a boundary where that frame silently changed:

  * a Panda base at [-0.60, 0, 0.000] and a UR5e base at [-0.60, 0, 0.912] in
    the SAME scene -- 912 mm, and the planner reported "unreachable"
  * base frames verified in `libero_goal` and assumed for `libero_object`,
    which uses a different arena entirely
  * a TCP offset that is a property of the GRIPPER, reused across grippers

Each produced a plausible-looking failure with a plausible-looking mechanism --
reach limits, collisions, self-collision, orientation tolerance, a missing
cspace -- and five of those mechanisms were investigated and refuted before
anyone checked the frame. The measurement that would have ended it took one
line.

So: carry the frame with the pose, and refuse to combine mismatched ones.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Where a robot stands and what its tool offset is."""

    robot: str
    scene: str
    base_link: str
    base_pos: tuple
    base_rpy_mat: tuple      # row-major 3x3, as a tuple of tuples
    tcp_z: float

    @staticmethod
    def from_sim(sim, robot, scene, base_link="robot0_base",
                 tool_site=None, flange="robot0_right_hand"):
        m, d = sim.model, sim.data
        b = m.body_name2id(base_link)
        tcp = 0.0
        if tool_site is not None:
            s = d.site_xpos[m.site_name2id(tool_site)]
            f = d.body_xpos[m.body_name2id(flange)]
            tcp = float(np.linalg.norm(s - f))
        R = np.asarray(d.body_xmat[b], float).reshape(3, 3)
        return Frame(robot, scene, base_link,
                     tuple(float(v) for v in d.body_xpos[b]),
                     tuple(tuple(float(v) for v in r) for r in R), tcp)

    def delta(self, other):
        dp = float(np.linalg.norm(np.array(self.base_pos)
                                  - np.array(other.base_pos))) * 1000.0
        Ra, Rb = np.array(self.base_rpy_mat), np.array(other.base_rpy_mat)
        ang = float(np.degrees(np.arccos(
            np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
        return dp, ang, abs(self.tcp_z - other.tcp_z) * 1000.0


@dataclass
class Goal:
    """A pose that knows which frame it is in. Pass these, never bare arrays."""

    T: np.ndarray
    frame: Frame
    label: str = ""
    meta: dict = field(default_factory=dict)


def require_compatible(a: Frame, b: Frame, pos_tol_mm=1.0, rot_tol_deg=0.1,
                       tcp_tol_mm=0.5):
    """Refuse to transfer poses between incompatible frames. LOUDLY.

    This is the check that would have saved five wrong mechanisms. It is four
    comparisons and it runs in microseconds.
    """
    dp, ang, dt = a.delta(b)
    problems = []
    if dp > pos_tol_mm:
        problems.append(f"base positions differ by {dp:.1f} mm "
                        f"({a.robot}@{a.scene} {a.base_pos} vs "
                        f"{b.robot}@{b.scene} {b.base_pos})")
    if ang > rot_tol_deg:
        problems.append(f"base rotations differ by {ang:.3f} deg")
    if dt > tcp_tol_mm:
        problems.append(f"TCP offsets differ by {dt:.2f} mm "
                        f"({a.tcp_z:.4f} vs {b.tcp_z:.4f} m) -- TCP is a "
                        f"property of the GRIPPER, re-measure it")
    if a.scene != b.scene:
        problems.append(f"different scenes ({a.scene} vs {b.scene}); LIBERO "
                        f"suites use different ARENAS and place robots by "
                        f"different rules")
    if problems:
        raise FrameMismatch(
            "poses are not transferable between these frames:\n  - "
            + "\n  - ".join(problems))
    return True


class FrameMismatch(ValueError):
    """Raised instead of letting a silent offset become a fake result."""


def validate_robot_cfg(cfg):
    """Catch the cuRobo config omissions that fail as 'infeasible'."""
    k = cfg["robot_cfg"]["kinematics"]
    missing = [x for x in ("cspace", "urdf_path", "base_link", "tool_frames",
                           "collision_spheres") if x not in k]
    if missing:
        raise ValueError(
            f"robot config missing {missing}. Without `cspace` cuRobo has no "
            f"seed configuration or convergence limits and reports failure "
            f"even at 0.0 mm position error.")
    c = k["cspace"]
    for f in ("joint_names", "default_joint_position", "null_space_weight",
              "cspace_distance_weight", "max_acceleration", "max_jerk"):
        if f not in c:
            raise ValueError(f"cspace missing `{f}` (note: the field is "
                             f"`default_joint_position`, not `retract_config`)")
    clip = c.get("position_limit_clip", 0.0)
    return {"joints": len(c["joint_names"]), "limit_clip": clip}
