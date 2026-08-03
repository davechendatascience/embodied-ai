"""Pin LIBERO's fixtures so they survive a change of robot.

`NOTES.md` §1 established that a pinned init state is not enough: a cabinet
base is a FIXED body whose pose lives in the MODEL, not the state, and
robosuite's placement sampler randomises it during `reset()` through
`np.random`. Seeding `np.random` before `reset()` fixed that — for one robot.

It does not survive a change of embodiment, and the reason is measured rather
than inferred (`robosuite/robots/robot.py:133`):

    noise = np.random.randn(len(self.init_qpos)) * self.initialization_noise["magnitude"]

The draw happens **even when the magnitude is 0.0** — the result is multiplied
by zero, but the numbers are still taken from the stream. Its LENGTH is the
arm's DOF. A Panda takes seven, a UR5e six, so every fixture sampled afterwards
lands somewhere else. Measured on `libero_goal/0` with an identical seed:

    flat_stove_1    6.87 mm     wine_rack_1    6.12 mm     (0.000 deg)

and burning one extra draw before the UR5e's reset takes all of them to
**0.0000 mm** — which identifies the cause exactly and leaves no room for a
"probably physics" explanation.

Compensating for the draw count would work and is two lines, but it encodes
"a Panda has seven joints" and "the noise is one randn per joint" into our
code, where a robosuite upgrade would silently invalidate it and the symptom
would be a 7 mm scene shift that nobody looks for. So instead the fixtures are
pinned OUTRIGHT, against a reference scene, and the correction applied is
reported rather than swallowed. Same rule as everywhere else here: an
experimental condition that can drift must be pinned loudly.
"""

import numpy as np

ROBOT_BODY_PREFIXES = ("robot", "gripper", "mount")


def _fixture_body_ids(sim):
    """World-child bodies with no joints of their own — the fixed furniture.

    Free-jointed objects are excluded on purpose: their poses live in `qpos`
    and are already restored by the recorded init state. Bodies with joints
    (drawers, the stove button) are excluded for the same reason.
    """
    m = sim.model
    ids = []
    for b in range(m.nbody):
        name = m.body_id2name(b)
        if not name or name == "world":
            continue
        if any(name.startswith(p) for p in ROBOT_BODY_PREFIXES):
            continue
        if m.body_parentid[b] != 0 or m.body_jntnum[b] != 0:
            continue
        ids.append((name, b))
    return ids


def snapshot(sim):
    """name -> (body_pos, body_quat), read from the MODEL, not the data."""
    m = sim.model
    return {name: (m.body_pos[b].copy(), m.body_quat[b].copy())
            for name, b in _fixture_body_ids(sim)}


def pin(sim, reference, warn=print, tol_mm=1e-6):
    """Force this scene's fixtures onto the reference poses.

    Returns the largest correction applied, in mm. A non-zero return is not an
    error — it is the sampler divergence this module exists to remove — but it
    is reported, because a correction that grows over time means something
    upstream changed.
    """
    m = sim.model
    here = _fixture_body_ids(sim)
    names = {n for n, _ in here}
    missing = sorted(set(reference) - names)
    extra = sorted(names - set(reference))
    if missing or extra:
        raise ValueError(
            f"fixture sets differ from the reference scene: "
            f"missing={missing} unexpected={extra}. The two scenes are not the "
            f"same world, so pinning them together would hide that."
        )

    worst = 0.0
    for name, b in here:
        pos, quat = reference[name]
        worst = max(worst, float(np.linalg.norm(m.body_pos[b] - pos)) * 1000.0)
        m.body_pos[b] = pos
        m.body_quat[b] = quat
    sim.forward()

    if worst > tol_mm and warn is not None:
        warn(f"fixtures: corrected placement drift of {worst:.4f} mm against "
             f"the reference scene ({len(here)} fixed bodies)")
    return worst
