"""Replay a Panda-recorded LIBERO init state on a different embodiment.

LIBERO ships its 50 pinned init states as FLATTENED MUJOCO STATES —
``[time, qpos, qvel]`` — recorded against the Panda model. Measured on
``libero_goal/0``:

    nq = 41, nv = 37, flattened length = 79, init_states.shape = (50, 79)

    qpos[0:7]   robot0_joint1..7
    qpos[7:9]   gripper0_finger_joint1..2
    qpos[9:41]  the objects: four free joints (7 each) + wooden_cabinet's
                three drawer slides + flat_stove's button

Swap the Panda for a UR5e and every one of those numbers moves. A UR5e carries
six arm joints, and Robotiq85 six gripper joints, so the robot block grows
9 -> 12 and **every object address shifts by +3**. ``set_init_state`` is
``sim.set_state_from_flattened``, which reads positionally: hand it the
Panda array and it either raises on the length or, worse, writes drawer
positions into a wine bottle's quaternion.

This is exactly the failure mode ``NOTES.md`` §1 is about — a state restore
that half-works and silently changes the experiment — so the layout is
ASSERTED here rather than assumed, and the assertion is derived from the
target model's own joint table instead of hard-coded.

**What transfers and what does not.** The object block transfers verbatim:
same bddl, same objects, same builder, so the non-robot joints appear in the
same order in both models. The robot block does not transfer at all and is
left at whatever ``reset()`` produced — the arm's own rest pose. That is a real
difference from the Panda runs, where the recorded arm pose varies ~0.11 rad
across the 50 states, and it is the honest choice: there is no meaningful
mapping from seven Panda joints to six UR5e ones. Velocities are zeroed;
measured, all 50 ``libero_goal`` states carry ``|qvel| = 0`` exactly, so on
that suite this is not an approximation. Other suites are NOT yet verified —
``remap_panda_init_state`` warns if it meets a moving state rather than
quietly discarding momentum.

The order assumption is verified end-to-end, not argued: see
``tests/test_ur5e_scene.py``, which applies the same recorded state to a real
Panda scene and a real UR5e scene and compares every object's world pose.
"""

import numpy as np

# MuJoCo joint types -> (qpos width, qvel width).
_JNT_WIDTHS = {
    0: (7, 6),  # free
    1: (4, 3),  # ball
    2: (1, 1),  # slide
    3: (1, 1),  # hinge
}

# The Panda block in a LIBERO-recorded state: 7 arm + 2 finger joints, all
# hinge/slide, so nine numbers in qpos and nine in qvel. Measured on
# libero_goal/0 and stable across suites (the ROBOT does not change between
# them, only the object count does).
PANDA_ROBOT_NQ = 9
PANDA_ROBOT_NV = 9

ROBOT_BODY_PREFIXES = ("robot", "gripper", "mount")


def joint_blocks(sim, robot_prefixes=ROBOT_BODY_PREFIXES):
    """Split a model's joints into robot and object, in address order.

    Returns ``(robot, objects)``, each a list of
    ``(name, qpos_addr, qpos_width, qvel_addr, qvel_width)``.
    """
    m = sim.model
    robot, objects = [], []
    for j in range(m.njnt):
        name = m.joint_id2name(j)
        body = m.body_id2name(m.jnt_bodyid[j])
        jtype = int(m.jnt_type[j])
        if jtype not in _JNT_WIDTHS:
            raise ValueError(f"joint {name!r} has unsupported type {jtype}")
        qw, vw = _JNT_WIDTHS[jtype]
        entry = (name, int(m.jnt_qposadr[j]), qw, int(m.jnt_dofadr[j]), vw)
        target = robot if any(body.startswith(p) for p in robot_prefixes) else objects
        target.append(entry)
    robot.sort(key=lambda e: e[1])
    objects.sort(key=lambda e: e[1])
    return robot, objects


def remap_panda_init_state(state, sim, warn=print):
    """Rewrite a Panda-recorded flattened state for the model behind ``sim``.

    Identity when ``sim`` IS a Panda scene, so the Panda experiments (E0/E1)
    keep bit-for-bit the state LIBERO recorded, including its arm pose.
    """
    state = np.asarray(state, dtype=np.float64).ravel()
    m = sim.model
    robot, objects = joint_blocks(sim)

    here_len = 1 + m.nq + m.nv
    robot_nq = sum(e[2] for e in robot)
    robot_nv = sum(e[4] for e in robot)
    if len(state) == here_len and robot_nq == PANDA_ROBOT_NQ:
        return state  # already a Panda; touch nothing

    n_obj_q = sum(e[2] for e in objects)
    n_obj_v = sum(e[4] for e in objects)
    expected = 1 + (PANDA_ROBOT_NQ + n_obj_q) + (PANDA_ROBOT_NV + n_obj_v)
    if len(state) != expected:
        raise ValueError(
            f"recorded init state has length {len(state)}, but this scene's "
            f"{len(objects)} object joints ({n_obj_q} qpos / {n_obj_v} qvel) "
            f"imply a Panda recording of length {expected}. Either the object "
            f"set differs from the one that was recorded, or the robot block "
            f"is not the 9-joint Panda this assumes."
        )

    src_q = state[1 + PANDA_ROBOT_NQ: 1 + PANDA_ROBOT_NQ + n_obj_q]
    src_v = state[1 + PANDA_ROBOT_NQ + n_obj_q + PANDA_ROBOT_NV:]
    assert len(src_v) == n_obj_v, (len(src_v), n_obj_v)

    if np.any(src_v != 0.0) and warn is not None:
        warn(
            f"init_state: recorded object velocities are NOT zero "
            f"(|qvel|max = {np.abs(src_v).max():.3e}); this remap drops them. "
            f"Measured zero on libero_goal — verify before trusting this suite."
        )

    # Start from the state reset() just produced: that already holds the
    # target robot's own rest pose and its gripper opening, which is precisely
    # what we want to keep.
    here = sim.get_state()
    qpos = np.array(here.qpos, dtype=np.float64, copy=True)
    qvel = np.zeros(m.nv, dtype=np.float64)

    cursor = 0
    for _name, qadr, qw, _vadr, _vw in objects:
        qpos[qadr: qadr + qw] = src_q[cursor: cursor + qw]
        cursor += qw
    assert cursor == n_obj_q, (cursor, n_obj_q)

    return np.concatenate([[0.0], qpos, qvel])


def describe(sim):
    """One-line layout summary — for logs, and for arguing with this file."""
    robot, objects = joint_blocks(sim)
    return (
        f"nq={sim.model.nq} nv={sim.model.nv} "
        f"robot={len(robot)}j/{sum(e[2] for e in robot)}q "
        f"objects={len(objects)}j/{sum(e[2] for e in objects)}q "
        f"first_object_qpos_addr={objects[0][1] if objects else None}"
    )
