"""Does a UR5e LIBERO scene hold the same world as the Panda one?

The cross-embodiment experiment is only meaningful if the SCENE is invariant:
same objects, same poses, same fixtures, same base position, with the arm as
the only thing that changed. That is not automatic — LIBERO's init states are
Panda-shaped flattened MuJoCo states and the placement sampler runs off
`np.random` during `reset()`, so both could drift for reasons that look like
nothing at all.

This builds both scenes under the same seed and the same recorded init state,
and measures:

  1. does a UR5e scene build at all
  2. object world poses, Panda vs UR5e            -> must match
  3. fixture (world-child body) poses             -> must match
  4. robot base position                          -> must match
  5. the start end-effector pose of each arm      -> reported, NOT asserted

(5) is the number that decides whether the UR5e's rest pose is usable as an
episode start, and there is no defensible tolerance for it yet — the arms are
different, so their rest poses differ, and the question is only whether the
UR5e's is sane. Print it, look at it, then pin it.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_ur5e_scene.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402

add_rekep_to_path()

from rekep_libero import fixtures as fx  # noqa: E402
from rekep_libero.init_state import (  # noqa: E402
    ROBOT_BODY_PREFIXES,
    describe,
    remap_panda_init_state,
)

SUITE = "libero_goal"
TASK_ID = 0
INIT_STATE_ID = 0
SEED = 0

POSE_TOL_MM = 1.0
QUAT_TOL_DEG = 1.0


def build(robot, gripper="default", fixture_ref=None):
    """A bare LIBERO scene — no ReKep config, no grasp stack, no IK solver."""
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from rekep_libero.grasp_cgn import allow_numpy_unpickling

    if robot != "Panda":
        from rekep_libero import robots_ur5e
        assert robots_ur5e.registered(), "MountedUR5e failed to register"

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)

    allow_numpy_unpickling(torch)
    states = suite.get_task_init_states(TASK_ID)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl, robots=[robot], gripper_types=gripper,
        camera_heights=128, camera_widths=128,
        controller="OSC_POSE", camera_depths=False,
        camera_names=["agentview"], horizon=20000,
    )
    np.random.seed(SEED)          # fixtures are placed during reset(), not restored
    env.reset()
    if fixture_ref is not None:   # ...and seeding alone does not survive a robot swap
        fx.pin(env.sim, fixture_ref)
    state = remap_panda_init_state(states[INIT_STATE_ID], env.sim)
    env.set_init_state(state)
    return env


def settle(env, steps=10):
    """As ReKepLiberoEnv does — objects drop onto the table before anyone looks."""
    for _ in range(steps):
        env.step(np.concatenate([np.zeros(env.env.action_dim - 1), [-1.0]]))
    return env


def scene_snapshot(env):
    """Every non-robot body's world pose, plus the arm's start pose."""
    sim = env.sim
    m, d = sim.model, sim.data
    bodies = {}
    for b in range(m.nbody):
        name = m.body_id2name(b)
        if not name or name == "world":
            continue
        if any(name.startswith(p) for p in ROBOT_BODY_PREFIXES):
            continue
        bodies[name] = (d.body_xpos[b].copy(), d.body_xquat[b].copy())

    rb = env.env.robots[0]
    eef_site = m.site_name2id(rb.controller.eef_name)
    return dict(
        bodies=bodies,
        base=d.body_xpos[m.body_name2id("robot0_base")].copy(),
        eef_pos=d.site_xpos[eef_site].copy(),
        eef_mat=d.site_xmat[eef_site].copy().reshape(3, 3),
        arm_dof=len(rb._ref_joint_pos_indexes),
        gripper=type(rb.gripper).__name__,
        layout=describe(sim),
    )


def quat_angle_deg(qa, qb):
    dot = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))


def main():
    failures = []

    print("=" * 70)
    print("PANDA")
    print("=" * 70)
    panda = build("Panda")
    ref = fx.snapshot(panda.sim)   # the reference placement every other arm must match
    a_pre = scene_snapshot(panda)
    a = scene_snapshot(settle(panda))
    print(f"  {a['layout']}")
    print(f"  fixtures pinned from this scene: {len(ref)}")
    print(f"  arm dof {a['arm_dof']}   gripper {a['gripper']}")
    print(f"  base    {np.round(a['base'], 4)}")
    print(f"  eef pos {np.round(a['eef_pos'], 4)}")
    panda.close()

    print()
    print("=" * 70)
    print("UR5e  (Panda gripper — the E3 configuration)")
    print("=" * 70)
    ur5e = build("UR5e", gripper="PandaGripper", fixture_ref=ref)
    b_pre = scene_snapshot(ur5e)
    b = scene_snapshot(settle(ur5e))
    print(f"  {b['layout']}")
    print(f"  arm dof {b['arm_dof']}   gripper {b['gripper']}")
    print(f"  base    {np.round(b['base'], 4)}")
    print(f"  eef pos {np.round(b['eef_pos'], 4)}")
    ur5e.close()

    print()
    print("=" * 70)
    print("SCENE INVARIANCE")
    print("=" * 70)

    only_a = sorted(set(a["bodies"]) - set(b["bodies"]))
    only_b = sorted(set(b["bodies"]) - set(a["bodies"]))
    if only_a or only_b:
        failures.append(f"body sets differ: only-Panda={only_a} only-UR5e={only_b}")
    common = sorted(set(a["bodies"]) & set(b["bodies"]))

    def drifts(sa, sb):
        out = []
        for name in common:
            pa, qa = sa["bodies"][name]
            pb, qb = sb["bodies"][name]
            out.append((name,
                        float(np.linalg.norm(pa - pb)) * 1000.0,
                        quat_angle_deg(qa, qb)))
        return sorted(out, key=lambda e: -e[1])

    # Split the two causes apart. BEFORE settling, any difference is placement
    # or a bad state remap. AFTER settling, it can also be physics -- a
    # different arm touching the scene, or contacts resolving differently.
    pre, post = drifts(a_pre, b_pre), drifts(a, b)
    print(f"  bodies compared          {len(common)}")
    print(f"  {'body':<34}{'pre-settle mm':>15}{'post-settle mm':>16}")
    pre_by_name = {n: d for n, d, _ in pre}
    for name, dp, _dr in post[:6]:
        print(f"  {name:<34}{pre_by_name[name]:>15.4f}{dp:>16.4f}")

    worst_pos = (post[0][0], post[0][1])
    worst_rot = max(((n, r) for n, _d, r in post), key=lambda e: e[1])
    print(f"  worst position drift     {worst_pos[1]:.4f} mm   ({worst_pos[0]})")
    print(f"  worst orientation drift  {worst_rot[1]:.4f} deg  ({worst_rot[0]})")
    print(f"  worst PRE-settle drift   {pre[0][1]:.4f} mm   ({pre[0][0]})")

    if worst_pos[1] > POSE_TOL_MM:
        failures.append(f"object position drift {worst_pos[1]:.3f} mm "
                        f"> {POSE_TOL_MM} mm on {worst_pos[0]}")
    if worst_rot[1] > QUAT_TOL_DEG:
        failures.append(f"object orientation drift {worst_rot[1]:.3f} deg "
                        f"> {QUAT_TOL_DEG} deg on {worst_rot[0]}")

    dbase = float(np.linalg.norm(a["base"] - b["base"])) * 1000.0
    print(f"  robot base offset        {dbase:.4f} mm")
    if dbase > POSE_TOL_MM:
        failures.append(f"robot bases differ by {dbase:.3f} mm — the arms are "
                        f"not standing in the same place, so reachability is "
                        f"not comparable")

    print()
    print("  START POSE — reported, not asserted:")
    d_eef = float(np.linalg.norm(a["eef_pos"] - b["eef_pos"])) * 1000.0
    print(f"    Panda eef {np.round(a['eef_pos'], 4)}")
    print(f"    UR5e  eef {np.round(b['eef_pos'], 4)}")
    print(f"    separation {d_eef:.1f} mm")
    print(f"    Panda tool axis (local +Z in world) {np.round(a['eef_mat'][:, 2], 3)}")
    print(f"    UR5e  tool axis (local +Z in world) {np.round(b['eef_mat'][:, 2], 3)}")
    # Does the stock OSC_POSE config hold each arm still under a zero command?
    # LIBERO hands both robots the generic osc_pose.json rather than
    # default_ur5e.json, so this is the first place Panda-tuned gains could
    # show up -- and an arm that sags under zero action makes any dense-delta
    # baseline meaningless before cuRobo enters the picture.
    print(f"    Panda eef sag over 10 zero-action steps "
          f"{np.linalg.norm(a['eef_pos'] - a_pre['eef_pos']) * 1000:.2f} mm")
    print(f"    UR5e  eef sag over 10 zero-action steps "
          f"{np.linalg.norm(b['eef_pos'] - b_pre['eef_pos']) * 1000:.2f} mm")

    print()
    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED — the UR5e scene is the Panda scene with a different arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
