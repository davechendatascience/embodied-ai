"""Does the keypose -> cuRobo frame math actually hold? Measured, not asserted.

The pipeline hands cuRobo a goal in the ROBOT BASE frame, derived from a
keypose recorded in the WORLD frame:

    T_world_tcp  = T_world_base . T_base_flange . T_flange_tcp
    goal         = T_base_flange = (T_world_base)^-1 . T_world_tcp . (T_flange_tcp)^-1

Every step above is an assumption until it is measured. Three of them can fail
silently and would present as "cuRobo plans badly" rather than as a frame bug:

  C1  T_flange_tcp is CONSTANT across arm configurations. If the grip site is
      not rigidly fixed to the flange -- e.g. it tracks the fingers, which move
      -- then a single measured offset is wrong everywhere except the pose it
      was measured at. `gripper_points.py` already documents that closing the
      gripper costs 37.6 mm of FK error under a rigidity assumption, so this is
      a live risk, not a formality.

  C2  T_flange_tcp is IDENTICAL on both arms with the matched gripper. This is
      what licenses replaying a Panda keypose on a UR5e verbatim. Measured once
      at the rest pose it was 0.0000 mm; that is one configuration, not a proof.

  C3  T_world_base is CONSTANT and identical on both arms. cuRobo plans in the
      base frame, so a base that moves -- or differs between arms -- silently
      offsets every goal.

And the round trip must close: reconstructing T_world_tcp from the three
factors has to return the pose MuJoCo reports, to numerical precision.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_frame_math.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

from rekep_libero import add_rekep_to_path  # noqa: E402

add_rekep_to_path()

N_SAMPLES = 40
SEED = 7

# tolerances: sub-millimetre and sub-milliradian, because these are supposed to
# be exact rigid-body relations, not approximations
POS_TOL_MM = 0.01
ROT_TOL_DEG = 0.01


def homo(pos, mat):
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = pos
    return T


def inv(T):
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def pose_delta(A, B):
    """(translation mm, rotation deg) between two homogeneous transforms."""
    d = inv(A) @ B
    ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1.0) / 2.0, -1, 1)))
    return float(np.linalg.norm(d[:3, 3]) * 1000.0), float(ang)


def frames(env):
    """T_world_base, T_world_flange, T_world_tcp from the live sim."""
    sim = env.sim
    m, d = sim.model, sim.data
    robot = env.env.robots[0]
    tcp = m.site_name2id(robot.controller.eef_name)
    base = m.body_name2id("robot0_base")
    flange = m.body_name2id("robot0_right_hand")
    return (
        homo(d.body_xpos[base], d.body_xmat[base].reshape(3, 3)),
        homo(d.body_xpos[flange], d.body_xmat[flange].reshape(3, 3)),
        homo(d.site_xpos[tcp], d.site_xmat[tcp].reshape(3, 3)),
    )


def sample_configs(env, n, rng):
    """Random arm configurations inside the joint limits, forward-kinematics only."""
    import mujoco

    sim = env.sim
    robot = env.env.robots[0]
    idx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    lo, hi = sim.model.jnt_range[robot._ref_joint_indexes].T
    out = []
    backup = sim.data.qpos.copy()
    for _ in range(n):
        # stay off the limits themselves; a joint pinned at its stop is not a
        # representative configuration
        q = rng.uniform(lo + 0.1, hi - 0.1)
        sim.data.qpos[idx] = q
        mujoco.mj_forward(sim.model._model, sim.data._data)
        out.append(frames(env))
    sim.data.qpos[:] = backup
    mujoco.mj_forward(sim.model._model, sim.data._data)
    return out


def main():
    import test_ur5e_scene as T
    from rekep_libero import fixtures as fx

    rng = np.random.default_rng(SEED)
    failures = []
    per_arm = {}

    panda = T.build("Panda", gripper="PandaGripper")
    ref = fx.snapshot(panda.sim)
    ur5e = T.build("UR5e", gripper="PandaGripper", fixture_ref=ref)

    for name, env in (("Panda", panda), ("UR5e", ur5e)):
        samples = sample_configs(env, N_SAMPLES, rng)

        # ---- C1: T_flange_tcp constant across configurations -------------
        offsets = [inv(Tf) @ Tt for _Tb, Tf, Tt in samples]
        ref_off = offsets[0]
        worst = max(pose_delta(ref_off, o) for o in offsets)
        print(f"{name}: C1 T_flange_tcp constant over {N_SAMPLES} configs -> "
              f"worst {worst[0]:.5f} mm / {worst[1]:.5f} deg")
        if worst[0] > POS_TOL_MM or worst[1] > ROT_TOL_DEG:
            failures.append(f"{name} C1: flange->tcp varies by {worst[0]:.4f} mm")
        per_arm[name] = ref_off

        # ---- C3: T_world_base constant --------------------------------
        bases = [Tb for Tb, _f, _t in samples]
        worst_b = max(pose_delta(bases[0], B) for B in bases)
        print(f"{name}: C3 T_world_base constant                -> "
              f"worst {worst_b[0]:.5f} mm / {worst_b[1]:.5f} deg")
        if worst_b[0] > POS_TOL_MM:
            failures.append(f"{name} C3: base moves by {worst_b[0]:.4f} mm")

        # ---- round trip -------------------------------------------------
        worst_rt = (0.0, 0.0)
        for Tb, Tf, Tt in samples:
            recon = Tb @ (inv(Tb) @ Tf) @ (inv(Tf) @ Tt)
            worst_rt = max(worst_rt, pose_delta(Tt, recon))
        print(f"{name}: round trip world = base.base_flange.flange_tcp -> "
              f"worst {worst_rt[0]:.6f} mm / {worst_rt[1]:.6f} deg")
        if worst_rt[0] > POS_TOL_MM:
            failures.append(f"{name} round trip off by {worst_rt[0]:.4f} mm")

        # ---- C4/C5: reconstruct the frames from the OBSERVATION fields ----
        # This is what a trace actually carries, and the pair is mixed:
        # eef_pos is the SITE, eef_quat is the hand BODY, 90 deg apart.
        import mujoco

        from rekep_libero.frames import flange_from_obs, tcp_from_obs

        sim = env.sim
        robot = env.env.robots[0]
        idx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        lo, hi = sim.model.jnt_range[robot._ref_joint_indexes].T
        fl = sim.model.body_name2id("robot0_right_hand")
        st = sim.model.site_name2id(robot.controller.eef_name)
        w4 = w5 = (0.0, 0.0)
        backup = sim.data.qpos.copy()
        for _ in range(N_SAMPLES):
            sim.data.qpos[idx] = rng.uniform(lo + 0.1, hi - 0.1)
            mujoco.mj_forward(sim.model._model, sim.data._data)
            # robosuite's observables only refresh on step(); writing qpos
            # directly leaves _get_observations() returning the PREVIOUS pose,
            # which measured as a 1069 mm error and looks exactly like broken
            # frame math. Same trap get_ee_pose()'s docstring documents.
            env.env._update_observables(force=True)
            obs = env.env._get_observations()
            w4 = max(w4, pose_delta(
                homo(sim.data.body_xpos[fl], sim.data.body_xmat[fl].reshape(3, 3)),
                flange_from_obs(obs["robot0_eef_pos"], obs["robot0_eef_quat"])))
            w5 = max(w5, pose_delta(
                homo(sim.data.site_xpos[st], sim.data.site_xmat[st].reshape(3, 3)),
                tcp_from_obs(obs["robot0_eef_pos"], obs["robot0_eef_quat"])))
        sim.data.qpos[:] = backup
        mujoco.mj_forward(sim.model._model, sim.data._data)
        print(f"{name}: C4 flange_from_obs vs true flange        -> "
              f"worst {w4[0]:.6f} mm / {w4[1]:.6f} deg")
        print(f"{name}: C5 tcp_from_obs vs true grip site        -> "
              f"worst {w5[0]:.6f} mm / {w5[1]:.6f} deg")
        if w4[0] > POS_TOL_MM or w4[1] > ROT_TOL_DEG:
            failures.append(f"{name} C4: flange_from_obs off by {w4[0]:.4f} mm")
        if w5[0] > POS_TOL_MM or w5[1] > ROT_TOL_DEG:
            failures.append(f"{name} C5: tcp_from_obs off by {w5[0]:.4f} mm")

        # what cuRobo will actually be handed, for one sample
        Tb, Tf, Tt = samples[0]
        goal = inv(Tb) @ Tt @ inv(per_arm[name])
        print(f"{name}: example cuRobo goal (base frame) "
              f"pos {np.round(goal[:3, 3], 4)}")
        print()

    # ---- C2: same offset on both arms ----------------------------------
    d_pos, d_rot = pose_delta(per_arm["Panda"], per_arm["UR5e"])
    print(f"C2 T_flange_tcp Panda vs UR5e (matched gripper) -> "
          f"{d_pos:.5f} mm / {d_rot:.5f} deg")
    print(f"   Panda flange->tcp translation {np.round(per_arm['Panda'][:3, 3], 5)}")
    print(f"   UR5e  flange->tcp translation {np.round(per_arm['UR5e'][:3, 3], 5)}")
    if d_pos > POS_TOL_MM or d_rot > ROT_TOL_DEG:
        failures.append(f"C2: flange->tcp differs between arms by {d_pos:.4f} mm")

    # ---- base frames equal across arms ---------------------------------
    bp, _, _ = frames(panda)
    bu, _, _ = frames(ur5e)
    d_pos_b, d_rot_b = pose_delta(bp, bu)
    print(f"   T_world_base Panda vs UR5e -> {d_pos_b:.5f} mm / {d_rot_b:.5f} deg")
    if d_pos_b > POS_TOL_MM:
        failures.append(f"base frames differ by {d_pos_b:.4f} mm")

    panda.close()
    ur5e.close()

    print()
    if failures:
        print("FRAME MATH FAILS")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("FRAME MATH HOLDS — keypose -> base-frame goal is exact for both arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
