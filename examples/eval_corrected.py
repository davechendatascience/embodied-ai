"""Closed-loop UR5e rollout with the corrector inline, against the 0/2 baseline.

The corrector was trained on MATCHED poses. In a rollout the arm is wherever it
drifted to, which that data cannot cover -- so this is the test that decides
whether one round of pairs was enough or whether DAgger is required.

Held-out init states: the corrector trained on 0,1,2 (object) and 0,3 (goal).
"""
import argparse, json, os, sys
import numpy as np, imageio

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/VLA_JEPA"); sys.path.insert(0, R + "/third_party/LIBERO")
import libero_ur5e as U
from xembody.pairs import from_raw, to_env
CK = R + "/third_party/VLA_JEPA/checkpoints_hf/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"


def grip2(qpos):
    """Panda-shaped 2-vector from any gripper's joint readout.

    The policy's state input is fixed-width and was trained with a 2-finger
    Panda hand; a Robotiq85 reports 6 joints, so the state vector goes 8 -> 12
    and the server rejects it. Synthesising [w/2, -w/2] from the jaw width is
    faithful because the policy demonstrably IGNORES proprioception -- swapping
    the state between arms changed its output by cos 1.000.
    """
    q = np.asarray(qpos, float).ravel()
    if q.size == 2:
        return q
    w = float(np.abs(q).max()) if q.size else 0.0
    return np.array([w / 2.0, -w / 2.0])


def q2aa(q):
    q = np.asarray(q, float).copy(); q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] ** 2)
    return np.zeros(3) if np.isclose(den, 0) else q[:3] * 2 * np.arccos(q[3]) / den


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", default="UR5e")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--gripper", default="PandaGripper")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-states", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--port", type=int, default=15090)
    ap.add_argument("--corrector", action="store_true")
    ap.add_argument("--align-camera", action="store_true",
                    help="Slide the wrist camera so camera-to-TCP matches the "
                         "Panda's 0.1091 m. A longer gripper moves the TCP away "
                         "from a camera that did not move.")
    ap.add_argument("--align-start", action="store_true",
                    help="IK the arm to the SOURCE arm's start EE pose. Without "
                         "it the UR5e begins 178 mm from where the policy has "
                         "ever seen the scene, from frame one.")
    a = ap.parse_args()

    from examples.LIBERO.model2libero_interface import M1Inference
    import torch as _t

    corr = None
    if a.corrector:
        import glob
        from xembody.adapt import Corrector, featurise
        T, S, TCP, G = [], [], [], []
        for f in sorted(glob.glob(os.path.join(R, "pairs", "*", "[pr]_*.npz"))):
            d = np.load(f); T.append(d["a_target"]); S.append(d["a_source"]); TCP.append(d["tcp"])
            G.append(d["geom"] if "geom" in d.files
                     else np.full((len(d["tcp"]), 2), np.nan, np.float32))
        T = np.concatenate(T); S = np.concatenate(S); TCP = np.concatenate(TCP)
        G = np.concatenate(G)
        corr = Corrector(hidden=64)
        corr.fit(featurise(T, TCP, G), S[:, :6].astype(np.float32),
                 np.linalg.norm(S[:, :3], axis=1).astype(np.float32),
                 epochs=600, verbose=False)

    if a.robot == "Panda":
        env, suite, task = U.build(a.suite, a.task_id)
    else:
        p, suite, _ = U.build(a.suite, a.task_id); ref = U.fixture_snapshot(p.sim); p.close()
        env, suite, task = U.build(a.suite, a.task_id, robot=a.robot,
                                   gripper=a.gripper, fixture_ref=ref)
    GEOM = U.gripper_geom(env)
    print(f"    gripper geom [flange->TCP, cam->TCP] = "
          f"[{GEOM[0]*1000:.1f}, {GEOM[1]*1000:.1f}] mm")
    if a.align_camera:
        b, af = U.align_wrist_camera(env.sim)
        print(f"    wrist camera realigned: cam-to-TCP {b*1000:.1f} -> {af*1000:.1f} mm")
    o = _t.load; _t.load = lambda *x, **k: o(*x, **{**k, "weights_only": False})
    init = suite.get_task_init_states(a.task_id); _t.load = o
    model = M1Inference(policy_ckpt_path=CK, unnorm_key="franka",
                        policy_setup="franka", host="127.0.0.1", port=a.port)
    succ = 0
    for ep in a.init_states:
        START = None
        if a.align_start:
            f = f"{R}/pairs/traj/traj_{a.suite}_Panda_raw_init{ep}.npy"
            START = np.load(f)[0] if os.path.exists(f) else None
        np.random.seed(0); env.reset()
        obs = env.set_init_state(U.remap_init_state(init[ep], env.sim))
        model.reset(task.language); frames = []; done = False; traj = []
        for _ in range(10):
            obs, _, done, _ = env.step([0.] * (env.env.action_dim - 1) + [-1.])
        if a.align_start and START is not None:
            m, d = env.sim.model, env.sim.data
            site = m.site_name2id(env.env.robots[0].controller.eef_name)
            for _ in range(250):
                dp = START - d.site_xpos[site]
                if np.linalg.norm(dp) < 0.004: break
                obs, _, done, _ = env.step(np.concatenate(
                    [np.clip(dp / 0.05, -1, 1), np.zeros(3), [-1.]]).tolist())
            print(f"    start aligned to {np.round(START,4)}, "
                  f"err {np.linalg.norm(START - d.site_xpos[site])*1000:.1f} mm")
        for t in range(a.steps):
            raw = model.step(
                images=[np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                        np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])],
                task_description=task.language, step=t,
                state=np.expand_dims(np.concatenate(
                    (obs["robot0_eef_pos"], q2aa(obs["robot0_eef_quat"]),
                     grip2(obs["robot0_gripper_qpos"]))), 0))["raw_action"]
            act = from_raw(raw)
            if corr is not None:
                act = corr(act, obs["robot0_eef_pos"], GEOM)[0]
            obs, _, done, _ = env.step(to_env(act).tolist())
            frames.append(np.ascontiguousarray(obs["agentview_image"][::-1]))
            traj.append(np.asarray(obs["robot0_eef_pos"], float).copy())
            if done: break
        succ += int(done)
        # The GRIPPER belongs in the name. Without it a Robotiq85 run silently
        # overwrites the PandaGripper trajectory of the same suite/init, and
        # --align-start then places future runs from a FAILED rollout's start.
        np.save(f"{R}/pairs/traj/traj_{a.suite}_{a.robot}_{a.gripper}_"
                f"{'corr' if corr is not None else 'raw'}_init{ep}.npy",
                np.stack(traj))
        tag = "corr" if corr is not None else "raw"
        imageio.mimsave(f"{R}/videos/eval_{a.suite}_{a.robot}_{a.gripper}{"_cam" if a.align_camera else ""}_{tag}_init{ep}_{'ok' if done else 'fail'}.mp4",
                        frames, fps=20, macro_block_size=1)
        print(f"  {a.robot} {tag} init{ep}: success={bool(done)} steps={len(frames)}")
    print(f"{a.robot} {'corrected' if corr else 'raw'}: {succ}/{len(a.init_states)}")
    env.close()


if __name__ == "__main__":
    main()
