"""Why does the UR5e succeed with a PandaGripper and fail with its native Robotiq85?

Start alignment fixed the arm; the gripper is the last unaligned thing. This logs
the two rollouts side by side so the failure mode is measured, not guessed:

  tcp        where the grasp site actually went
  d_obj      TCP-to-target distance -- does it even arrive?
  cmd        commanded gripper channel (+1 = close)
  width      realised jaw opening -- does the command take effect?

Run: MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python examples/diag_gripper.py
"""
import os, sys, json
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/VLA_JEPA"); sys.path.insert(0, R + "/third_party/LIBERO")
import libero_ur5e as U
from xembody.pairs import from_raw, to_env
from eval_corrected import q2aa, grip2

CK = R + "/third_party/VLA_JEPA/checkpoints_hf/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
SUITE, TASK, INIT, STEPS = "libero_spatial", 0, 0, 300


def jaw_width(obs):
    q = np.asarray(obs["robot0_gripper_qpos"], float).ravel()
    return float(np.abs(q).max()) * (2.0 if q.size == 2 else 1.0)


def target_xpos(env, task_lang):
    """Body whose name best matches the object named in the instruction."""
    m = env.sim.model
    names = [m.body_id2name(i) for i in range(m.nbody)]
    toks = [w for w in task_lang.lower().replace("_", " ").split() if len(w) > 3]
    best, score = None, 0
    for n in names:
        if n is None: continue
        s = sum(t in n.lower() for t in toks)
        if s > score: best, score = n, s
    return best


def run(gripper, model, ref, cam=False, corr=None):
    env, suite, task = U.build(SUITE, TASK, robot="UR5e", gripper=gripper, fixture_ref=ref)
    o = __import__("torch").load
    import torch as _t
    _t.load = lambda *x, **k: o(*x, **{**k, "weights_only": False})
    init = suite.get_task_init_states(TASK); _t.load = o
    np.random.seed(0); env.reset()
    obs = env.set_init_state(U.remap_init_state(init[INIT], env.sim))
    model.reset(task.language)
    for _ in range(10):
        obs, _, done, _ = env.step([0.] * (env.env.action_dim - 1) + [-1.])

    START = np.load(f"{R}/pairs/traj/traj_{SUITE}_Panda_raw_init{INIT}.npy")[0]
    m, d = env.sim.model, env.sim.data
    site = m.site_name2id(env.env.robots[0].controller.eef_name)
    for _ in range(250):
        dp = START - d.site_xpos[site]
        if np.linalg.norm(dp) < 0.004: break
        obs, _, done, _ = env.step(np.concatenate(
            [np.clip(dp / 0.05, -1, 1), np.zeros(3), [-1.]]).tolist())

    if cam:
        U.align_wrist_camera(env.sim)
    tgt = target_xpos(env, task.language)
    tid = m.body_name2id(tgt)
    log, done = [], False
    for t in range(STEPS):
        raw = model.step(
            images=[np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
                    np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])],
            task_description=task.language, step=t,
            state=np.expand_dims(np.concatenate(
                (obs["robot0_eef_pos"], q2aa(obs["robot0_eef_quat"]),
                 grip2(obs["robot0_gripper_qpos"]))), 0))["raw_action"]
        act = from_raw(raw)
        if corr is not None:
            act = corr(act, obs["robot0_eef_pos"], U.gripper_geom(env))[0]
        obs, _, done, _ = env.step(to_env(act).tolist())
        tcp = np.asarray(d.site_xpos[site], float).copy()
        log.append(dict(t=t, tcp=tcp.tolist(),
                        d_obj=float(np.linalg.norm(tcp - d.body_xpos[tid])),
                        obj_z=float(d.body_xpos[tid][2]),
                        cmd=float(act[6]), width=jaw_width(obs)))
        if done: break
    env.close()
    return dict(gripper=gripper, target=tgt, lang=task.language,
                success=bool(done), steps=len(log), log=log)


def main():
    from examples.LIBERO.model2libero_interface import M1Inference
    p, _, _ = U.build(SUITE, TASK); ref = U.fixture_snapshot(p.sim); p.close()
    model = M1Inference(policy_ckpt_path=CK, unnorm_key="franka",
                        policy_setup="franka", host="127.0.0.1", port=15090)
    import glob
    from xembody.adapt import Corrector, featurise
    from xembody.pairs import WV
    T, S, TCP, G = [], [], [], []
    for f in sorted(glob.glob(f"{R}/pairs/*/[pr]_*.npz")):
        d = np.load(f); T.append(d["a_target"]); S.append(d["a_source"]); TCP.append(d["tcp"])
        G.append(d["geom"] if "geom" in d.files
                 else np.full((len(d["tcp"]), 2), np.nan, np.float32))
    T = np.concatenate(T); S = np.concatenate(S)
    TCP = np.concatenate(TCP); G = np.concatenate(G)
    c = Corrector(hidden=64)
    c.fit(featurise(T, TCP, G), S[:, :6].astype(np.float32),
          np.linalg.norm(S[:, WV], axis=1).astype(np.float32), epochs=600, verbose=False)

    out = [run("PandaGripper", model, ref),
           run("Robotiq85Gripper", model, ref),
           run("Robotiq85Gripper", model, ref, cam=True),
           run("Robotiq85Gripper", model, ref, cam=True, corr=c)]
    out[2]["gripper"] = "Robotiq85+cam"
    out[3]["gripper"] = "Robotiq85+cam+corr"
    json.dump(out, open(f"{R}/pairs/diag/diag_gripper.json", "w"))

    print(f"\ntask: {out[0]['lang']}   target body: {out[0]['target']}\n")
    print(f"{'gripper':<20}{'ok':<6}{'steps':<7}{'min d_obj':<11}"
          f"{'t@min':<7}{'obj lift':<10}{'width lo->hi':<14}{'first close'}")
    for r in out:
        L = r["log"]
        d = np.array([x["d_obj"] for x in L]); w = np.array([x["width"] for x in L])
        z = np.array([x["obj_z"] for x in L]); c = np.array([x["cmd"] for x in L])
        close = np.where(c > 0)[0]
        print(f"{r['gripper']:<20}{str(r['success']):<6}{r['steps']:<7}"
              f"{d.min()*1000:>7.1f} mm  {int(d.argmin()):<7}"
              f"{(z.max()-z[0])*1000:>6.1f} mm  "
              f"{w.min()*1000:>4.0f}->{w.max()*1000:<7.0f}"
              f"{(str(int(close[0])) if close.size else 'never')}")


if __name__ == "__main__":
    main()
