"""Collect (UR5e obs -> Panda action) pairs. Two passes, one env at a time.

Two LIBERO envs must never be live together: the second EGL context corrupts
the first. So pass 1 drives the Panda to a list of TCP targets and records its
actions and ACHIEVED poses; pass 2 drives the UR5e to those achieved poses and
records its actions. Pairing is by index.
"""
import argparse, json, os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/VLA_JEPA"); sys.path.insert(0, R + "/third_party/LIBERO")
import libero_ur5e as U
from xembody.pairs import PairSet, from_raw
CK = R + "/third_party/VLA_JEPA/checkpoints_hf/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"


def q2aa(q):
    q = np.asarray(q, float).copy(); q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] ** 2)
    return np.zeros(3) if np.isclose(den, 0) else q[:3] * 2 * np.arccos(q[3]) / den


def drive_and_query(env, model, task, targets, jitter, rng):
    """Move to each target, query the policy, record action + achieved TCP."""
    m, d = env.sim.model, env.sim.data
    site = m.site_name2id(env.env.robots[0].controller.eef_name)
    out = []
    for tgt in targets:
        t = np.asarray(tgt, float) + rng.normal(0, jitter, 3)
        for _ in range(180):
            dp = t - d.site_xpos[site]
            if np.linalg.norm(dp) < 0.008:
                break
            env.step(np.concatenate([np.clip(dp / 0.05, -1, 1), np.zeros(3), [-1.]]).tolist())
        o = env.env._get_observations()
        st = np.concatenate((o["robot0_eef_pos"], q2aa(o["robot0_eef_quat"]),
                             o["robot0_gripper_qpos"]))
        model.reset(task)
        raw = model.step(images=[np.ascontiguousarray(o["agentview_image"][::-1, ::-1]),
                                 np.ascontiguousarray(o["robot0_eye_in_hand_image"][::-1, ::-1])],
                         task_description=task, step=0,
                         state=np.expand_dims(st, 0))["raw_action"]
        out.append((o["robot0_eef_pos"].copy(), from_raw(raw)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--trace", default="/tmp/vla_jepa_obj/libero_object_task0_ep0.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--jitter", type=float, default=0.03)
    ap.add_argument("--port", type=int, default=15090)
    ap.add_argument("--out", default="/tmp/pairs.npz")
    ap.add_argument("--init-state", type=int, default=0)
    a = ap.parse_args()

    steps = [json.loads(l) for l in open(a.trace)]
    steps = [s for s in steps if s.get("record_type") == "env_step"]
    idx = np.linspace(0, len(steps) - 1, a.n).astype(int)
    targets = [np.array(steps[i]["obs_after_step"]["robot0_eef_pos"]) for i in idx]

    from examples.LIBERO.model2libero_interface import M1Inference
    import torch as _t

    def init_states(suite):
        o = _t.load; _t.load = lambda *x, **k: o(*x, **{**k, "weights_only": False})
        s = suite.get_task_init_states(a.task_id); _t.load = o
        return s

    rng = np.random.default_rng(0)
    model = M1Inference(policy_ckpt_path=CK, unnorm_key="franka",
                        policy_setup="franka", host="127.0.0.1", port=a.port)

    p, suite, task = U.build(a.suite, a.task_id)
    init = init_states(suite)
    p.reset(); p.set_init_state(U.remap_init_state(init[a.init_state], p.sim))
    for _ in range(10): p.step([0.] * (p.env.action_dim - 1) + [-1.])
    src = drive_and_query(p, model, task.language, targets, a.jitter, rng)
    ref = U.fixture_snapshot(p.sim); p.close()

    u, _, _ = U.build(a.suite, a.task_id, robot="UR5e", gripper="PandaGripper",
                      fixture_ref=ref)
    u.reset(); u.set_init_state(U.remap_init_state(init[a.init_state], u.sim))
    for _ in range(10): u.step([0.] * (u.env.action_dim - 1) + [-1.])
    # drive the UR5e to the poses the Panda ACHIEVED, not the ones requested
    tgt = drive_and_query(u, model, task.language, [s[0] for s in src], 0.0, rng)
    u.close()

    ps = PairSet()
    for k, ((p_tcp, p_a), (u_tcp, u_a)) in enumerate(zip(src, tgt)):
        ps.add(u_tcp, u_a, p_a,
               gap_mm=float(np.linalg.norm(p_tcp - u_tcp) * 1000),
               phase=float(idx[k]) / max(len(steps) - 1, 1),
               step=int(idx[k]), init=a.init_state, task=a.task_id)
    ps.save(a.out)
    import json as _j
    _j.dump(ps.meta, open(a.out.replace('.npz', '_meta.json'), 'w'))
    st = ps.stats()
    gaps = np.array([m["gap_mm"] for m in ps.meta])
    print(f"pairs {len(ps)} -> {a.out}")
    print(f"  TCP gap median {np.median(gaps):.1f} mm (max {gaps.max():.1f})")
    print(f"  cos median {st['cos_median']:+.3f} (min {st['cos_min']:+.3f})")
    print(f"  |v| ratio median {st['ratio_median']:.2f}  IQR {np.round(st['ratio_iqr'],2).tolist()}")
    print(f"  gripper channel disagreements: {st['grip_disagree']}/{len(ps)}")


if __name__ == "__main__":
    main()
