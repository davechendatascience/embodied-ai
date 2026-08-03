"""How long does each gripper take to actually close, and how long does the
policy give it?

The policy emits a close command and then, some number of steps later, starts
lifting. That gap was fixed by the Panda's hardware during training. If a
different gripper needs MORE steps to reach full closure than the policy waits,
the lift begins on a jaw that is still moving -- the object is squeezed out or
never gripped, no matter how well the arm was positioned.

Two measurements, both with the arm held still:

  free      close on nothing. Pure actuator dynamics: steps to 90% of final.
  on_bowl   close on the object at the known-good grasp pose. Steps until the
            jaw stops moving, i.e. until force is established.

Then the budget: in the Panda's own successful rollout, how many steps pass
between the close command and the start of the lift.

Run: MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python examples/diag_close_time.py
"""
import json, os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + "/examples")
sys.path.insert(0, R + "/third_party/LIBERO")
import libero_ur5e as U

SUITE, TASK, INIT = "libero_spatial", 0, 0
BOWL = "akita_black_bowl_1_main"
GRIPPERS = ("PandaGripper", "RethinkGripper", "Robotiq85Gripper")
N = 80


def width(env):
    return float(np.abs(env.env._get_observations()["robot0_gripper_qpos"]).max())


def close_curve(env, n=N):
    """Jaw width per step while commanding close, arm held still."""
    w = [width(env)]
    for _ in range(n):
        env.step([0.] * (env.env.action_dim - 1) + [1.])
        w.append(width(env))
    return np.array(w)


def steps_to_settle(w, frac=0.90):
    """Steps to reach `frac` of total travel, and to stop moving entirely."""
    travel = w[0] - w[-1]
    if abs(travel) < 1e-6:
        return None, None
    hit = np.where(np.abs(w - w[0]) >= frac * abs(travel))[0]
    moving = np.where(np.abs(np.diff(w)) > 1e-4)[0]
    return (int(hit[0]) if hit.size else None,
            int(moving[-1]) + 1 if moving.size else 0)


def drive(env, target, steps=200, grip=-1.0, tol=0.003):
    m, d = env.sim.model, env.sim.data
    site = m.site_name2id(env.env.robots[0].controller.eef_name)
    for _ in range(steps):
        dp = target - d.site_xpos[site]
        if np.linalg.norm(dp) < tol:
            break
        env.step(np.concatenate([np.clip(dp / 0.05, -1, 1), np.zeros(3), [grip]]).tolist())


def policy_budget():
    """From the Panda's successful rollout: close command -> lift onset."""
    f = f"{R}/pairs/diag/diag_gripper.json"
    if not os.path.exists(f):
        return None
    log = next(r for r in json.load(open(f)) if r["gripper"] == "PandaGripper")["log"]
    cmd = np.array([x["cmd"] for x in log])
    z = np.array([x["tcp"][2] for x in log])
    close = np.where(cmd > 0)[0]
    if not close.size:
        return None
    # last sustained close before the object leaves the table
    t0 = int(close[np.argmax(np.diff(np.r_[close, close[-1] + 2]) > 1)]) if close.size else 0
    after = z[t0:]
    rise = np.where(after - after[0] > 0.01)[0]     # 10 mm of lift
    return t0, (int(rise[0]) if rise.size else None)


def main():
    import torch as _t
    traj = np.load(f"{R}/pairs/traj/traj_{SUITE}_Panda_raw_init{INIT}.npy")
    p, suite, _ = U.build(SUITE, TASK)
    o = _t.load; _t.load = lambda *x, **k: o(*x, **{**k, "weights_only": False})
    init = suite.get_task_init_states(TASK); _t.load = o
    np.random.seed(0); p.reset(); p.set_init_state(U.remap_init_state(init[INIT], p.sim))
    for _ in range(10): p.step([0.] * (p.env.action_dim - 1) + [-1.])
    bowl = np.asarray(p.sim.data.body_xpos[p.sim.model.body_name2id(BOWL)], float).copy()
    ref = U.fixture_snapshot(p.sim); p.close()
    grasp = traj[int(np.argmin(np.linalg.norm(traj - bowl, axis=1)))].copy()

    print(f"\n  {'gripper':<20}{'free: 90% @':>14}{'stops @':>10}"
          f"{'on_bowl: 90% @':>17}{'stops @':>10}{'  travel(free)'}")
    for g in GRIPPERS:
        row = {}
        for mode in ("free", "on_bowl"):
            e, _, _ = U.build(SUITE, TASK, robot="UR5e", gripper=g, fixture_ref=ref)
            np.random.seed(0); e.reset()
            e.set_init_state(U.remap_init_state(init[INIT], e.sim))
            for _ in range(10): e.step([0.] * (e.env.action_dim - 1) + [-1.])
            if mode == "on_bowl":
                drive(e, grasp + np.array([0, 0, 0.08]))
                drive(e, grasp)
            row[mode] = close_curve(e)
            e.close()
        f90, fst = steps_to_settle(row["free"])
        b90, bst = steps_to_settle(row["on_bowl"])
        print(f"  {g:<20}{str(f90):>14}{str(fst):>10}{str(b90):>17}{str(bst):>10}"
              f"{(row['free'][0]-row['free'][-1]):>12.3f}")

    b = policy_budget()
    if b:
        print(f"\n  policy budget (Panda rollout): close at step {b[0]}, "
              f"10 mm of lift {b[1]} steps later\n")


if __name__ == "__main__":
    main()
