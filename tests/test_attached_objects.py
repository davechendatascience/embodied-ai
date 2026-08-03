"""Does the world split correctly into obstacles and held objects?

Two failures this prevents, both of which present as "cuRobo returns nothing"
rather than as anything to do with grasping:

  1. A grasped object left in the WORLD is an obstacle rigidly attached to the
     gripper. The planner is asked to avoid something that moves with the hand,
     so every motion after the grasp is infeasible -- permanently, and with no
     diagnostic that points at the object.

  2. A world computed ONCE is stale the moment anything moves. The drawer opens
     while being pulled; a carried object travels. Planning against tick-0
     geometry means avoiding obstacles that are no longer there and colliding
     with ones that are.

The claim worth testing is the second half of the fix: while an object is
genuinely held, its pose in the TOOL frame is CONSTANT. That is what licenses
attaching it to the robot model. If it drifts, the grasp is slipping -- which
is a real event worth seeing, and quite different from a planning failure.

This replays a successful pick-and-place through the E1 keypose path and
watches the split at every keypose.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_attached_objects.py
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))

from rekep_libero import add_rekep_to_path  # noqa: E402

add_rekep_to_path()

from rekep_libero.frames import inv  # noqa: E402
from rekep_libero.world_export import counts, refresh  # noqa: E402

TRACE = "/tmp/vla_jepa_obj/libero_object_task0_ep0.jsonl"
SUITE, TASK_ID = "libero_object", 0

# A held object's tool-frame pose should be rigid. 5 mm allows for the finger
# compliance NOTES.md already documents (closing costs 37.6 mm of FK error
# under a rigidity assumption, so this is not a tight bound by accident).
HELD_DRIFT_TOL_MM = 5.0


def pose_of(attached, name):
    spec = attached.get("cuboid", {}).get(name)
    if spec is None:
        return None
    return np.asarray(spec["pose"][:3], dtype=np.float64)


def main():
    import json

    import test_e1_keypose_replay as E
    from rekep_libero.environment_libero import EpisodeFinished
    from rekep_libero.keypose import keyposes, load_trace

    if not os.path.exists(TRACE):
        print(f"no trace at {TRACE}; run tests/run_vla_jepa_rollout.py first")
        return 1

    steps = load_trace(TRACE)
    marks = keyposes(steps, 2.0, turn_deg=20.0)
    env = E.build_env(SUITE, TASK_ID, 0, 0)

    print(f"{SUITE}/{TASK_ID}: {len(marks)} keyposes\n")
    print(f"  {'keypose':<10}{'held':<26}{'world':>7}{'attached':>10}"
          f"{'tool-frame drift':>18}")

    baseline = {}
    worst_drift, worst_obj = 0.0, ""
    saw_held = False
    counts_seen = []
    agree = []

    for idx, why in marks:
        s = steps[idx]
        target = np.concatenate([s["pos"], s["quat"]])
        try:
            env._move_to_waypoint(target, pos_threshold=0.005,
                                  rot_threshold=2.0, max_steps=60)
            if s["grip_cmd"] > 0:
                env.close_gripper()
            else:
                env.open_gripper()
        except EpisodeFinished:
            pass

        # cross-check the proprioceptive detector against the contact list it
        # replaces: they must agree, or the swap changed behaviour
        from rekep_libero.grasp_detect import report as grasp_report
        w_hold, width, cmd_closed = grasp_report(env)
        c_hold = bool(env._contacting_objects()) and env.is_grasping()
        agree.append((w_hold, c_hold, width, cmd_closed))

        world, att, held, _skipped = refresh(env)
        n_world = sum(counts(world).values())
        n_att = sum(counts(att).values())
        counts_seen.append(n_world)
        if held:
            saw_held = True

        drift_txt = "-"
        for name in att.get("cuboid", {}):
            p = pose_of(att, name)
            if name not in baseline:
                baseline[name] = p
            else:
                d = float(np.linalg.norm(p - baseline[name])) * 1000.0
                if d > worst_drift:
                    worst_drift, worst_obj = d, name
                drift_txt = f"{d:.2f} mm"

        print(f"  {s['step']:<10}{(','.join(held) or '-'):<26}"
              f"{n_world:>7}{n_att:>10}{drift_txt:>18}")

    env.env.close()   # ReKepLiberoEnv wraps the env; it has no close()

    print()
    failures = []
    if not saw_held:
        failures.append("no object was ever detected as held -- the split was "
                        "never exercised, so this test proved nothing")
    if worst_drift > HELD_DRIFT_TOL_MM:
        failures.append(f"held object {worst_obj} drifted {worst_drift:.2f} mm "
                        f"in the tool frame (> {HELD_DRIFT_TOL_MM} mm): it is "
                        f"not rigidly attached, so attaching it to the robot "
                        f"model would be wrong")
    if len(set(counts_seen)) == 1 and saw_held:
        failures.append("the world obstacle count never changed, so the held "
                        "object was never actually removed from it")

    print(f"  worst tool-frame drift while held: {worst_drift:.2f} mm "
          f"({worst_obj or 'n/a'})")
    print(f"  world obstacle count across keyposes: {counts_seen}")
    print(f"  {'cmd':>6}{'width-based':>14}{'contact-list':>14}{'jaw':>12}")
    for w, c, wid, cmd in agree:
        print(f"  {('shut' if cmd else 'open'):>6}{str(w):>14}{str(c):>14}"
              f"{wid * 1000:>9.1f} mm")
    # Only require agreement while the gripper is commanded SHUT. At release
    # the contact list still sees the object brushing the fingers as it drops;
    # width correctly says "not holding", because the command is open. That is
    # the proprioceptive detector being better, not worse.
    mismatch = [i for i, (w, c, _, cmd) in enumerate(agree) if cmd and w != c]
    if mismatch:
        failures.append(f"width vs contact-list disagree while COMMANDED SHUT "
                        f"at keyposes {mismatch} -- not a drop-in replacement")
    print()
    if failures:
        print("FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASSED — held objects leave the world, ride the tool rigidly, and "
          "the world tracks the scene as it changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
