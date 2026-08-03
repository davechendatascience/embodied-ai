"""Cut a dense GAM rollout into sparse SE(3) keyposes.

GAM predicts 8-step chunks of 7-dim end-effector DELTAS and executes them
open-loop; it has no keypose output and no subgoal head. E1 asks whether that
dense stream can be compressed to a handful of absolute poses without losing
the task. If it cannot, no amount of cuRobo or UR5e work matters, so this runs
on the PANDA first.

Input is GAM's own `--trace-actions` output — `action_traces/<suite>/
task{id}_ep{n}.jsonl`, one JSON object per line — so nothing under
`third_party/GAM` is modified and nothing here imports torch. numpy and stdlib
only, which means it runs in either venv or neither. The `.jsonl` on disk is
the interface, same rule as the PointWorld bridge.

The cut is the standard keyframe heuristic (PerAct's, and K-VIL's before it):
a step is a keypose when the gripper CHANGES STATE, or when the end-effector is
stationary while the gripper does not change. The last step is always one.
Gripper transitions matter more than dwells — a grasp is a keypose whether or
not the arm happened to pause — so they are never merged away.

WHAT THIS THROWS AWAY, stated plainly because it is the thing E1 measures:
the path BETWEEN keyposes. For a drawer pull that path is a straight line and
losing it should cost nothing. For a curved or contact-constrained motion it is
the task. PPI (RSS 2025) motivates its entire design on exactly this point —
keyframe-and-planner methods "struggle to execute curved motions" — so a pass
on `libero_goal/0` is evidence about straight-line tasks and nothing more.
"""

import argparse
import json
import pathlib

import numpy as np


def load_trace(path):
    """The env_step records from one GAM action trace, in execution order."""
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") != "env_step":
                continue
            obs = rec.get("obs_after_step") or {}
            if "robot0_eef_pos" not in obs:
                continue
            steps.append({
                "step": int(rec["executed_step"]),
                "pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                "quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64),
                # env_gripper is the command actually sent: LIBERO's convention
                # is +1 close / -1 open. `robot0_gripper_width` is what the
                # hardware did about it, which lags and is not the intent.
                "grip_cmd": float(rec["env_gripper"]),
                "width": float(obs.get("robot0_gripper_width", np.nan)),
                "success": bool(rec.get("success_after_step", False)),
            })
    steps.sort(key=lambda s: s["step"])
    return steps


def keyposes(steps, vel_eps_mm=2.0, min_gap=2, turn_deg=30.0, turn_window=5):
    """Indices into `steps` that are keyposes, and why each one is.

    vel_eps_mm  end-effector displacement per step below which it counts as
                stationary. 2 mm at LIBERO's 20 Hz is 40 mm/s -- slow enough
                to be a dwell, loose enough to survive OSC jitter.
    min_gap     steps that must separate two dwell/turn keyposes. Gripper
                transitions ignore it; they are never spurious.
    turn_deg    a CORNER in the path is a keypose. Set to None to disable.

    **Why the turn criterion exists**, because it is not in the standard
    heuristic and it is not optional here. PerAct's rule -- gripper transition,
    or stationary-and-gripper-unchanged -- assumes the gripper marks task
    structure. Measured on `libero_goal/0`: the policy opens the drawer with
    the gripper OPEN for all 121 steps, hooking the handle rather than grasping
    it, so the gripper criterion never fires once. Dwell alone then yields FOUR
    keyposes, all of them in the last 34 steps, and the entire 87-step reach is
    represented by nothing at all. Swept across `vel_eps_mm` from 0.5 to 12 it
    never produces more than four.

    A reach-then-pull trajectory has no pause between the two phases, but it
    does have a corner. Segmenting on direction change recovers the structure
    the gripper would have marked in a pick-and-place. Any task done with a
    static gripper -- pushing, hooking, wiping, levering -- needs this.
    """
    if not steps:
        return []

    pos = np.stack([s["pos"] for s in steps])
    grip = np.array([s["grip_cmd"] for s in steps])
    disp = np.zeros(len(steps))
    disp[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0

    # A gripper transition is a keypose AT the step where the command flips,
    # and the pose that matters is the one the arm had when it flipped -- so
    # the transition is attributed to the step BEFORE the change.
    changed = np.zeros(len(steps), dtype=bool)
    changed[1:] = np.sign(grip[1:]) != np.sign(grip[:-1])

    # A dwell is the EDGE where motion stops, not every step spent stopped.
    # Testing `disp[i] < eps` alone emits one keypose every `min_gap` steps for
    # as long as the arm sits still, which turns a single pause into a dozen
    # identical targets and makes the compression ratio meaningless.
    stationary = disp < vel_eps_mm
    entering = np.zeros(len(steps), dtype=bool)
    entering[1:] = stationary[1:] & ~stationary[:-1]

    # Corners. Direction is taken over a window rather than step to step,
    # because a single step's displacement is ~3 mm and its direction is mostly
    # OSC jitter; over `turn_window` steps the intent is legible.
    turning = np.zeros(len(steps), dtype=bool)
    if turn_deg is not None and len(steps) > 2 * turn_window:
        w = turn_window
        ang = np.zeros(len(steps))
        for i in range(w, len(steps) - w):
            a = pos[i] - pos[i - w]
            b = pos[i + w] - pos[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6:
                continue
            ang[i] = np.degrees(np.arccos(
                np.clip(np.dot(a / na, b / nb), -1.0, 1.0)))
        # keep only local maxima above threshold, so one corner yields one
        # keypose instead of a run of them
        for i in range(w + 1, len(steps) - w - 1):
            if ang[i] >= turn_deg and ang[i] >= ang[i - 1] and ang[i] > ang[i + 1]:
                turning[i] = True

    marks = []
    last = -min_gap - 1
    for i in range(len(steps)):
        if changed[i]:
            marks.append((max(i - 1, 0), "gripper"))
            last = i
        elif entering[i] and i - last > min_gap:
            marks.append((i, "dwell"))
            last = i
        elif turning[i] and i - last > min_gap:
            marks.append((i, "turn"))
            last = i
    if not marks or marks[-1][0] != len(steps) - 1:
        marks.append((len(steps) - 1, "final"))

    # collapse duplicates, keeping the first reason given for each index
    seen, out = set(), []
    for idx, why in marks:
        if idx not in seen:
            seen.add(idx)
            out.append((idx, why))
    return sorted(out)


def summarize(steps, marks):
    pos = np.stack([s["pos"] for s in steps])
    path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()) * 1000.0
    chord = 0.0
    for a, b in zip(marks[:-1], marks[1:]):
        chord += float(np.linalg.norm(pos[b[0]] - pos[a[0]])) * 1000.0
    chord += float(np.linalg.norm(pos[marks[0][0]] - pos[0])) * 1000.0
    return {
        "steps": len(steps),
        "keyposes": len(marks),
        "compression": len(steps) / max(len(marks), 1),
        "path_mm": path,
        "keypose_chord_mm": chord,
        # >1 means the dense path wanders relative to the straight lines
        # between keyposes. Near 1.0 the motion IS piecewise straight and a
        # planner loses nothing; well above 1.0 is where E1 should be expected
        # to fail, and is worth knowing BEFORE the replay is run.
        "path_over_chord": path / chord if chord > 0 else float("nan"),
        "succeeded": any(s["success"] for s in steps),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+", help="GAM action_traces/**/task*_ep*.jsonl")
    ap.add_argument("--vel-eps-mm", type=float, default=2.0)
    ap.add_argument("--min-gap", type=int, default=2)
    ap.add_argument("--out", type=str, default=None,
                    help="write extracted keyposes to this .npz")
    args = ap.parse_args()

    bundle, rows = {}, []
    for path in args.traces:
        steps = load_trace(path)
        if not steps:
            print(f"{pathlib.Path(path).name}: no env_step records with obs")
            continue
        marks = keyposes(steps, args.vel_eps_mm, args.min_gap)
        s = summarize(steps, marks)
        rows.append((pathlib.Path(path).name, s))
        print(f"{pathlib.Path(path).name}")
        print(f"  {s['steps']} steps -> {s['keyposes']} keyposes "
              f"({s['compression']:.1f}x), success={s['succeeded']}")
        print(f"  dense path {s['path_mm']:.1f} mm, keypose chords "
              f"{s['keypose_chord_mm']:.1f} mm, ratio {s['path_over_chord']:.2f}")
        for idx, why in marks:
            st = steps[idx]
            print(f"    step {st['step']:>4}  {why:<8} "
                  f"pos {np.round(st['pos'], 4)}  grip {st['grip_cmd']:+.0f}")
        if args.out:
            key = pathlib.Path(path).stem
            bundle[f"{key}/pos"] = np.stack([steps[i]["pos"] for i, _ in marks])
            bundle[f"{key}/quat"] = np.stack([steps[i]["quat"] for i, _ in marks])
            bundle[f"{key}/grip"] = np.array(
                [steps[i]["grip_cmd"] for i, _ in marks])
            bundle[f"{key}/step"] = np.array([steps[i]["step"] for i, _ in marks])
            bundle[f"{key}/why"] = np.array([w for _, w in marks])

    if rows:
        comp = np.mean([s["compression"] for _, s in rows])
        ratio = np.mean([s["path_over_chord"] for _, s in rows])
        print(f"\n{len(rows)} episodes: mean {comp:.1f}x compression, "
              f"mean path/chord {ratio:.2f}")
    if args.out and bundle:
        np.savez(args.out, **bundle)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
