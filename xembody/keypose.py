"""Dense end-effector trace -> sparse SE(3) keyposes. numpy only."""

import numpy as np

DEFAULTS = dict(vel_eps_mm=2.0, min_gap=2, turn_deg=20.0, turn_window=5)


def extract(poses, grip, vel_eps_mm=2.0, min_gap=2, turn_deg=20.0,
            turn_window=5):
    """Indices of `poses` that are keyposes, with the reason for each.

    poses (N, 7) achieved [x y z qx qy qz qw]; grip (N,) command, >0 closing.

    USE ACHIEVED POSES, not integrated action deltas. An OSC controller
    realises ~25% of each commanded delta, so the product of deltas is a pose
    the arm never occupied.

    Three criteria, and the third is not in the standard heuristic:
      gripper  the command changes state -- a grasp or a release
      dwell    motion stops (the EDGE, not every step spent stopped)
      turn     the path corners

    `turn` exists because the gripper criterion fires ZERO times on any task
    performed with a static gripper -- pushing, hooking, wiping. Measured on a
    drawer opened by hooking: dwell alone gave 4 keyposes, all in the last 34
    of 121 steps, leaving the entire reach unrepresented at every threshold
    from 0.5 to 12 mm.
    """
    poses = np.asarray(poses, dtype=np.float64)
    grip = np.asarray(grip, dtype=np.float64).ravel()
    n = len(poses)
    if n == 0:
        return []
    pos = poses[:, :3]

    disp = np.zeros(n)
    disp[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0

    changed = np.zeros(n, bool)
    changed[1:] = np.sign(grip[1:]) != np.sign(grip[:-1])

    stationary = disp < vel_eps_mm
    entering = np.zeros(n, bool)
    entering[1:] = stationary[1:] & ~stationary[:-1]

    turning = np.zeros(n, bool)
    if turn_deg is not None and n > 2 * turn_window:
        w = turn_window
        ang = np.zeros(n)
        for i in range(w, n - w):
            a, b = pos[i] - pos[i - w], pos[i + w] - pos[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6:
                continue
            ang[i] = np.degrees(np.arccos(
                np.clip(np.dot(a / na, b / nb), -1.0, 1.0)))
        for i in range(w + 1, n - w - 1):
            if ang[i] >= turn_deg and ang[i] >= ang[i - 1] and ang[i] > ang[i + 1]:
                turning[i] = True

    marks, last = [], -min_gap - 1
    for i in range(n):
        if changed[i]:
            marks.append((max(i - 1, 0), "gripper"))
            last = i
        elif entering[i] and i - last > min_gap:
            marks.append((i, "dwell"))
            last = i
        elif turning[i] and i - last > min_gap:
            marks.append((i, "turn"))
            last = i
    if not marks or marks[-1][0] != n - 1:
        marks.append((n - 1, "final"))

    seen, out = set(), []
    for i, why in marks:
        if i not in seen:
            seen.add(i)
            out.append((i, why))
    return sorted(out)


def prehensile(grip):
    """Does the policy ever CLOSE the gripper?

    If not, its solution is non-prehensile and keyposes will not carry the
    task -- measured: a hooked-open drawer moved 3.5 mm of 141 with every
    keypose reached within 5 mm. Check this before planning, not after.
    """
    return bool((np.asarray(grip).ravel() > 0).any())


def fidelity(poses, marks):
    """path length / summed keypose chords. ~1.0 means piecewise straight, so
    the chords reconstruct the dense path and the abstraction loses little
    GEOMETRY -- it says nothing about contact."""
    pos = np.asarray(poses, dtype=np.float64)[:, :3]
    path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    idx = [i for i, _ in marks]
    chord = float(np.linalg.norm(pos[idx[0]] - pos[0]))
    chord += sum(float(np.linalg.norm(pos[b] - pos[a]))
                 for a, b in zip(idx[:-1], idx[1:]))
    return path / chord if chord > 0 else float("nan")
