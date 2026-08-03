"""Paired (target-arm, source-arm) actions with free labels.

Put both arms at the SAME TCP pose in the same scene, ask the policy for an
action from each, and the SOURCE arm's action is the supervision target for the
TARGET arm's observation. No human labelling, no reward, no real robot.

    sample:  (obs_target, a_target)  ->  label: a_source     at the same TCP

WHAT A CORRECTOR HAS TO LEARN, measured across 8 trajectory phases:
  * direction is already preserved   cos 0.989 median, 0.961 worst
  * magnitude is right on median     ratio 0.95, but noisy 0.80-2.19
  * proprioception is IGNORED        state swap changes the output by cos 1.000
So it is a state-dependent gain-and-nudge, not a re-learning of the task, and
the arm's own joint state need not be an input.

THE DISTRIBUTION-SHIFT CAVEAT. Pairs come from MATCHED poses. At deployment the
target arm is wherever it drifted to, which the pairing cannot cover. This is a
DAgger problem: collect -> train -> roll out -> collect from the states the
corrected policy actually visits -> retrain. One round of pairs is round one,
not the answer.

ACTION SPACE. Everything here stays in the policy's own action space so a
corrected action is directly executable AND directly comparable to what the
policy would have emitted:

    a = [world_vector(3), rotation_delta(3), open_gripper(1)]

`open_gripper` is the model's own convention: [0, 1] with 1 = OPEN. The
simulator wants +1 = CLOSE, so `to_env` applies the inversion -- copied from
the reference implementation, not restated, because paraphrasing it as "close
when > 0.5" inverted the gripper and cost a 0/2 evaluation that looked exactly
like a broken checkpoint.
"""

import numpy as np

ACTION_DIM = 7
WV, ROT, GRIP = slice(0, 3), slice(3, 6), 6


def from_raw(raw):
    """M1Inference `raw_action` dict -> a (7,) vector in the policy's space."""
    return np.concatenate([
        np.asarray(raw["world_vector"], np.float32).reshape(-1),
        np.asarray(raw["rotation_delta"], np.float32).reshape(-1),
        np.asarray(raw["open_gripper"], np.float32).reshape(-1)[:1]])


def to_raw(a):
    """Inverse of `from_raw` -- what the policy would have said."""
    a = np.asarray(a, np.float32).reshape(-1)
    return {"world_vector": a[WV], "rotation_delta": a[ROT],
            "open_gripper": a[GRIP:GRIP + 1]}


def to_env(a):
    """Policy action -> simulator action. The gripper channel INVERTS here."""
    a = np.asarray(a, np.float32).reshape(-1)
    grip = 1.0 - 2.0 * (float(a[GRIP]) > 0.5)      # verbatim, do not restate
    return np.concatenate([a[WV], a[ROT], [grip]])


def check_roundtrip(a, tol=1e-6):
    """`to_raw` then `from_raw` must be the identity, or the corrector's output
    is not something the policy would ever have produced."""
    b = from_raw(to_raw(a))
    return bool(np.max(np.abs(np.asarray(a, np.float32) - b)) <= tol)


class PairSet:
    """Accumulate and persist pairs. Plain arrays; no torch at collection time."""

    def __init__(self):
        self.tcp, self.a_target, self.a_source, self.meta = [], [], [], []

    def add(self, tcp, a_target, a_source, **meta):
        self.tcp.append(np.asarray(tcp, np.float32))
        self.a_target.append(np.asarray(a_target, np.float32))
        self.a_source.append(np.asarray(a_source, np.float32))
        self.meta.append(meta)

    def __len__(self):
        return len(self.tcp)

    def save(self, path):
        # `geom` is the TARGET gripper's [tcp_len, cam_to_tcp] in metres. It is
        # stored per-pair, not per-file, so one corrector can be trained across
        # several grippers -- a constant column carries no gradient, and the
        # whole point is for the model to see the depth difference vary.
        np.savez(path, tcp=np.stack(self.tcp),
                 a_target=np.stack(self.a_target),
                 a_source=np.stack(self.a_source),
                 geom=np.array([m.get("geom", (np.nan, np.nan))
                                for m in self.meta], np.float32),
                 gap_mm=np.array([m.get("gap_mm", np.nan) for m in self.meta],
                                 np.float32))

    def stats(self):
        t, s = np.stack(self.a_target), np.stack(self.a_source)
        nt = np.linalg.norm(t[:, WV], axis=1)
        ns = np.linalg.norm(s[:, WV], axis=1)
        ok = (nt > 1e-6) & (ns > 1e-6)
        cos = (t[ok][:, WV] * s[ok][:, WV]).sum(1) / (nt[ok] * ns[ok])
        return {"n": len(t), "cos_median": float(np.median(cos)),
                "cos_min": float(cos.min()),
                "ratio_median": float(np.median(nt[ok] / ns[ok])),
                "ratio_iqr": [float(np.percentile(nt[ok] / ns[ok], 25)),
                              float(np.percentile(nt[ok] / ns[ok], 75))],
                "grip_disagree": int((np.abs(t[:, GRIP] - s[:, GRIP]) > 0.5).sum())}
