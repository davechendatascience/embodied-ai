"""How much of the action is predictable from the observed state at all?

A correlation number is only interpretable against a ceiling, and the ceiling is
not 1.0: behaviour cloning targets one demonstration among many valid ones, so
two demonstrations passing through the same state disagree. This measures that
disagreement with a nearest-neighbour predictor handed the privileged tool state
instead of pixels, on the same split and the same metric as training.

Read it as a LOWER bound on the ceiling, not the ceiling. Two reasons it
understates: tool pose does not identify the scene, so demonstrations that
legitimately differ (different object layouts) are counted as disagreement; and
k-NN in 10-D on ~5k frames per task is a weak estimator of the conditional mean.
A trained policy beating these numbers therefore does not show it is at ceiling.

What it does support: k=1 is one demonstration predicting another at matched
tool pose, and where that is low the target is genuinely ambiguous at the level
of state our policy's non-visual branch sees. Larger k approaches the
conditional mean, which is what an MSE objective converges to; past the peak,
averaging smooths across genuinely different states and the estimate decays.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screwhead.interface import ActionSpec        # noqa: E402
from screwhead.libero import panda_chain          # noqa: E402
from screwhead.state import tool_state            # noqa: E402

LAB = ["wx", "wy", "wz", "vx", "vy", "vz", "grip"]


def load(cache: Path, chunk: int):
    spec, chain = ActionSpec(), panda_chain()
    sc = np.array([spec.rot_scale * spec.control_hz] * 3 +
                  [spec.pos_scale * spec.control_hz] * 3, np.float32)
    tasks = {}
    for meta in sorted(cache.glob("task*_meta.npz")):
        d = np.load(meta, allow_pickle=True)
        ti, dem = int(d["task_index"]), np.asarray(d["demo"])
        act = np.concatenate([np.asarray(d["twist"]) / sc,
                              np.asarray(d["gripper"])[:, None]], -1).astype(np.float32)
        st = tool_state(chain, torch.tensor(np.asarray(d["qpos"]), dtype=torch.float64),
                        torch.tensor(np.asarray(d["gripper_state"]),
                                     dtype=torch.float64)).float().numpy()
        idx = [i for i in range(len(dem) - chunk) if dem[i] == dem[i + chunk - 1]]
        tasks[ti] = dict(act=act, state=st, demo=dem, idx=np.array(idx))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/libero_spatial_img")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--val-demos", type=int, default=5)
    ap.add_argument("--val-cap", type=int, default=600)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--pool-tasks", action="store_true",
                    help="search neighbours across all tasks instead of within the "
                         "task. The difference is what task identity is worth, which "
                         "is what the language branch supplies -- one fixed "
                         "instruction per task is a 1-of-N label, not a description.")
    ap.add_argument("--phase", action="store_true",
                    help="add normalised timestep to the observation")
    args = ap.parse_args()

    tasks = load(Path(args.cache), args.chunk)
    print(f"{len(tasks)} tasks, {sum(len(t['idx']) for t in tasks.values())} windows")

    # the split training uses: lowest-numbered demos per task are held out
    tr, va = [], []
    for ti, t in tasks.items():
        val_ids = set(sorted(np.unique(t["demo"]).tolist())[: args.val_demos])
        for i in t["idx"]:
            (va if t["demo"][i] in val_ids else tr).append((ti, int(i)))
    print(f"train {len(tr)} val {len(va)}")

    # observation the oracle is given, standardised so no channel dominates distance
    def obs(ti, ii):
        t = tasks[ti]
        o = t["state"][ii]
        if args.phase:
            d = t["demo"][ii]
            w = np.flatnonzero(t["demo"] == d)
            o = np.concatenate([o, [(ii - w[0]) / max(len(w) - 1, 1)]])
        return o

    allst = np.stack([obs(ti, i) for ti, i in tr])
    mu, sd = allst.mean(0), allst.std(0).clip(1e-6)

    by_task = {}
    for ti in tasks:
        sel = [(t, i) for t, i in tr if t == ti]
        by_task[ti] = (np.array([i for _, i in sel]),
                       np.stack([(obs(ti, i) - mu) / sd for _, i in sel]))
    if args.pool_tasks:
        # one pool over every task; the action still comes from the neighbour's
        # own task, so the predictor simply cannot tell which task it is in
        allidx = np.concatenate([np.stack([np.full(len(by_task[t][0]), t),
                                           by_task[t][0]], 1) for t in sorted(tasks)])
        allobs = np.concatenate([by_task[t][1] for t in sorted(tasks)])

    # va is in task order; sample it so a cap covers the suite, not tasks 0-1
    np.random.default_rng(0).shuffle(va)
    val = va if args.val_cap <= 0 else va[: args.val_cap]
    Y = np.stack([tasks[ti]["act"][i] for ti, i in val])
    out = {}
    for k in args.ks:
        P = []
        for ti, i in val:
            q = (obs(ti, i) - mu) / sd
            if args.pool_tasks:
                d = np.linalg.norm(allobs - q, axis=1)
                nn = allidx[np.argpartition(d, k)[:k]]
                P.append(np.stack([tasks[int(t)]["act"][int(j)] for t, j in nn]).mean(0))
            else:
                tidx, tst = by_task[ti]
                d = np.linalg.norm(tst - q, axis=1)
                nn = tidx[np.argpartition(d, k)[:k]]
                P.append(tasks[ti]["act"][nn].mean(0))
        P = np.stack(P)
        c = [float(np.corrcoef(P[:, j], Y[:, j])[0, 1]) for j in range(7)]
        out[k] = c
        print(f"k={k:<3} " + "  ".join(f"{l}:{v:.2f}" for l, v in zip(LAB, c)), flush=True)

    se = [(1 - c ** 2) / np.sqrt(len(val)) for c in out[args.ks[-1]]]
    print("approx s.e. at n=%d: " % len(val) +
          "  ".join(f"{l}:{v:.03f}" for l, v in zip(LAB, se)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
