#!/usr/bin/env python
"""How LIBERO's humans swing a hinged door: the tool's pose in the door's own frame, against the
door's angle, while the door moves.

  door_survey.py --demos 50 --out screwhead/teacher/door_swing.json

Replays the recorded simulator states of every task whose door is driven (nothing is stepped; the
fixture poses each demo ran with are written first, as demo_survey does). At every step the door's
hinge moves, it records the door angle q, the tool point and the tool's axes in the door body's frame,
the jaws' aperture, and on which side of the door's panel the tool point is (front: the side the
handle stands on; behind: the other). The steps are split by the direction the door moves (open or
close) and by that side, binned by q, and each bin keeps the median tool point and the tool frame
nearest the median of its axes (the rotation closest, in the Frobenius sense, to their mean).

In libero_90 35 ('open the microwave') every demo swings the door with the jaws open and nothing held:
first from the front, past the free edge, to about -0.8 rad, then from behind the panel to -1.5 to
-2.1; in libero_90 33 ('close the microwave') from the front, closing. The table is what the skill
teacher's door law reads (skills.py, _swing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# every LIBERO task that drives the microwave's door, and the way its goal drives it: a task's steps count
# only in that direction (a door just closed bounces open by a hair, and those are not openings)
TASKS = {"libero_90": {35: "open", 33: "close"}, "libero_10": {9: "close"}}
BIN = 0.1                  # rad per bin of the door's angle
MIN_SAMPLES = 20           # steps a bin needs to be kept
MOVING = 1e-3              # rad per step: the door is being driven


def _mean_rotation(Rs: np.ndarray) -> np.ndarray:
    U, _s, Vt = np.linalg.svd(Rs.mean(0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def survey(demos: int) -> dict:
    import h5py
    from demo_survey import DATASETS, Replay
    rows = {"open": [], "close": []}
    sources = []
    for suite, tasks in TASKS.items():
        for task, goal in tasks.items():
            r = Replay(suite, task)
            path = DATASETS / suite / f"{r.spec.name}_demo.hdf5"
            names = [n for n in r.joints if n.startswith("microwave")]
            if not path.exists() or not names:
                r.env.close()
                continue
            from screwhead.geometry.kin_np import NpChain, fk
            chain, arm = NpChain.of(r.env.chain), np.asarray(r.env.joint_indexes, int)
            jid = r.M.joint(names[0]).id
            adr, door = int(r.M.jnt_qposadr[jid]), int(r.M.jnt_bodyid[jid])
            # the door's panel: its box geom; its face on the handle's side is its least y in the door frame
            panel = [g for g in range(r.M.ngeom) if int(r.M.geom_bodyid[g]) == door and int(r.M.geom_type[g]) == 6]
            face = float(r.M.geom_pos[panel[0]][1] - r.M.geom_size[panel[0]][1]) if panel else 0.0
            n = 0
            with h5py.File(path) as f:
                keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:demos]
                for k in keys:
                    e = f["data"][k]
                    xml = e.attrs["model_file"]
                    r.fixtures_from(xml.decode() if isinstance(xml, bytes) else xml)
                    S = e["states"][()]
                    g = e["obs"]["gripper_states"][()]
                    qs = []
                    for t in range(len(S)):
                        r.at(S[t])
                        qs.append(float(r.D.qpos[adr]))
                    qs = np.asarray(qs)
                    for t in np.nonzero(np.abs(np.diff(qs)) > MOVING)[0]:
                        r.at(S[t])
                        # the teacher's own tool frame (its chain's forward kinematics, as its snapshot reads
                        # it): the same point as LIBERO's grip site, turned 90 deg about the approach
                        T = fk(chain, np.asarray(r.D.qpos[arm], float)[None])[0]
                        Rt, pt = T[:3, :3], T[:3, 3]
                        Rd = r.D.xmat[door].reshape(3, 3)
                        pd = r.D.xpos[door] - r.sc.base
                        p = Rd.T @ (pt - pd)
                        front = bool(p[1] < face)          # the handle's side of the panel
                        mode = "open" if qs[t + 1] < qs[t] else "close"
                        if mode != goal:
                            continue
                        rows[mode].append(dict(q=qs[t], p=p, R=Rd.T @ Rt, front=front,
                                               aperture=float(g[t, 0] - g[t, 1])))
                    n += 1
            sources.append(dict(suite=suite, task=task, name=r.spec.name, demos=n))
            r.env.close()
    out = dict(sources=sources, bin_rad=BIN, frame="door body: origin at the hinge anchor, x toward the "
               "free edge, y out of the handle's face; the teacher's tool point (m) and tool axes as columns", modes={})
    for mode, rs in rows.items():
        out["modes"][mode] = {}
        for side in ("front", "behind"):
            sel = [x for x in rs if x["front"] == (side == "front")]
            table = []
            for lo in np.arange(-2.2, 0.1, BIN):
                b = [x for x in sel if lo <= x["q"] < lo + BIN]
                if len(b) < MIN_SAMPLES:
                    continue
                table.append(dict(q=round(float(np.median([x["q"] for x in b])), 4),
                                  p=np.median([x["p"] for x in b], axis=0).round(4).tolist(),
                                  R=_mean_rotation(np.stack([x["R"] for x in b])).round(4).tolist(),
                                  aperture_mm=round(1000 * float(np.median([x["aperture"] for x in b])), 1),
                                  n=len(b)))
            out["modes"][mode][side] = table
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=50)
    ap.add_argument("--out", default="screwhead/teacher/door_swing.json")
    args = ap.parse_args()
    table = survey(args.demos)
    Path(args.out).write_text(json.dumps(table, indent=1))
    for mode, sides in table["modes"].items():
        for side, rows in sides.items():
            if rows:
                print(f"{mode:5s} {side:6s} {len(rows):2d} bins, q {rows[-1]['q']:+.2f}..{rows[0]['q']:+.2f}, "
                      f"steps {sum(r['n'] for r in rows)}, aperture {np.median([r['aperture_mm'] for r in rows]):.0f} mm")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
