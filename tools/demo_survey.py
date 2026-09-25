#!/usr/bin/env python
"""What LIBERO's human demonstrations do, task by task: the ground truth the skill teacher is
measured against.

  demo_survey.py --suite libero_goal [--tasks 9] [--demos 50] --out runs/demo_survey

Replays each demo's recorded simulator states (nothing is stepped) and reads, per demo:
  grasps    every close..open of the jaws: what the fingers hold (an object, or a handle of an
            articulated fixture), the object's tilt from upright when taken and when let go, the
            tool's approach axis (world) at both, where on the object the tool point is (object
            frame), the aperture once closed, the height the object is let go above where it ends
  drives    every articulated joint that moved, from/to, and whether the jaws held its handle
  unheld    objects that moved more than UNHELD_MOVE while no grasp held them (pushed or knocked)
  outcome   every goal conjunct by LIBERO's own predicate on the final state, and the moved
            object's final pose in its target region's frame
  timing    length, first close, last open

Fixtures are not in the state vector: LIBERO re-samples their poses at reset and each demo's
model XML records the ones it ran with. Replayed without them, 37 of 50 final states of
libero_goal 9 failed LIBERO's own predicate; with them, 50 of 50 held. So every demo's fixture
poses are written into the model before its states are read.

Writes <out>/<suite>.json (per demo) and prints a per-task summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = ROOT / "third_party/LIBERO/libero/datasets"
UNHELD_MOVE = 0.02        # m: an object moved this far with no grasp holding it was pushed or knocked
JOINT_MOVE = {2: 0.02, 3: 0.1}   # slide (m), hinge (rad): a joint that moved this far was driven
SETTLE = 8                # steps after the close command at which the jaws are read as closed
TOP_DOWN_COS = 0.9        # an approach within about 25 deg of straight down counts as top-down
SITE = "gripper0_grip_site"


def _quat_tilt(q: np.ndarray) -> float:
    """Degrees between a body's z axis and world up, from its (w, x, y, z) quaternion."""
    w, x, y, z = q
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1.0, 1.0))))


class Replay:
    def __init__(self, suite: str, task: int):
        from screwhead.sim.task_env import StartNoise, TaskEnv
        from screwhead.sim import contacts
        self.contacts = contacts
        self.env = TaskEnv(suite, task, horizon=600, seed=1, render=False, start=StartNoise())
        self.env.reset()
        e = self.env
        self.sim = e.env.env.sim if hasattr(e.env, "env") else e.env.sim
        self.M, self.D = e.scene.raw()
        self.m, self.d = e.scene.m, e.scene.d
        self.sc, self.spec = e.scene, e.task_spec
        self.site = self.M.site(SITE).id
        self.objects = list(self.spec.objects)
        self.free = {}                       # object -> qpos address of its free joint
        for o in self.objects:
            b = self.sc.body_id(o)
            j = [j for j in range(self.M.njnt) if int(self.M.jnt_bodyid[j]) == b and int(self.M.jnt_type[j]) == 0]
            if j:
                self.free[o] = int(self.M.jnt_qposadr[j[0]])
        self.joints = {}                     # articulated fixture joint name -> (type, qpos address)
        for j in range(self.M.njnt):
            t = int(self.M.jnt_type[j])
            name = self.M.joint(j).name
            if t in (2, 3) and not name.startswith(("robot0", "gripper0")):
                self.joints[name] = (t, int(self.M.jnt_qposadr[j]))
        self.static = {}                     # fixture body -> id, for the per-demo poses
        for b in range(self.M.nbody):
            if int(self.M.body_jntnum[b]) == 0 and self.M.body(b).name.endswith("_main"):
                self.static[self.M.body(b).name] = b

    def fixtures_from(self, xml: str) -> None:
        for name, b in self.static.items():
            m = re.search(rf'<body[^>]*name="{re.escape(name)}"[^>]*>', xml)
            if m is None:
                continue
            pos = re.search(r'pos="([^"]+)"', m.group(0))
            quat = re.search(r'quat="([^"]+)"', m.group(0))
            if pos:
                self.M.body_pos[b] = [float(v) for v in pos.group(1).split()]
            if quat:
                self.M.body_quat[b] = [float(v) for v in quat.group(1).split()]

    def at(self, s: np.ndarray) -> None:
        self.sim.set_state_from_flattened(s)
        self.sim.forward()

    def tool(self):
        return self.D.site_xmat[self.site].reshape(3, 3).copy(), self.D.site_xpos[self.site] - self.sc.base

    def held_by_fingers(self) -> tuple[str | None, str | None]:
        """(object held by both finger groups, articulated joint whose body both touch)."""
        for o in self.objects:
            try:
                if len(self.contacts.finger_sides(self.m, self.d, self.sc.body_id(o))) == 2:
                    return o, None
            except ValueError:
                continue
        for name in self.joints:
            b = int(self.M.jnt_bodyid[self.M.joint(name).id])
            if len(self.contacts.finger_sides(self.m, self.d, b)) == 2:
                return None, name
        return None, None


def survey_demo(r: Replay, e) -> dict:
    xml = e.attrs["model_file"]
    r.fixtures_from(xml.decode() if isinstance(xml, bytes) else xml)
    S, A = e["states"][()], e["actions"][()]
    g = e["obs"]["gripper_states"][()]
    ap = g[:, 0] - g[:, 1]
    T = len(S)
    closing = A[:, -1] > 0
    starts = [t for t in range(T) if closing[t] and (t == 0 or not closing[t - 1])]
    ends = [t for t in range(T) if not closing[t] and t > 0 and closing[t - 1]]
    grasps, held_steps = [], np.zeros(T, bool)
    for c in starts:
        o_ = next((t for t in ends if t > c), T - 1)
        # a human closes slowly: SETTLE steps after the command the fingers were not both on
        # the bottle in 3 of 6 rack demos, so the first step of the hold that has both is used
        obj = joint = None
        rd = c
        for rd in range(min(c + SETTLE, o_ - 1), o_, 2) if o_ - 1 > c else [c]:
            r.at(S[rd])
            obj, joint = r.held_by_fingers()
            if obj or joint:
                break
        Rt, pt = r.tool()
        row = dict(close=c, open=o_, aperture_mm=round(1000 * float(ap[rd]), 1),
                   approach=np.round(Rt[:, 2], 2).tolist(), holds=obj or joint or None,
                   kind="object" if obj else "handle" if joint else "nothing")
        if obj:
            held_steps[c:o_] = True
            a = r.free.get(obj)
            if a is not None:
                q0, q1 = S[rd, 1 + a + 3: 1 + a + 7], S[o_, 1 + a + 3: 1 + a + 7]
                row.update(tilt_taken=round(_quat_tilt(q0), 1), tilt_released=round(_quat_tilt(q1), 1))
            Ro, po = r.sc.body_pose(obj)
            row["grasp_in_object_mm"] = np.round(1000 * (Ro.T @ (pt - po)), 1).tolist()
            r.at(S[o_])
            Rt1, _pt1 = r.tool()
            row["approach_released"] = np.round(Rt1[:, 2], 2).tolist()
            if a is not None:
                z_rel = float(S[o_, 1 + a + 2]); z_end = float(S[-1, 1 + a + 2])
                row["let_go_above_end_mm"] = round(1000 * (z_rel - z_end), 1)
        grasps.append(row)
    drives = []
    for name, (t, adr) in r.joints.items():
        q = S[:, 1 + adr]
        if abs(q[-1] - q[0]) > JOINT_MOVE[t]:
            drives.append(dict(joint=name, type="slide" if t == 2 else "hinge",
                               q0=round(float(q[0]), 3), q1=round(float(q[-1]), 3)))
    # net displacement with no grasp holding it, apart: before the first grasp (a push, or a nudge
    # on the way in), and after the last release (settling where it was put -- the rack's bottle
    # slides into its cradle, which a path-length sum over the whole demo read as a push)
    unheld = []
    held_idx = np.flatnonzero(held_steps)
    first = int(held_idx[0]) if held_idx.size else T - 1
    last = int(held_idx[-1]) + 1 if held_idx.size else T - 1
    for o, a in r.free.items():
        p = S[:, 1 + a: 1 + a + 3]
        before = float(np.linalg.norm(p[first] - p[0]))
        after = float(np.linalg.norm(p[-1] - p[min(last, T - 1)]))
        if max(before, after) > UNHELD_MOVE:
            unheld.append(dict(object=o, before_grasp_mm=round(1000 * before, 1),
                               after_release_mm=round(1000 * after, 1)))
    r.at(S[-1])
    outcome = []
    for goal in r.spec.goals:
        ok = bool(r.sc.env._eval_predicate([goal[0].lower(), *goal[1:]]))
        row = dict(goal=list(goal), ok=ok)
        if len(goal) == 3 and goal[1] in r.free:
            try:
                R_reg, c_reg, _half = r.sc.region(goal[2])
                Ro, po = r.sc.body_pose(goal[1])
                row.update(centre_in_region_mm=np.round(1000 * (R_reg.T @ (po - c_reg)), 1).tolist(),
                           axis_in_region=np.round(R_reg.T @ Ro[:, 2], 2).tolist(),
                           tilt_end=round(float(np.degrees(np.arccos(np.clip(Ro[2, 2], -1, 1)))), 1))
            except ValueError:
                pass                                        # an object target, not a site region
        outcome.append(row)
    return dict(T=T, first_close=starts[0] if starts else None, last_open=ends[-1] if ends else None,
                grasps=grasps, drives=drives, unheld=unheld, outcome=outcome)


def summarise(task: int, name: str, demos: list[dict]) -> list[str]:
    out = [f"task {task}: {name}"]
    T = [d["T"] for d in demos]
    out.append(f"  length median {int(np.median(T))} (p10 {int(np.percentile(T, 10))}, p90 {int(np.percentile(T, 90))});"
               f" goal met at the last state {sum(all(o['ok'] for o in d['outcome']) for d in demos)}/{len(demos)}")
    kinds = collections.Counter(tuple(g["kind"] for g in d["grasps"]) for d in demos)
    out.append(f"  grasp sequences: {dict(kinds.most_common(3))}")
    obj_g = [g for d in demos for g in d["grasps"] if g["kind"] == "object"]
    if obj_g:
        app = np.array([g["approach"] for g in obj_g]); app_r = np.array([g.get("approach_released", g["approach"]) for g in obj_g])
        tt = [g.get("tilt_taken") for g in obj_g if g.get("tilt_taken") is not None]
        tr = [g.get("tilt_released") for g in obj_g if g.get("tilt_released") is not None]
        gio = np.array([g["grasp_in_object_mm"] for g in obj_g])
        lg = [g.get("let_go_above_end_mm") for g in obj_g if g.get("let_go_above_end_mm") is not None]
        top = float(np.mean(app[:, 2] < -TOP_DOWN_COS))
        out.append(f"  object grasps {len(obj_g)}: approach mean {np.round(app.mean(0), 2).tolist()} "
                   f"(top-down in {top:.0%}), at release {np.round(app_r.mean(0), 2).tolist()}; "
                   f"aperture median {np.median([g['aperture_mm'] for g in obj_g]):.1f} mm")
        out.append(f"    tilt taken/released median {np.median(tt):.1f}/{np.median(tr):.1f} deg; "
                   f"grasp point in object frame median {np.round(np.median(gio, 0), 1).tolist()} mm; "
                   f"let go {np.median(lg):.0f} mm above where it ended")
    hg = [g for d in demos for g in d["grasps"] if g["kind"] == "handle"]
    if hg:
        out.append(f"  handle grasps {len(hg)}: approach mean {np.round(np.array([g['approach'] for g in hg]).mean(0), 2).tolist()}")
    dr = collections.Counter(f"{x['joint']} {x['type']} {x['q0']:+.2f}->{x['q1']:+.2f}" for d in demos for x in d["drives"])
    if dr:
        js = collections.Counter(x["joint"] for d in demos for x in d["drives"])
        out.append(f"  drives: {dict(js)}")
    for o in sorted({x["object"] for d in demos for x in d["unheld"]}):
        rows = [x for d in demos for x in d["unheld"] if x["object"] == o]
        pre = [x["before_grasp_mm"] for x in rows if x["before_grasp_mm"] > 1000 * UNHELD_MOVE]
        post = [x["after_release_mm"] for x in rows if x["after_release_mm"] > 1000 * UNHELD_MOVE]
        out.append(f"  {o} moved unheld: before any grasp in {len(pre)}/{len(demos)} demos"
                   + (f" (median {np.median(pre):.0f} mm)" if pre else "")
                   + f", after the last release in {len(post)}/{len(demos)}"
                   + (f" (median {np.median(post):.0f} mm)" if post else ""))
    for i, goal in enumerate(demos[0]["outcome"]):
        rows = [d["outcome"][i] for d in demos]
        if "axis_in_region" in rows[0]:
            ax = np.array([x["axis_in_region"] for x in rows]); ce = np.array([x["centre_in_region_mm"] for x in rows])
            out.append(f"  {goal['goal']}: final axis in region frame {np.round(ax.mean(0), 2).tolist()} "
                       f"(sd {np.round(ax.std(0), 2).tolist()}), centre {np.round(ce.mean(0), 0).tolist()} mm, "
                       f"tilt {np.median([x['tilt_end'] for x in rows]):.0f} deg")
    return out


def main() -> int:
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", type=int, nargs="*", default=None)
    ap.add_argument("--demos", type=int, default=50)
    ap.add_argument("--out", default="runs/demo_survey")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    result = {}
    for task in args.tasks if args.tasks is not None else range(10):
        r = Replay(args.suite, task)
        path = DATASETS / args.suite / f"{r.spec.name}_demo.hdf5"
        if not path.exists():
            print(f"task {task}: {r.spec.name}: no demo file at {path}")
            continue
        with h5py.File(path) as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[: args.demos]
            demos = [dict(demo=k, **survey_demo(r, f["data"][k])) for k in keys]
        result[task] = dict(name=r.spec.name, goals=[list(g) for g in r.spec.goals], demos=demos)
        print("\n".join(summarise(task, r.spec.name, demos)), flush=True)
        r.env.close()
    out = Path(args.out) / f"{args.suite}.json"
    out.write_text(json.dumps(result))
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
