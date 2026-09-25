#!/usr/bin/env python
"""Does another arm start in the Panda's scene? component-belief's TST-other-arm-scene.

  arm_scene_check.py --trials $OUT [--arms UR5e IIWA Kinova3 Jaco] [--suites ...] [--inits 0]

Per task and initial state, resets LIBERO with the Panda and with each other arm (BRN-other-arm-starts-at-the-
panda-tool-pose) and compares the scenes the episodes would begin in: the largest distance of any object from where
the Panda's reset leaves it, and the tool's distance from the pose the Panda's tool was recorded at. One trial per
(arm, task, initial state). Started in robosuite's own pose for its model, the Kinova3 knocked a wine bottle 3 m off
the table in libero_goal 3, and 42 of 160 such resets moved some object more than 5 mm.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
ARMS = ["UR5e", "IIWA", "Kinova3", "Jaco"]
SCENE_TOL_MM = 5.0        # CTR-other-arm-scene-as-panda's bound on any object's offset


def _objects(env):
    import numpy as np

    from screwhead.sim.libero_env import _joint_blocks
    sim = env.env.sim
    m, q = sim.model, np.asarray(sim.data.qpos, float)
    _robot, objs = _joint_blocks(sim)
    name = {int(m.jnt_qposadr[j]): m.joint_id2name(j) for j in range(m.njnt)}
    return {name[adr]: q[adr:adr + 3].copy() for adr, width, _v in objs if width == 7}   # by name: addresses shift with the arm


def _pin(cpus) -> None:
    """One LIBERO process per core: each worker takes a core of its own for its life."""
    os.sched_setaffinity(0, {cpus.get()})


def _task(job: tuple) -> list[dict]:
    suite, task, inits, arms = job
    import numpy as np

    from screwhead.geometry.kin_np import NpChain, fk
    from screwhead.sim.libero_env import panda_tool_pose
    from screwhead.sim.sim_arm import Execution
    from screwhead.sim.task_env import TaskEnv
    horizon = 800 if suite == "libero_10" else 600

    def reset(robot: str, k: int):
        env = TaskEnv(suite, task, horizon=horizon, seed=555 * 100 + task, render=False, execution=Execution(robot=robot))
        try:
            env.reset(init_index=k)
        except RuntimeError as e:                 # no start touching nothing at the recorded tool pose
            env.close()
            return None, str(e)
        return env, None

    rows = []
    for k in inits:
        ref, _ = reset("Panda", k)
        panda = _objects(ref)
        recorded = panda_tool_pose(ref.init_states[k])[:3, 3]
        ref.close()
        for arm in arms:
            env, err = reset(arm, k)
            metrics = {"start_ok": env is not None}
            if env is not None:
                objs = _objects(env)
                metrics["max_object_offset_mm"] = 1000 * max(float(np.linalg.norm(objs[a] - panda[a])) for a in panda)
                q = np.asarray(env.raw["robot0_joint_pos"], float)
                metrics["tool_offset_mm"] = 1000 * float(np.linalg.norm(fk(NpChain.of(env.chain), q[None])[0][:3, 3] - recorded))
                env.close()
            rows.append({"metrics": metrics, "conditions": {"robot": arm, "suite": suite, "task": task, "init": k},
                         "detail": {"error": err} if err else {}})
            print(f"{arm} {suite} {task} init {k}: {json.dumps(metrics)}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--suites", nargs="+", default=SUITES)
    ap.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--inits", nargs="+", type=int, default=[0])
    ap.add_argument("--cpus", default="5,6,7,8,9,15,16,17")
    args = ap.parse_args()
    from teacher_report import teacher_revision
    cpus = [int(c) for c in args.cpus.split(",")]
    jobs = [(s, t, args.inits, args.arms) for s in args.suites for t in args.tasks]
    rev = teacher_revision()
    free = multiprocessing.Manager().Queue()
    for c in cpus:
        free.put(c)
    with ProcessPoolExecutor(len(cpus), initializer=_pin, initargs=(free,)) as pool:
        trials = [r for rows in pool.map(_task, jobs) for r in rows]
    for t in trials:
        t["repro"] = {"teacher_revision": rev, "robot": t["conditions"]["robot"], "task_suite": t["conditions"]["suite"],
                      "task": t["conditions"]["task"], "init": t["conditions"]["init"]}
    Path(args.trials).write_text(json.dumps({"trials": trials}))
    for arm in args.arms:
        rs = [t for t in trials if t["conditions"]["robot"] == arm]
        ok = [t for t in rs if t["metrics"].get("max_object_offset_mm", 1e9) <= SCENE_TOL_MM]
        print(f"{arm}: {len(ok)}/{len(rs)} within {SCENE_TOL_MM:g} mm of the Panda's scene")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
