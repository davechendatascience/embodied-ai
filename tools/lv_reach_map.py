#!/usr/bin/env python
"""Compute LIBERO-Variations' reach map for a robot over the kitchen table (BRN-lv-action-space), or the shape
catalog of a benchmark's pool, and store it; the benchmark's metadata declares each by digest.

    PYTHONPATH=third_party/LIBERO:. MUJOCO_GL=egl .venv-libero/bin/python tools/lv_reach_map.py \
        --out screwhead/variations/benchmarks/reach_panda_kitchen_table.json
    ... tools/lv_reach_map.py --catalog screwhead/variations/benchmarks/v0.yaml \
        --out screwhead/variations/benchmarks/catalog_v0.json

The reference scene is the table and the robot, with one object parked at the table's far corner (LIBERO's
task-file format needs a goal, and a goal needs an object; free objects play no part in the map).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    from screwhead.sim.sim_arm import Execution
    from screwhead.sim.task_env import TaskEnv
    from screwhead.variations import reach_map, task_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default="Panda")
    ap.add_argument("--catalog", help="a benchmark's metadata: measure its pool's shapes instead of the map")
    ap.add_argument("--families", help="a families file: compute the map of --family's --variant")
    ap.add_argument("--family", default="")
    ap.add_argument("--variant", default="closed")
    args = ap.parse_args()
    if args.catalog:
        _catalog(args)
        return
    workspace, fixtures, park_xy, joints = task_file.WORKSPACE, (), (0.40, 0.50), {}
    if args.families:
        import yaml
        with open(args.families) as f:
            fam = yaml.safe_load(f)["families"][args.family]
        workspace, park_xy, joints = fam["workspace"], tuple(fam["park"]), fam["variants"][args.variant]
        fixtures = tuple(task_file.Placement(task_file.instance_name(x["category"], 1), x["category"], tuple(x["xy"]),
                                             FIXTURE_HALF, float(x["yaw"])) for x in fam["fixtures"])
    park = task_file.Placement("butter_1", "butter", park_xy, 0.001, 0.0)
    text = task_file.write("pick up the butter", [park], [("On", "butter_1", f"{workspace}_{park.region}")],
                           fixtures=fixtures, workspace=workspace)
    path = os.path.join(tempfile.mkdtemp(), "reference.bddl")
    with open(path, "w") as f:
        f.write(text)
    env = TaskEnv("variations", path, render=False, seed=0, execution=Execution(robot=args.robot, hard_reset=True))
    pins = _pins(env, joints) if fixtures else {}
    t0 = time.time()
    rm = reach_map.compute(env, reach_map.table_origin(env.scene, f"{workspace}_{park.region}", park.xy))
    if pins:
        rm.reference["pins"] = {k: [list(map(float, p)), list(map(float, q)), j] for k, (p, q, j) in pins.items()}
    with open(args.out, "w") as f:
        json.dump(rm.as_dict(), f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{args.out}: {int(rm.inside.sum())} points in {time.time() - t0:.0f} s, "
          f"digest {reach_map.ReachMap.load(args.out).digest()}")
    env.close()


FIXTURE_HALF = 1e-4      # m: a fixture's region, as small as MuJoCo takes; the family's pins then fix the pose


def _pins(env, joints: dict) -> dict:
    """The reference's fixtures pinned where the reset put them, their joints at the variant's positions, and
    written back: every scene of the family writes these same values after its reset (TaskEnv.draw_scene)."""
    from screwhead.sim.task_env import pin_fixtures
    from screwhead.sim.task_env_place import drawn_fixtures
    m, d = env.env.sim.model._model, env.env.sim.data._data
    pins = {}
    for b in drawn_fixtures(m):
        name = m.body(b).name
        mine = {j: q for j, q in joints.items() if j.startswith(name.removesuffix("_main"))}
        pins[name] = (m.body_pos[b].round(9).copy(), m.body_quat[b].round(9).copy(), mine)
    pin_fixtures(m, d, pins)
    env.env.sim.forward()
    return pins


def _catalog(args) -> None:
    import yaml

    from screwhead.variations import generator
    with open(args.catalog) as f:
        meta = yaml.safe_load(f)
    pool = list(dict.fromkeys(meta["pool"]["pick"] + meta["pool"]["targets"]))
    raw = generator.measure_catalog(pool, set(meta["pool"]["containers"]), tempfile.mkdtemp(), robot=args.robot)
    with open(args.out, "w") as f:
        json.dump(raw, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{args.out}: {len(raw)} categories, digest {generator.catalog_digest(raw)}")


if __name__ == "__main__":
    main()
