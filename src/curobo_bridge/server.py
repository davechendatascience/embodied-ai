"""Collision-aware IK as a service, in .venv-curobo.

The joint executor in `.venv` needs cuRobo, and cuRobo cannot live in `.venv`
(torch 2.13 vs the pinned LIBERO stack). So it runs here and answers over a
socket. One request carries everything that changes: the start configuration,
the goal pose IN THE ROBOT BASE FRAME, and the CURRENT world.

The world travels with every request on purpose. `world_export` is a snapshot;
the drawer moves as it opens and a held object rides the hand, so a world
uploaded once is stale by the second waypoint.

    scripts/curobo_serve.sh configs/curobo/ur5e_pandagripper.yml
"""

import argparse
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, __file__.rsplit("/src/", 1)[0] + "/src")

from curobo_bridge.protocol import SOCKET_PATH, serve  # noqa: E402


def build(robot_yml, world_dict):
    from curobo.inverse_kinematics import (
        InverseKinematics,
        InverseKinematicsCfg,
    )

    cfg_dict = yaml.safe_load(open(robot_yml))
    kin = cfg_dict["robot_cfg"]["kinematics"]
    # scene_model MUST be a real scene at construction. With None, cuRobo
    # allocates no collision cache and update_world() then dies with
    # `NoneType has no attribute load_collision_...`. Bootstrapping with a
    # world exported from the same task sizes the cache correctly; every
    # subsequent request replaces its contents via update_world().
    cfg = InverseKinematicsCfg.create(
        robot=cfg_dict, num_seeds=32,
        self_collision_check=True, load_collision_spheres=True,
        position_tolerance=0.002, orientation_tolerance=0.05,
        scene_model=world_dict, collision_cache={"obb": 512, "mesh": 8},
    )
    return InverseKinematics(cfg), kin


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("robot_yml")
    ap.add_argument("--socket", default=SOCKET_PATH)
    ap.add_argument("--world", required=True,
                    help="bootstrap world JSON; sizes the collision cache")
    args = ap.parse_args()

    import torch
    from curobo.scene import Cuboid, Scene
    from curobo.types import GoalToolPose

    import json as _json

    bootstrap = _json.load(open(args.world))
    ik, kin_cfg = build(args.robot_yml, bootstrap)
    tool = kin_cfg["tool_frames"][0]
    print(f"curobo-ik: {args.robot_yml} joints={list(ik.kinematics.joint_names)}",
          flush=True)

    def handler(req):
        t0 = time.time()
        try:
            cuboids = [
                Cuboid(name=n, pose=list(s["pose"]), dims=list(s["dims"]))
                for n, s in (req.get("world", {}).get("cuboid") or {}).items()
            ]
            if cuboids:
                ik.update_world(Scene(cuboid=cuboids))
            g = np.asarray(req["goal"], dtype=np.float64)
            pos = torch.as_tensor(g[:3].reshape(1, 1, 1, 1, 3),
                                  dtype=torch.float32, device="cuda").contiguous()
            quat = torch.as_tensor(g[3:].reshape(1, 1, 1, 1, 4),
                                   dtype=torch.float32, device="cuda").contiguous()
            res = ik.solve_pose(GoalToolPose(tool_frames=[tool],
                                             position=pos, quaternion=quat))
            ok = bool(res.success.detach().cpu().numpy().reshape(-1)[0])
            q = res.solution.detach().cpu().numpy().reshape(-1).tolist()
            err = float(res.position_error.detach().cpu().numpy().reshape(-1)[0])
            return {"ok": ok, "q": q, "pos_err": err,
                    "joint_names": list(ik.kinematics.joint_names),
                    "obstacles": len(cuboids), "ms": (time.time() - t0) * 1e3}
        except Exception as exc:  # noqa: BLE001 - the client must see the reason
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "ms": (time.time() - t0) * 1e3}

    serve(args.socket, handler)


if __name__ == "__main__":
    main()
