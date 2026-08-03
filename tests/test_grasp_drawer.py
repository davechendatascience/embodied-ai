"""Does Contact-GraspNet find the drawer handle?

This is the case that separates a grasp network from a heuristic, and the one
articulated manipulation stands or falls on. Two things are being asked, and
they are different:

  1. LOCALISATION -- given the whole cabinet, does the top-scoring grasp land on
     the handle, or somewhere on the large flat drawer front? PCA cannot do this
     at all: the cabinet's narrowest horizontal axis is a statement about the
     whole box, and the handle is a ~26 mm feature on a ~210 mm face.

  2. APPROACH DIRECTION -- a drawer handle must be approached HORIZONTALLY, into
     the cabinet face (-Y here). Every grasp this project has proposed so far has
     been top-down, and `AnalyticGraspProposer` can only ever be top-down: it
     hard-codes `world_approach = [0, 0, -1]`. A top-down grasp on a handle
     closes on the drawer's top surface and pulls nothing.

Ground truth is the handle's own geoms in the MuJoCo model, so "on the handle"
is measured rather than eyeballed.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_grasp_drawer.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import ReKepLiberoEnv, AGENTVIEW  # noqa: E402
from rekep_libero.grasp import AnalyticGraspProposer  # noqa: E402
from rekep_libero import grasp_cgn  # noqa: E402
import transform_utils as T  # noqa: E402

CABINET = "wooden_cabinet_1"
# The handle stands proud of the drawer front. Everything on the drawer body
# sits at y <= -0.163; these four geoms are the only ones in front of it.
HANDLE_MAX_Y = -0.160


def cabinet_geoms(env, drawer=None):
    """Geom ids for the cabinet, optionally just one drawer's body."""
    model = env.sim.model
    ids = []
    for g in range(model.ngeom):
        body = model.body_id2name(model.geom_bodyid[g]) or ""
        if not body.startswith(CABINET):
            continue
        if drawer is not None and not body.endswith(drawer):
            continue
        ids.append(g)
    return ids


def handle_truth(env, drawer="cabinet_middle"):
    """World position of the handle: the geoms standing in front of the face."""
    model, data = env.sim.model, env.sim.data
    pts = [data.geom_xpos[g] for g in cabinet_geoms(env, drawer)
           if data.geom_xpos[g][1] > HANDLE_MAX_Y]
    return np.mean(pts, axis=0), np.asarray(pts)


def mask_from_geoms(env, geom_ids, margin=None):
    """(H, W) pixel mask covering the given geoms, via the env's own predicate.

    This used to reimplement the geom-bounding-sphere test locally. That is
    badly wrong for a cabinet: its panel geoms have `rbound` up to 0.206 m, so
    the sphere union covered the whole cabinet volume plus a great deal of empty
    space and table, and the mask handed to the network was far larger than the
    cabinet itself. The env now tests the exact collision boxes.
    """
    points = env.get_cam_obs()[AGENTVIEW]["points"]
    return env.points_in_geoms(points, geom_ids, margin=margin)


def evaluate(label, env, proposer, mask, truth, off):
    depth, K, cam2world = env.camera_view()
    if mask.sum() < 20:
        print(f"{label:22s} mask too small ({int(mask.sum())} px)")
        return
    cands = proposer._candidates(depth, mask, K, cam2world)
    if not cands:
        print(f"{label:22s} no candidates")
        return

    def describe(c):
        return (f"{float(np.linalg.norm(c['position'] - truth)) * 1000:6.1f} mm, "
                f"approach.-Y {float(c['approach'] @ [0, -1, 0]):+.2f}, "
                f"score {c['score']:.3f}")

    # The two selection rules, on the identical candidate set. `truth` stands in
    # for the grasp keypoint a VLM constraint would name on the handle.
    by_score = proposer.rank(cands)[0]
    by_keypoint = proposer.rank(cands, truth)[0]
    print(f"{label:22s} {len(cands):3d} cands")
    print(f"{'':22s}  by score    {describe(by_score)}")
    print(f"{'':22s}  by keypoint {describe(by_keypoint)}")
    return by_score, by_keypoint


def main():
    if not grasp_cgn.available():
        print("no Contact-GraspNet checkpoint")
        return 1

    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite="libero_goal", task_id=0,
                         robot="Panda", resolution=config["libero"]["resolution"])
    print(f"task   : {env.instruction}")

    truth, handle_pts = handle_truth(env)
    print(f"handle : {np.round(truth, 4)}  ({len(handle_pts)} geoms, "
          f"span {np.round(handle_pts.max(0) - handle_pts.min(0), 3)})")
    print(f"ee home: {np.round(env.get_ee_pos(), 3)}\n")

    off = env.finger_offset()
    proposer = grasp_cgn.ContactGraspNetProposer(finger_offset=off, top_k=64)
    grasp_cgn.estimator()

    print("approach.-Y = +1 means straight into the cabinet face (what a drawer needs);")
    print("0 means top-down (what the analytic proposer can only ever do).\n")

    evaluate("whole cabinet", env, proposer,
             mask_from_geoms(env, cabinet_geoms(env)), truth, off)
    evaluate("middle drawer only", env, proposer,
             mask_from_geoms(env, cabinet_geoms(env, "cabinet_middle")), truth, off)

    # Handing the network ONLY the handle separates two failures that look the
    # same from the outside: "cannot find a 26 mm feature on a 210 mm face"
    # (a localisation problem, fixable with a part detector) versus "cannot
    # grasp a handle even when told exactly where it is" (a model problem,
    # not fixable that way).
    handle_geoms = [g for g in cabinet_geoms(env, "cabinet_middle")
                    if env.sim.data.geom_xpos[g][1] > HANDLE_MAX_Y]
    # Masked exactly, the handle yields too few points for the network to
    # propose anything at all, so the padded variants show how much surrounding
    # drawer front it needs. This is the number a part detector would have to hit.
    for pad in (0.003, 0.02, 0.04):
        evaluate(f"handle +{pad * 1000:.0f}mm (oracle)", env, proposer,
                 mask_from_geoms(env, handle_geoms, margin=pad), truth, off)

    # what the heuristic does with the same geometry
    points = env.get_cam_obs()[AGENTVIEW]["points"].reshape(-1, 3)
    m = mask_from_geoms(env, cabinet_geoms(env, "cabinet_middle")).reshape(-1)
    a = AnalyticGraspProposer(finger_offset=off)
    pos, quat, width = a.propose(points[m], env.GRASP_APPROACH_AXIS,
                                 env.gripper_closing_axis_idx())
    approach = T.quat2mat(quat) @ env.GRASP_APPROACH_AXIS
    print(f"\n{'analytic (PCA)':22s} width {width * 1000:5.1f} mm "
          f"({'fits' if a.fits(width) else 'TOO WIDE'}), "
          f"{float(np.linalg.norm(pos - truth)) * 1000:6.1f} mm from handle, "
          f"approach.-Y {float(approach @ [0, -1, 0]):+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
