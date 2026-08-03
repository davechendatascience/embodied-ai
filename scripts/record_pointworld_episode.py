"""Record MuJoCo episodes in the exact tensor layout PointWorld consumes.

Why this exists
---------------
PointWorld's contract (`pointworld/base.py:434`) is:

    scene_flows   (B, T, Ns, 3)   scene point positions over time
    robot_flows   (B, T, Nr, 3)   robot point positions over time -- THE ACTION
    scene_features(B, T, Ns, Ds)  31 raw channels, assembled by `gather_features`
    robot_features(B, T, Nr, Fr)  16 raw channels, likewise
    *_exists                      per-point validity masks

The raw channels the released checkpoint was trained on are fixed by its own
args, and this file records every input they need:

    robot_features = flows(3) colors(3) normals(3) gripper_open(1)
                     velocity(3) acceleration(3)                    = 16
    scene_features  = flows(3) colors(3) normals(3) gripper_open(T)
                     dist2robot(T)                                  = 31

Velocity and acceleration are finite differences of `robot_flows`, which is why
the gripper points must be the SAME points from step to step. Resampling them
each step -- as the first version of this file did -- makes those six channels
pure sampling noise. `GripperPoints` fixes local offsets once and pushes them
through MuJoCo's geom poses instead.

The action being "where the robot's own points are" is what makes the model
embodiment-agnostic, and it is also what makes our simulator a legitimate source
of evaluation data: MuJoCo gives us forward kinematics for the gripper and, more
importantly, EXACT POINT CORRESPONDENCE over time.

That correspondence is the whole reason this is worth doing. PointWorld predicts
the displacement of each INITIAL scene point, so scoring a prediction requires
knowing where point i actually went. On real data that needs annotation -- the
paper built a ~2M-trajectory dataset with "custom high-precision 3D
annotations". In sim we get it for free: bind each t=0 point to the body it
lies on, then push it through that body's pose at every later step.

Two processes, on purpose
-------------------------
This runs in the ReKep venv (`.venv`: mujoco 3.1.6, numpy 1.26.4) and writes
`.npz`. PointWorld runs in `.venv-pw` (torch 2.11, numpy 2.5). Those stacks
cannot share a process, so disk is the interface. Do not try to merge them.

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python scripts/record_pointworld_episode.py \
        --task-suite libero_goal --task-id 0 --out data/pw_episodes
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import (  # noqa: E402
    ReKepLiberoEnv, AGENTVIEW, WRISTVIEW, EpisodeFinished,
)
# One definition of "the gripper's points", shared with the planner. A second
# copy here would be a second definition of the ACTION (NOTES.md section 2).
from rekep_libero.gripper_points import GripperPoints  # noqa: E402
from rekep_libero.pw_observation import (  # noqa: E402
    NS, NR, T_LEN, GRID, observe, organized_normals, voxel_downsample,
)
import transform_utils as T  # noqa: E402

def bind_points_to_bodies(env, points, normals):
    """Which MuJoCo body each scene point belongs to, so it can be tracked.

    Returns (body_ids, local, local_n): the point and its normal in that body's
    frame. Points on no known body (table, background) get body_id -1 and are
    treated as static -- correct here, since the table does not move.
    """
    model, data = env.sim.model, env.sim.data
    body_ids = np.full(len(points), -1, dtype=np.int64)
    local = points.copy()
    local_n = normals.copy()

    # Reuse the same geometry predicate the grasp pipeline uses, so "which
    # object is this point on" means the same thing everywhere in the project.
    for name in env._object_geom_ids:
        mask = env._object_point_mask(points, name)
        if not mask.any():
            continue
        gids = env._object_geom_ids[name]
        bid = int(model.geom_bodyid[gids[0]])
        pos, quat_wxyz = data.xpos[bid], data.xquat[bid]
        R = T.quat2mat(np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]))
        body_ids[mask] = bid
        local[mask] = (points[mask] - pos) @ R
        local_n[mask] = normals[mask] @ R          # rotation only; normals do not translate
    return body_ids, local, local_n


def body_poses(env, body_ids):
    """Current world pose of each distinct body, as (R, t) lookup."""
    data = env.sim.data
    out = {}
    for bid in np.unique(body_ids):
        if bid < 0:
            continue
        q = data.xquat[bid]
        out[int(bid)] = (T.quat2mat(np.array([q[1], q[2], q[3], q[0]])),
                         np.asarray(data.xpos[bid]).copy())
    return out


def scene_points_now(points0, body_ids, local, poses):
    """Where each initial point is now, given current body poses."""
    out = points0.copy()
    for bid, (R, t) in poses.items():
        m = body_ids == bid
        if m.any():
            out[m] = local[m] @ R.T + t
    return out


def scene_normals_now(normals0, body_ids, local_n, poses):
    """Where each initial normal points now. Rotation only, no translation."""
    out = normals0.copy()
    for bid, (R, _) in poses.items():
        m = body_ids == bid
        if m.any():
            out[m] = local_n[m] @ R.T
    return out


def record(env, steps, action_fn, rng, cam_ids=(AGENTVIEW, WRISTVIEW)):
    """Roll out `steps` and return the PointWorld tensors for one episode."""
    cams, pts, colors, normals, n_robot_px = observe(env, cam_ids)
    # Fusing the views before the 1.5 cm grid subsample is what removes the
    # duplicate surface where the two cameras overlap.
    keep = voxel_downsample(pts, GRID, NS)
    points0, colors0, normals0 = pts[keep], colors[keep], normals[keep]

    body_ids, local, local_n = bind_points_to_bodies(env, points0, normals0)
    n_tracked = int((body_ids >= 0).sum())

    gripper = GripperPoints(env, NR, rng)

    # EpisodeFinished is LIBERO reporting SUCCESS, not a failure: it sets
    # done=True the moment the task is solved and the next step raises. Losing
    # the episode over that would throw away precisely the rollouts where
    # something interesting happened, so record what we have and pad.
    scene_seq, scene_n_seq, robot_seq, robot_n_seq, grip_seq = [], [], [], [], []
    truncated_at = None
    for t in range(steps):
        poses = body_poses(env, body_ids)
        scene_seq.append(scene_points_now(points0, body_ids, local, poses))
        scene_n_seq.append(scene_normals_now(normals0, body_ids, local_n, poses))
        rp, rn = gripper(env)
        robot_seq.append(rp)
        robot_n_seq.append(rn)
        # `last_gripper_action` is -1 open / +1 closed; the feature is "open".
        grip_seq.append([1.0 if env.last_gripper_action == -1.0 else 0.0])
        if t < steps - 1:
            try:
                action_fn(env, t)
            except EpisodeFinished:
                truncated_at = t + 1
                break
    for seq in (scene_seq, scene_n_seq, robot_seq, robot_n_seq, grip_seq):
        while len(seq) < steps:        # hold the final state
            seq.append(np.copy(seq[-1]) if isinstance(seq[-1], np.ndarray) else list(seq[-1]))

    scene = np.stack(scene_seq)[None]          # (1, T, Ns, 3)
    robot = np.stack(robot_seq)[None]          # (1, T, Nr, 3)
    return {
        "scene_flows": scene.astype(np.float32),
        "robot_flows": robot.astype(np.float32),
        "scene_normals": np.stack(scene_n_seq)[None].astype(np.float32),
        "robot_normals": np.stack(robot_n_seq)[None].astype(np.float32),
        "scene_colors": np.repeat(colors0[None], steps, 0)[None].astype(np.uint8),
        "gripper_open": np.array(grip_seq, dtype=np.float32)[None],   # (1, T, 1)
        "scene_exists": np.ones(scene.shape[:3], dtype=bool),
        "robot_exists": np.ones(robot.shape[:3], dtype=bool),
        "point_body_ids": body_ids[None].astype(np.int64),
        "n_tracked": np.array([n_tracked]),
        "n_robot_px": np.array([n_robot_px]),
        "truncated_at": np.array([-1 if truncated_at is None else truncated_at]),
        # Camera axis first after batch: (1, C, ...), matching the release's
        # per-camera `camN_*` keys.
        "rgb": np.stack([c["rgb"] for c in cams])[None].astype(np.uint8),
        "depth": np.stack([c["depth"] for c in cams])[None].astype(np.float32),
        "intrinsic": np.stack([c["intrinsic"] for c in cams])[None].astype(np.float32),
        "extrinsic": np.stack([c["extrinsic"] for c in cams])[None].astype(np.float32),
    }


def push_forward(env, t, step=0.02):
    """A simple probing action: advance the wrist +Y and keep the grip.

    Kept as the trivial baseline, and worth knowing it IS trivial: measured on
    libero_goal/0 it moves the arm through free space, touches nothing, and
    produces `max scene motion 0.0 mm`. A dynamics model scored on that looks
    perfect while demonstrating nothing. Use `--motion drawer` for real data.
    """
    target = env.get_ee_pose().copy()
    target[1] += step
    env.execute_action(np.concatenate([target[:3], target[3:], [env.get_gripper_null_action()]]),
                       precise=False)


def drawer_motion(env, step=0.02):
    """Grasp the middle drawer handle and pull, yielding an action fn.

    Uses the scripted IDEAL grasp -- approach straight -Y into the face, jaws
    closing vertically across the bar -- which was the control experiment that
    proved the mechanism, driving the slide joint 0 -> -0.131 with contact held.
    Deterministic, and it exercises articulation, contact and a moving fixture,
    which is exactly the dynamics worth asking a world model to predict.

    The grasp itself is done BEFORE recording starts, so the recorded window is
    the part with interesting motion rather than the approach.
    """
    from rekep_libero.grasp import ee_rotation

    model, data = env.sim.model, env.sim.data
    handle = [g for g in range(model.ngeom)
              if (model.body_id2name(model.geom_bodyid[g]) or "").endswith("cabinet_middle")
              and data.geom_xpos[g][1] > -0.160]
    if not handle:
        raise SystemExit("no drawer handle in this task — use --motion push")
    truth = np.mean([data.geom_xpos[g] for g in handle], axis=0)

    approach = np.array([0.0, -1.0, 0.0])
    R = ee_rotation(approach, np.array([0.0, 0.0, 1.0]),
                    env.GRASP_APPROACH_AXIS, env.gripper_closing_axis_idx())
    quat = T.mat2quat(R)
    pos = truth - approach * env.finger_offset()

    env.execute_action(np.concatenate([pos - approach * 0.10, quat,
                                       [env.get_gripper_open_action()]]), precise=True)
    env.execute_action(np.concatenate([pos, quat, [env.get_gripper_null_action()]]), precise=True)
    env.close_gripper()

    def pull(e, t):
        target = e.get_ee_pose().copy()
        target[1] += step
        e.execute_action(np.concatenate([target[:3], quat, [e.get_gripper_null_action()]]),
                         precise=True)
    return pull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--steps", type=int, default=T_LEN)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default="data/pw_episodes")
    ap.add_argument("--step", type=float, default=0.02,
                    help="commanded wrist travel per recorded step, metres. The "
                         "checkpoint's 10-step chunks were trained on DROID scene "
                         "displacements with std 0.03-0.074 m, so this sets where "
                         "in that range the episode sits.")
    ap.add_argument("--motion", default="drawer", choices=["drawer", "push"],
                    help="drawer exercises articulation+contact; push is the trivial baseline")
    args = ap.parse_args()

    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    for ep in range(args.episodes):
        env = ReKepLiberoEnv(ec, task_suite=args.task_suite, task_id=args.task_id,
                             robot=config["libero"]["robot"],
                             resolution=config["libero"]["resolution"],
                             reset_seed=ep)
        motion = (drawer_motion(env, args.step) if args.motion == "drawer"
                  else push_forward)
        data = record(env, args.steps, motion, rng)
        tag = "" if args.step == 0.02 else f"_step{int(args.step * 1000)}mm"
        path = os.path.join(args.out, f"{args.task_suite}_{args.task_id}_ep{ep}{tag}.npz")
        np.savez_compressed(path, instruction=env.instruction, **data)

        moved = np.linalg.norm(data["scene_flows"][0, -1] - data["scene_flows"][0, 0], axis=1)
        # Per-step robot displacement is the first thing to check: with the old
        # resample-every-step sampler this was ~the geom radius (centimetres of
        # pure noise) instead of the millimetres of real gripper travel.
        step = np.linalg.norm(np.diff(data["robot_flows"][0], axis=0), axis=-1)
        rp = data["robot_flows"][0, 0]
        extent = (rp.max(0) - rp.min(0)) * 1000
        print(f"{os.path.basename(path)}: scene {data['scene_flows'].shape} "
              f"robot {data['robot_flows'].shape} | tracked {int(data['n_tracked'][0])} pts | "
              f"dropped {int(data['n_robot_px'][0])} arm px | "
              f"max scene motion {moved.max() * 1000:.1f} mm, "
              f"{int((moved > 0.002).sum())} pts moved >2mm | "
              f"robot step {step.mean() * 1000:.1f} mm mean / {step.max() * 1000:.1f} max | "
              f"robot cloud {np.round(extent).astype(int).tolist()} mm | "
              f"gripper_open {data['gripper_open'][0].ravel().astype(int).tolist()}")
    print(f"\nwrote {args.episodes} episode(s) to {args.out}/")
    print("Load these in .venv-pw with tests/run_pointworld_on_episode.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
