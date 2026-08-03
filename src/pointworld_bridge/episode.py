"""Turn a recorded MuJoCo episode into the batch `BaseModel.forward` consumes.

This deliberately calls PointWorld's OWN transforms rather than re-deriving
them. Two of them are not optional and are easy to miss:

  * `center_shift` subtracts the mean of the first-frame scene+robot points
    from every point AND folds the same shift into the camera extrinsics. The
    model's raw coordinate channels are normalized against stats with mean ~0
    and std ~0.24 m; LIBERO's table sits at z ~ 1.1 m, so skipping this feeds
    coordinates ~5 sigma out of distribution.
  * `gather_features` assembles the 31 scene and 16 robot channels in the exact
    order the checkpoint's `scene_raw_feat_proj` and `robot_proj` expect. The
    order comes from `args.scene_features` / `args.robot_features` in the
    checkpoint, so it is read, not assumed.

The camera extrinsic is the one conversion we must do ourselves: our recorder
stores cam2world (OpenCV), and the featurizer treats `*_extrinsic` as
world2cam (`scene_featurizer.py:211`). `center_shift`'s extrinsic update is
also only correct for the world2cam form, so the inverse has to happen first.
"""

import sys
import types

import numpy as np
import torch

from .model import PW, add_pointworld_to_path


def _import_pointworld_transforms():
    """`dataset_components.robot` pulls in the URDF robot sampler at import.

    We never call it -- our robot points come from MuJoCo, not from forward
    kinematics on a Franka URDF -- but the import is unconditional, and urdfpy
    is not installable on this stack. Stubbing the module is narrower than
    vendoring a copy of `gather_features`, and if anything ever does reach for
    the sampler it will fail loudly rather than silently.
    """
    add_pointworld_to_path()
    for name in ("urdfpy", "trimesh"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)

    from dataset_components.robot import gather_features
    from dataset_components.transforms import center_shift, normalize_colors
    return gather_features, center_shift, normalize_colors


def load_episode(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


REQUIRED = ("scene_flows", "scene_colors", "scene_normals",
            "robot_flows", "robot_normals", "gripper_open")


def rigid_trajectory(robot_flows0, direction, per_step, T):
    """A counterfactual gripper trajectory: rigid translation at a fixed rate.

    The gripper is a rigid body over a short horizon with the fingers held, so
    a candidate action is well described by where its points go. That is the
    whole reason PointWorld's action space is point flow: a candidate needs no
    IK, no controller and no dynamics to write down.
    """
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else d
    steps = np.arange(T, dtype=np.float32)[:, None, None]
    return robot_flows0[None] + steps * (d * per_step)[None, None, :]


def build_data_dict(ep, args, device, domain="auto", gripper_open=None,
                    robot_flows=None, scene_flows=None, centre=None):
    """(data_dict, meta) ready for `BaseModel.forward`.

    `gripper_open` overrides the recorded gripper state; it exists only so the
    runner can measure how much that one channel matters, since which polarity
    DROID used for it is not documented in the release.

    `robot_flows` (T, Nr, 3) replaces the recorded action with a counterfactual
    one. Everything downstream is recomputed from it -- velocity, acceleration
    and `dist2robot` are all derived inside `gather_features` -- so a candidate
    trajectory produces a fully consistent input rather than a spliced one.
    Normals are left alone because a rigid translation does not rotate them.

    `scene_flows` (T, Ns, 3) replaces the observed cloud, for the second and
    third passes of an AUTOREGRESSIVE rollout: the paper's MPC rolls 30 steps
    as three chained 10-step chunks, so each chunk starts from the previous
    chunk's predicted points.

    `centre` pins the frame. `center_shift` normally derives the offset from
    whatever it is handed, which is right for one observation and WRONG across
    a chained rollout -- the points move, so each chunk would be centred
    differently while the cached scene features still describe the first
    frame. Passing the observation's centre keeps every chunk in one frame.
    It also removes the double-centring foot-gun that made every counterfactual
    score identically badly (`NOTES.md` section 4).
    """
    missing = [k for k in REQUIRED if k not in ep]
    if missing:
        raise KeyError(
            f"episode is missing {missing}. Re-record with the current "
            "scripts/record_pointworld_episode.py -- earlier versions stored "
            "positions only."
        )
    gather_features, center_shift, normalize_colors = _import_pointworld_transforms()

    # LIBERO is SIMULATION. The guide's own recipe for simulation evaluation is
    # `--domains=behavior --norm_stats_path=stats/droid_behavior`; using the
    # `droid` row instead un-normalises predictions through real-robot
    # statistics, silently and at the wrong scale.
    if domain == "auto":
        domain = "behavior" if "behavior" in list(args.domains) else "droid"
    bimanual = bool(getattr(args, "_bimanual", False))

    T = ep["scene_flows"].shape[1]
    n_cam = ep["rgb"].shape[1]

    grip = np.asarray(ep["gripper_open"][0], dtype=np.float32).reshape(T, 1)
    if gripper_open is not None:
        grip = np.full((T, 1), float(gripper_open), dtype=np.float32)

    sample = {
        "scene_flows": (ep["scene_flows"][0] if scene_flows is None
                        else scene_flows).astype(np.float32),          # (T,Ns,3)
        "scene_colors": ep["scene_colors"][0].astype(np.float32),      # (T,Ns,3) 0..255
        "scene_normals": ep["scene_normals"][0].astype(np.float32),
        "robot_flows": (ep["robot_flows"][0] if robot_flows is None
                        else robot_flows).astype(np.float32),          # (T,Nr,3)
        "robot_normals": ep["robot_normals"][0].astype(np.float32),
        # gather_features replaces robot colour with a constant magenta, but
        # normalize_colors still divides the key by 255 on the way past.
        "robot_colors": np.zeros_like(ep["robot_flows"][0], dtype=np.float32),
        "right_gripper_open": grip,
        # A single-arm scene expressed in the bimanual layout the DROID+BEHAVIOR
        # checkpoint was trained on. Absent sides are zero, which is what
        # `canonicalize_gripper_keys_and_flags` does upstream.
        "left_gripper_open": np.zeros_like(grip),
    }
    for c in range(n_cam):
        cam2world = np.asarray(ep["extrinsic"][0, c], dtype=np.float64)
        sample[f"cam{c}_initial_rgb"] = ep["rgb"][0, c].astype(np.uint8)
        sample[f"cam{c}_initial_depth"] = ep["depth"][0, c].astype(np.float32)
        sample[f"cam{c}_intrinsic"] = ep["intrinsic"][0, c].astype(np.float32)
        sample[f"cam{c}_extrinsic"] = np.linalg.inv(cam2world).astype(np.float32)

    if centre is None:
        sample = center_shift(sample)
        # center_shift stores the NEGATED shift; keep the centroid itself, so
        # `world = centred + centre` reads the way it sounds.
        centre = -np.asarray(sample["__shift_amount__"], dtype=np.float32)
    else:
        # Same arithmetic as center_shift, with the offset pinned rather than
        # re-derived, including its correction to the world2cam extrinsics.
        centre = np.asarray(centre, dtype=np.float32)
        for key in ("scene_flows", "robot_flows"):
            sample[key] = sample[key] - centre
        for key in list(sample):
            if key.endswith("_extrinsic"):
                e = np.array(sample[key], dtype=np.float32)
                e[:3, 3] = e[:3, 3] + e[:3, :3] @ centre
                sample[key] = e
    sample = normalize_colors(sample)
    sample = gather_features(
        sample,
        robot_features=args.robot_features,
        scene_features=args.scene_features,
        random_context_mode="fixed",
        context_horizon=1,
        has_bimanual_robot=bimanual,
        domain=domain,
    )

    Ns = sample["scene_flows"].shape[1]
    Nr = sample["robot_flows"].shape[1]

    def t(x, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device).unsqueeze(0)

    data_dict = {
        "scene_flows": t(sample["scene_flows"]),                      # (1,T,Ns,3)
        "scene_features": t(sample["scene_features"]),                # (1,1,Ns,31)
        "scene_exists": t(np.ones((T, Ns), dtype=bool), torch.bool),
        "robot_flows": t(sample["robot_flows"]),                      # (1,T,Nr,3)
        "robot_features": t(sample["robot_features"]),                # (1,T,Nr,16)
        "robot_exists": t(np.ones((T, Nr), dtype=bool), torch.bool),
        "__domain__": [domain],
    }
    for c in range(n_cam):
        data_dict[f"cam{c}_initial_rgb"] = t(sample[f"cam{c}_initial_rgb"], torch.uint8)
        data_dict[f"cam{c}_initial_depth"] = t(sample[f"cam{c}_initial_depth"])
        data_dict[f"cam{c}_intrinsic"] = t(sample[f"cam{c}_intrinsic"])
        data_dict[f"cam{c}_extrinsic"] = t(sample[f"cam{c}_extrinsic"])

    meta = {"centre": centre, "T": T, "Ns": Ns, "Nr": Nr, "cameras": n_cam,
            "gripper_open": float(grip[0, 0])}
    return data_dict, meta
