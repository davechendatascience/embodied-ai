"""Contact-GraspNet as ReKep's grasp proposer, replacing PCA.

Why this model
--------------
`AnalyticGraspProposer` finds an object's narrowest horizontal axis and pinches
across it. That is correct for the boxes and cubes it was proved on, and it is
what got robosuite Lift to succeed. It has no notion of *where on an object* a
grasp should go, so it fails exactly where the interesting tasks live: a bowl
(PCA centres the grasp on the empty middle, not the rim), a mug (the handle is
invisible to a covariance), a drawer front (the pull is a small feature on a
large flat face).

Contact-GraspNet predicts 6-DOF grasps per surface contact point, so it
proposes rim and handle grasps that no principal axis can express.

Why *this* Contact-GraspNet
---------------------------
It is the model already running on the robot, in
`wardmate_ws/src/llm_robot_control` (whose ROS 2 package is confusingly named
`graspgen` -- it wraps Contact-GraspNet, and is unrelated to NVIDIA's GraspGen,
which cannot be installed here: it pins `torch==2.1.0`, which has no cp312
wheel for aarch64).

Weights, thresholds and the entry point are deliberately taken from that
project's `cap_demo/grasp/contact_gn.py` rather than re-derived, so what is
tuned here stays true there. Two of its findings are load-bearing and were
paid for in debugging there, not here:

  - enter at `extract_point_clouds` + `predict_scene_grasps`, because the
    convenience wrapper `predict_scene_grasps_from_depth_K_and_2d_seg` is
    broken in this port (it unpacks two values from a three-tuple)
  - `filter_thres` 0.006 rather than the checkpoint's 0.0001, which is
    calibrated for a full-resolution depth mask

Frames
------
CGN's rotation is `[jaw_axis, cross, approach]` by column and its translation
is the gripper BASE. The Panda's ee frame approaches along local +Z and closes
along local Y. Neither the column mapping nor the base->fingertip->site offsets
are guessed; both go through `grasp.ee_rotation` and the env's measured
`finger_offset()`, which is also what the analytic proposer uses.
"""

import os
import sys
from pathlib import Path

import numpy as np

import transform_utils as T

from .grasp import ee_rotation

# The install the robot uses. Sharing it means one set of weights, not a copy
# that can drift.
CGN_ROOT = Path(os.path.expanduser("~/contact_graspnet_pytorch"))
CHECKPOINT_DIR = CGN_ROOT / "checkpoints" / "contact_graspnet"

# All four carried over from cap_demo/grasp/contact_gn.py, which took them from
# the robot's own config/graspgen_params.yaml.
SCORE_THRESHOLD = 0.15
FILTER_THRES = 0.006
FORWARD_PASSES = 3
# CGN reports the gripper base; its Panda gripper model puts the fingertip
# midpoint this far along the approach axis.
GRIPPER_DEPTH = 0.1034

_ESTIMATOR = None


def available():
    return CHECKPOINT_DIR.exists()


def _restore_numpy_compat():
    """`np.in1d` was removed in numpy 2 but this project pins 1.26.4.

    Kept as a no-op guard rather than dropped: the robot's venv and this one
    may not stay on the same numpy, and the failure is an AttributeError deep
    inside the network's first forward pass.
    """
    if not hasattr(np, "in1d"):
        np.in1d = np.isin


def allow_numpy_unpickling(torch):
    """Let torch>=2.6 load numpy-bearing pickles without weights_only=False.

    torch 2.6 flipped `torch.load(weights_only=)` to True. Two separate assets
    in this project trip over it, and they need different globals:

      * the Contact-GraspNet checkpoint stores `epoch_it` as a numpy SCALAR
      * LIBERO's per-task init states are numpy ARRAYS, which unpickle through
        `numpy.core.multiarray._reconstruct`

    Allowlisting these reconstructors is far narrower than weights_only=False,
    which permits arbitrary code execution during unpickling.

    The (callable, name) form is REQUIRED, not optional. numpy aliases
    `numpy.core.multiarray` to `numpy._core.multiarray`, so registering the
    bare callable files it under the wrong name and the pickle's
    'numpy.core.multiarray.scalar' is still rejected.

    Worth calling early and unconditionally: when LIBERO's init-state load
    failed, object placement silently fell back to random-per-reset, and three
    "repeat" runs were three different scenes -- which is indistinguishable
    from grasp variance until you check.
    """
    try:
        import numpy._core.multiarray as multiarray   # numpy >= 2
    except ImportError:
        import numpy.core.multiarray as multiarray    # numpy 1.x (pinned here)

    entries = [(multiarray.scalar, "numpy.core.multiarray.scalar"),
               (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               (np.ndarray, "numpy.ndarray"),
               (np.dtype, "numpy.dtype")]
    for name in ("Float64DType", "Float32DType", "Int64DType", "Int32DType",
                 "UInt8DType", "BoolDType"):
        dtype = getattr(getattr(np, "dtypes", None), name, None)
        if dtype is not None:
            entries.append((dtype, f"numpy.dtypes.{name}"))
    torch.serialization.add_safe_globals(entries)


# the previous name, kept so nothing silently breaks on import
_allow_numpy_scalar_unpickling = allow_numpy_unpickling


def estimator():
    """Build the network and load weights once. Cached — the load is ~2 s."""
    global _ESTIMATOR
    if _ESTIMATOR is not None:
        return _ESTIMATOR
    import torch

    _restore_numpy_compat()
    for path in (CGN_ROOT, CGN_ROOT / "Pointnet_Pointnet2_pytorch"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator

    _allow_numpy_scalar_unpickling(torch)

    cfg = config_utils.load_config(str(CHECKPOINT_DIR), batch_size=1)
    cfg["TEST"]["filter_thres"] = FILTER_THRES
    model = GraspEstimator(cfg)
    CheckpointIO(checkpoint_dir=str(CHECKPOINT_DIR / "checkpoints"),
                 model=model.model).load("model.pt")
    _ESTIMATOR = (model, cfg)
    return _ESTIMATOR


class ContactGraspNetProposer:
    """Learned 6-DOF proposer with the same seam as `AnalyticGraspProposer`.

    `propose_for_object(env, name, ...) -> (position, quat_xyzw, width)` so
    `runner._grasp_orientation` swaps one for the other without knowing which
    it holds.
    """

    def __init__(self, max_opening=0.068, clearance=0.9, finger_offset=0.0,
                 score_threshold=SCORE_THRESHOLD, top_k=32, seed=0,
                 min_view_alignment=0.1):
        self.max_opening = max_opening
        self.clearance = clearance
        self.finger_offset = finger_offset
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.seed = seed
        # A grasp cannot approach a surface from behind it. The gripper must
        # travel roughly the way the camera is looking, because that is the only
        # side of the surface there is free space on. See `actionable`.
        self.min_view_alignment = min_view_alignment
        self.last_candidates = []   # for tests/diagnostics, not the pipeline
        self.last_rejected = 0

    def fits(self, width):
        return width <= self.max_opening * self.clearance

    def _candidates(self, depth, mask, K, cam2world):
        """Camera-frame CGN output -> world (position, approach, jaw, width, score)."""
        model, _cfg = estimator()
        if self.seed is not None:
            # BOTH generators. The network samples internally (`forward_passes`
            # draws multiple stochastic passes), so seeding only numpy left the
            # candidate count varying 8-13 across runs of a scene that was
            # otherwise byte-identical -- the last hiding place of the "grasp
            # variance" that turned out to be three separate determinism bugs.
            import torch

            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        segmap = np.zeros(depth.shape, np.uint8)
        segmap[np.asarray(mask, bool)] = 1
        pc_full, pc_segments, _colors = model.extract_point_clouds(
            np.asarray(depth, np.float64), np.asarray(K, np.float64),
            segmap=segmap, segmap_id=1, z_range=[0.05, 3.0])
        if pc_full is None or len(pc_full) < 100 or not pc_segments:
            return []

        predicted, scores, _contacts, openings = model.predict_scene_grasps(
            pc_full, pc_segments=pc_segments, local_regions=True,
            filter_grasps=True, forward_passes=FORWARD_PASSES)

        rotation, origin = np.asarray(cam2world)[:3, :3], np.asarray(cam2world)[:3, 3]
        out = []
        for key, transforms in predicted.items():
            transforms = np.asarray(transforms)
            if transforms.size == 0:
                continue
            key_scores = np.asarray(scores[key]).reshape(-1)
            key_openings = np.asarray(openings.get(key, [])).reshape(-1)
            for i, transform in enumerate(transforms):
                score = float(key_scores[i])
                if score < self.score_threshold:
                    continue
                jaw = rotation @ transform[:3, 0]
                approach = rotation @ transform[:3, 2]
                # the 4x4 translation is the gripper BASE; advance to the
                # fingertip midpoint, which is what a grasp position means here
                base = rotation @ transform[:3, 3] + origin
                width = (float(key_openings[i]) if i < len(key_openings)
                         else self.max_opening)
                out.append({"position": base + GRIPPER_DEPTH * approach,
                            "approach": approach, "jaw": jaw,
                            "width": width, "score": score})
        out.sort(key=lambda g: -g["score"])
        return out[: self.top_k]

    FINGER_LENGTH = 0.054      # Panda fingertip pad length along the approach

    def jaw_occupancy(self, cand, points, approach_axis, closing_axis_idx):
        """How many target points lie inside the volume the jaws will close on.

        The decisive check, and the one distance-to-keypoint cannot express.
        Ranking by distance is ISOTROPIC; objects are not. A drawer handle is
        ~65 mm along the bar and a few mm thick, so an error of 22 mm ALONG it
        is harmless while 7 mm ABOVE it closes on air. Measured: a grasp 22.5 mm
        away (dz +1.7 mm) opened the drawer 139.8 mm; grasps 14 mm away
        (dz +7 mm) closed on nothing, three runs out of three.

        A grasp that will hold something has that something between its
        fingers. Everything else is a proxy for this.
        """
        R = ee_rotation(cand["approach"], cand["jaw"], approach_axis, closing_axis_idx)
        approach = R @ approach_axis
        tips = cand["position"] - self.finger_offset * approach
        local = (np.asarray(points) - tips) @ R

        approach_idx = int(np.argmax(np.abs(approach_axis)))
        third_idx = ({0, 1, 2} - {approach_idx, closing_axis_idx}).pop()
        along = local[:, approach_idx] * np.sign(approach_axis[approach_idx])

        inside = (along > -self.FINGER_LENGTH) & (along < 0.005)
        inside &= np.abs(local[:, closing_axis_idx]) < max(cand["width"], 0.02) / 2 + 0.005
        inside &= np.abs(local[:, third_idx]) < 0.02
        return int(inside.sum())

    def actionable(self, candidates, view_dir=None, action_axis=None,
                   min_action_alignment=0.5, points=None, approach_axis=None,
                   closing_axis_idx=None, min_points=3):
        """Drop grasps that are stable but cannot accomplish anything.

        Contact-GraspNet scores whether a grasp would HOLD. That is not the same
        question as whether it would WORK, and the gap is exactly what AO-Grasp
        was built to close ("stable and actionable"). Two filters, both cheap:

        VIEW ALIGNMENT (always applied). The gripper has to reach the surface,
        and the only side with free space on it is the side the camera is
        looking from. So the approach must point roughly along the view
        direction. Measured on the libero_goal cabinet, the rejected candidates
        had approach . (-Y) = -1.00 -- approaching the drawer from inside the
        cabinet, outward. Those are not merely poor grasps, they are not
        executable at all, and they were being ranked first.

        ACTION AXIS (optional). When the task defines the direction the gripper
        must travel -- the pull axis of a drawer, the swing of a door -- a grasp
        misaligned with it will slip off even if it closes correctly. Measured:
        an approach at 0.49 alignment stalled the drawer, 0.84 slipped off after
        2.3 mm, 0.97 pulled it 125 mm.

        Returns the surviving candidates and records how many were dropped.
        """
        kept, empty_jaws = [], 0
        for cand in candidates:
            approach = cand["approach"]
            if view_dir is not None and float(approach @ view_dir) < self.min_view_alignment:
                continue
            if action_axis is not None and float(approach @ action_axis) < min_action_alignment:
                continue
            if points is not None and len(points):
                try:
                    n = self.jaw_occupancy(cand, points, approach_axis, closing_axis_idx)
                except ValueError:      # degenerate frame; the width test drops it later
                    n = 0
                if n < min_points:
                    empty_jaws += 1
                    continue
                cand["jaw_points"] = n
            kept.append(cand)
        self.last_rejected = len(candidates) - len(kept)
        self.last_empty_jaws = empty_jaws
        return kept

    def propose_multiview(self, env, mask_fn, target, directions, cam_id=None,
                          approach_axis=None, closing_axis_idx=None, keypoint=None,
                          action_axis=None, min_action_alignment=0.5):
        """Pool candidates from several aimed viewpoints instead of trusting one.

        A single wrist view is not a stable input. Measured on the drawer
        handle, three runs from nominally the same aimed pose produced 28, 11
        and 2 candidates with approach alignments 0.97, 0.84 and 0.49 -- the
        first pulled the drawer 125 mm, the second slipped off after 2.3 mm, the
        third stalled. The arm cannot land on exactly the same pose twice, and
        Contact-GraspNet is sensitive to which 2.5D slice it gets.

        Pooling fixes that the way the robot pipeline already does it
        (cap_demo/grasp/cloud.py captures and fuses multiple views): a view that
        happens to yield nothing is covered by its neighbours, and the best
        candidate across all views is at least as good as the best from any one.

        Grasps are in the world frame, so they pool directly -- no fusion of the
        clouds themselves is needed. The VIEW filter must be applied per view
        though, before pooling, because each view has its own view direction and
        "reachable from here" is a statement about that view alone.
        """
        from .environment_libero import WRISTVIEW

        if cam_id is None:
            cam_id = WRISTVIEW
        if approach_axis is None:
            approach_axis = env.GRASP_APPROACH_AXIS
        if closing_axis_idx is None:
            closing_axis_idx = env.gripper_closing_axis_idx()

        pooled, views_used, rejected, empty = [], 0, 0, 0
        for direction in directions:
            if env.look_at(target, direction=direction) is None:
                continue
            mask = mask_fn(cam_id)
            if np.asarray(mask).sum() < 20:
                continue
            depth, K, cam2world = env.camera_view(cam_id)
            cands = self._candidates(depth, mask, K, cam2world)
            # the target's own points from THIS view, so the jaw test is against
            # the same surface the network was looking at
            points = env.get_cam_obs()[cam_id]["points"].reshape(-1, 3)[
                np.asarray(mask).reshape(-1)]
            keep = self.actionable(cands, view_dir=np.asarray(cam2world)[:3, 2],
                                   action_axis=action_axis,
                                   min_action_alignment=min_action_alignment,
                                   points=points, approach_axis=approach_axis,
                                   closing_axis_idx=closing_axis_idx)
            rejected += self.last_rejected
            empty += getattr(self, "last_empty_jaws", 0)
            pooled.extend(keep)
            views_used += 1

        self.last_candidates = pooled
        self.last_rejected = rejected
        self.last_empty_jaws = empty
        self.last_views = views_used
        self.last_too_wide = 0
        self.last_degenerate = 0
        # Among candidates that all hold SOMETHING, prefer the one holding the
        # most: jaw_points is a direct measure of grip, where distance to the
        # keypoint is only a proxy and an isotropic one at that.
        ordered = self.rank(pooled, keypoint)
        ordered.sort(key=lambda c: -c.get("jaw_points", 0))
        for cand in ordered:
            if not self.fits(cand["width"]):
                self.last_too_wide += 1
                continue
            try:
                R = ee_rotation(cand["approach"], cand["jaw"],
                                approach_axis, closing_axis_idx)
            except ValueError:
                self.last_degenerate += 1
                continue
            position = cand["position"] - self.finger_offset * cand["approach"]
            return position, T.mat2quat(R), cand["width"]
        return None

    def rank(self, candidates, keypoint=None):
        """Order candidates the way ReKep does: nearest the grasp keypoint.

        Upstream states it plainly -- "we always return the grasp closest to a
        specified 'grasp keypoint' by exploiting the fact that ReKep related to
        grasping always associates a dummy keypoint on the end-effector and one
        actual keypoint" -- and uses AnyGrasp purely as a detector, never as a
        metric, because it is too expensive to evaluate inside the optimizer.

        Ranking by the network's own confidence instead is subtly wrong for
        exactly the cases that matter. Measured on the libero_goal cabinet, the
        highest-scoring grasp sits 187.2 mm from the drawer handle while a
        handle-adjacent candidate is present at rank 10. The set contained the
        right grasp; sorting by score threw it away. The keypoint is what
        carries the task's intent about WHERE to grip, so it is what should
        choose.

        With no keypoint this falls back to score order, which is the sensible
        default when nothing has expressed an intent.
        """
        if keypoint is None:
            return list(candidates)
        target = np.asarray(keypoint, float)
        return sorted(candidates, key=lambda c: float(np.linalg.norm(c["position"] - target)))

    def propose_for_object(self, env, name, approach_axis=None, closing_axis_idx=None,
                           keypoint=None):
        """Best grasp on `name`, in the env's ee-frame convention.

        `keypoint` is the 3D grasp keypoint the VLM's constraint named; when
        given it selects the grasp, per ReKep. Returns None when the network
        proposes nothing that fits the gripper — a real answer (the object is
        too wide), not an error, and one `runner` already handles by falling
        back to ReKep's own subgoal.
        """
        mask = env.object_pixel_mask(name)
        return self.propose_from_mask(env, mask, approach_axis=approach_axis,
                                      closing_axis_idx=closing_axis_idx,
                                      keypoint=keypoint)

    def propose_from_mask(self, env, mask, cam_id=None, approach_axis=None,
                          closing_axis_idx=None, keypoint=None,
                          action_axis=None, min_action_alignment=0.5):
        """Same, from an explicit pixel mask and camera.

        Split out because the interesting targets are not always movable
        objects. A cabinet is scene furniture and never appears in
        `_object_geom_ids`, and the drawer handle has to be captured from the
        WRIST camera anyway -- the fixed agentview sees that face at grazing
        incidence and yields no grasps at all.
        """
        from .environment_libero import AGENTVIEW

        if cam_id is None:
            cam_id = AGENTVIEW
        if approach_axis is None:
            approach_axis = env.GRASP_APPROACH_AXIS
        if closing_axis_idx is None:
            closing_axis_idx = env.gripper_closing_axis_idx()

        depth, K, cam2world = env.camera_view(cam_id)
        if np.asarray(mask).sum() < 20:
            return None

        self.last_candidates = self._candidates(depth, mask, K, cam2world)
        # OpenCV convention: column 2 of the extrinsic is the view direction
        view_dir = np.asarray(cam2world)[:3, 2]
        usable = self.actionable(self.last_candidates, view_dir=view_dir,
                                 action_axis=action_axis,
                                 min_action_alignment=min_action_alignment)
        for cand in self.rank(usable, keypoint):
            if not self.fits(cand["width"]):
                continue
            try:
                R = ee_rotation(cand["approach"], cand["jaw"],
                                approach_axis, closing_axis_idx)
            except ValueError:
                continue        # degenerate frame; the next candidate is fine
            # command the SITE, not the fingertips: they sit `finger_offset`
            # beyond it along the approach, so without this the fingers close
            # that far past the object
            position = cand["position"] - self.finger_offset * cand["approach"]
            return position, T.mat2quat(R), cand["width"]
        return None
