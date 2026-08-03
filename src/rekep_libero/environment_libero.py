"""LIBERO/robosuite backend for ReKep, replacing the OmniGibson `environment.py`.

Implements the ~20-method contract `main.py` expects from `self.env`. Three
things are done differently from the original, all deliberate:

  * SDF: OmniGibson raycasts scene meshes with open3d. That wheel does not
    exist for aarch64/cp312, so we voxelise the depth point cloud and run a
    Euclidean distance transform instead. Cheaper, and it only sees what the
    camera sees -- occluded geometry is invisible to collision avoidance.
  * Keypoint tracking: OmniGibson attaches keypoints to USD prims. LIBERO
    publishes per-object poses in the obs dict (`plate_1_pos`/`_quat`), so we
    bind each keypoint to the nearest object body and recover it from the live
    pose. Same idea, simpler plumbing.
  * IK: Lula is Isaac-only. We use damped least squares over MuJoCo's own
    Jacobian, wrapped to expose the four attributes the solvers read
    (`success`, `position_error`, `num_descents`, `cspace_position`).
"""

import os
import time

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import add_rekep_to_path
from .init_state import remap_panda_init_state

add_rekep_to_path()

import transform_utils as T  # noqa: E402  (upstream ReKep, flat imports)
from utils import (  # noqa: E402
    bcolors,
    get_clock_time,
    angle_between_quats,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
)

# ReKep's camera dict is keyed by id; 0 is `vlm_camera` in configs/config.yaml.
AGENTVIEW = 0
WRISTVIEW = 1
CAM_NAMES = {AGENTVIEW: "agentview", WRISTVIEW: "robot0_eye_in_hand"}


class EpisodeFinished(RuntimeError):
    """LIBERO terminated the episode (success or horizon) mid-plan."""


class IKResult:
    """The four attributes subgoal_solver.py:54-60 / path_solver.py:78-82 read."""

    def __init__(self, success, position_error, num_descents, cspace_position):
        self.success = success
        self.position_error = position_error
        self.num_descents = num_descents
        self.cspace_position = cspace_position


class MujocoIKSolver:
    """Damped least squares IK against the live MuJoCo model.

    Replaces the Lula CCD solver. `num_descents` is reported as the iteration
    count actually used, which keeps ReKep's reachability cost meaningful (it
    scales cost by num_descents/max_iterations).
    """

    def __init__(self, env, reset_joint_pos, damping=0.05):
        self.env = env
        self.reset_joint_pos = np.asarray(reset_joint_pos, dtype=float)
        self.damping = damping

    def solve(self, target_pose_homo, position_tolerance=0.01, orientation_tolerance=0.05,
              max_iterations=20, initial_joint_pos=None, **kw):
        import mujoco

        sim = self.env.sim
        model, data = sim.model._model, sim.data._data
        eef_id = self.env.eef_site_id
        arm_qpos_idx = self.env.arm_qpos_idx

        q = np.array(self.reset_joint_pos if initial_joint_pos is None else initial_joint_pos, dtype=float)
        target_pos = target_pose_homo[:3, 3]
        # incoming target is in the observation convention (hand-body
        # orientation); convert it into the grip site's frame, which is what
        # site_xmat below reports. See _cache_model_refs for why they differ.
        target_mat = target_pose_homo[:3, :3] @ self.env.eef_rot_offset.T

        # Work on a scratch copy so IK probing never disturbs the live sim.
        qpos_backup = data.qpos.copy()
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        used, pos_err = 0, np.inf
        try:
            for used in range(1, max_iterations + 1):
                data.qpos[arm_qpos_idx] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)

                cur_pos = data.site_xpos[eef_id].copy()
                cur_mat = data.site_xmat[eef_id].reshape(3, 3).copy()
                pos_err_vec = target_pos - cur_pos
                rot_err_vec = 0.5 * (
                    np.cross(cur_mat[:, 0], target_mat[:, 0])
                    + np.cross(cur_mat[:, 1], target_mat[:, 1])
                    + np.cross(cur_mat[:, 2], target_mat[:, 2])
                )
                pos_err = np.linalg.norm(pos_err_vec)
                if pos_err < position_tolerance and np.linalg.norm(rot_err_vec) < orientation_tolerance:
                    break

                mujoco.mj_jacSite(model, data, jacp, jacr, eef_id)
                J = np.vstack([jacp[:, arm_qpos_idx], jacr[:, arm_qpos_idx]])
                err = np.concatenate([pos_err_vec, rot_err_vec])
                # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 e
                JJt = J @ J.T + (self.damping ** 2) * np.eye(6)
                q = q + J.T @ np.linalg.solve(JJt, err)
                q = np.clip(q, self.env.arm_joint_limits[:, 0], self.env.arm_joint_limits[:, 1])
        finally:
            data.qpos[:] = qpos_backup
            mujoco.mj_forward(model, data)

        return IKResult(
            success=bool(pos_err < position_tolerance * 3),
            position_error=float(pos_err),
            num_descents=int(used),
            cspace_position=q,
        )


class ReKepLiberoEnv:
    """ReKep's env contract, backed by a LIBERO OffScreenRenderEnv."""

    def __init__(self, config, task_suite="libero_spatial", task_id=0,
                 robot="Panda", resolution=256, verbose=False, horizon=20000,
                 init_state_id=0, reset_seed=0, gripper="default",
                 controller="OSC_POSE", ik_socket=None):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.config = config
        self.verbose = verbose
        self.bounds_min = np.array(config["bounds_min"])
        self.bounds_max = np.array(config["bounds_max"])
        self.interpolate_pos_step_size = config["interpolate_pos_step_size"]
        self.interpolate_rot_step_size = config["interpolate_rot_step_size"]
        self.resolution = resolution

        suite = benchmark.get_benchmark_dict()[task_suite]()
        self.task = suite.get_task(task_id)
        # Without a pinned init state robosuite RE-SAMPLES object placement on
        # every reset, so two runs of the same task are different scenes. That
        # was showing up as grasp "variance": cookies_1 gave 614 points one run
        # and 663 the next, and the same object drew a 0.033 m opening once and
        # 0.025 m the next. Comparing runs is meaningless until this is fixed.
        # LIBERO ships init states as a torch pickle, so this hits the SAME
        # torch>=2.6 `weights_only=True` wall as the Contact-GraspNet checkpoint
        # and needs the same allowlist. Getting this wrong is silent and
        # expensive: the first version swallowed the exception, left the scene
        # randomised, and three "repeat" runs were three different scenes with
        # the drawer handle 12 mm apart -- which reads exactly like grasp
        # variance. Warn loudly rather than degrade quietly.
        try:
            import torch

            from .grasp_cgn import allow_numpy_unpickling
            allow_numpy_unpickling(torch)
            self.init_states = suite.get_task_init_states(task_id)
        except Exception as exc:  # noqa: BLE001 - a suite may genuinely ship none
            print(f"{bcolors.WARNING}env: no init states for {task_suite}/{task_id} "
                  f"({type(exc).__name__}: {str(exc)[:80]}) — object placement will "
                  f"be RANDOM per reset, so runs are not comparable{bcolors.ENDC}")
            self.init_states = None
        self.init_state_id = init_state_id
        self.reset_seed = reset_seed
        self.instruction = self.task.language
        bddl = os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file)

        # LIBERO defaults to horizon=1000, which terminates the episode mid-task:
        # ReKep interpolates each subgoal into many waypoints and iterates the
        # OSC controller per waypoint, so a single stage can spend hundreds of
        # sim steps. Stepping past termination raises inside robosuite.
        # LIBERO's problem classes prefix the robot name with `Mounted` and
        # resolve it through LIBERO's own registry, which knows two Pandas and
        # nothing else -- robots=["UR5e"] dies with KeyError: 'MountedUR5e'
        # before a frame renders. Registering is a side effect of the import.
        if robot != "Panda":
            from . import robots_ur5e
            assert robots_ur5e.registered(), "MountedUR5e failed to register"
        self._robot_name = robot
        self._controller = controller
        # Collision-aware IK, if the cuRobo service is up. The world sent with
        # each request is the PERCEIVED one (depth, no simulator geometry), so
        # the executor avoids what it can SEE rather than what the model knows.
        self._ik_client = None
        self._ik_socket = ik_socket
        self._bddl = bddl
        self._fixture_ref = None

        # Take the fixture reference BEFORE the main env exists, not lazily on
        # first reset. The reference is a throwaway LIBERO env, and building
        # one alongside a live env creates a SECOND EGL context whose
        # destruction corrupts the first: the symptom is `get_real_depth_map`
        # asserting `0 <= depth <= 1` on a garbage depth buffer, several calls
        # later and nowhere near the cause. Constructing and closing it here
        # means only one render context is ever live at a time.
        if robot != "Panda":
            self._fixture_ref = self._fixture_reference()

        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl, robots=[robot], gripper_types=gripper,
            camera_heights=resolution, camera_widths=resolution,
            controller=controller, camera_depths=True,
            camera_segmentations="instance",
            camera_names=[CAM_NAMES[AGENTVIEW], CAM_NAMES[WRISTVIEW]],
            horizon=horizon,
        )
        self._post_env_init()

    SETTLE_STEPS = 10

    def _hold(self, gripper_action):
        """A do-nothing action of the right width for THIS controller.

        `np.zeros(6) + [grip]` is OSC-shaped and only coincidentally correct
        elsewhere: under JOINT_POSITION the width is one per joint, so a UR5e
        also wants 7 and a Panda wants 8. Hardcoding 7 made the UR5e work and
        the Panda fail with `invalid action dimension -- expected 8, got 7`.
        """
        # ControlEnv does not proxy action_dim; the inner robosuite env does
        dim = self.env.env.action_dim
        return np.concatenate([np.zeros(dim - 1), [gripper_action]])

    def _settle(self):
        """Let the scene come to rest before anything looks at it.

        LIBERO places every object at a nominal z=0.970 and lets physics drop it
        onto the table. Measured on libero_spatial task 0, the objects fall
        60.6-71.6 mm within the first FIVE sim steps and are then bit-stable
        forever:

            akita_black_bowl_1   0.9700 -> 0.8984   (-71.6 mm)
            cookies_1            0.9700 -> 0.9094   (-60.6 mm)
            plate_1              0.9700 -> 0.9025   (-67.5 mm)

        Everything downstream reads the observation captured at reset, so
        without this the keypoints, the image sent to the VLM, the collision SDF
        and every grasp pose describe a scene that no longer exists by the time
        the arm arrives. It is what made the physical grasp test close on air
        with 242 object points nominally between the jaws -- those points were
        where the object had been, 6 cm higher.

        Ten steps rather than five: settling is cheap, and five is the measured
        boundary rather than a margin.
        """
        for _ in range(self.SETTLE_STEPS):
            self._step(self._hold(-1.0))

    def _post_env_init(self):
        """Everything after the simulator handle exists, shared across backends."""
        self.world = self.env  # ReKep only uses this as an opaque handle
        self.step_counter = 0
        self.last_gripper_action = -1.0  # start open
        self.disturbance_seq = None
        self._frames = []
        self._last_obs = None

        # reset BEFORE caching any model handles: robosuite rebuilds the MjSim
        # on reset and frees the old one, so anything captured earlier points at
        # a dead object whose attributes have been deleted.
        # Seed BEFORE reset, not after. A pinned init state restores `qpos`,
        # which covers the robot and every free-jointed object -- but a cabinet
        # base is a FIXED body whose pose lives in the model, not the state, and
        # robosuite's placement sampler randomises it during reset() through
        # np.random. So the init state alone still left the drawer handle
        # wandering ~2 mm between runs (down from ~12 mm, which is exactly the
        # kind of partial fix that reads as success).
        np.random.seed(getattr(self, "reset_seed", 0))
        self._last_obs = self.env.reset()
        self._pin_fixtures()
        self._apply_init_state()
        self._closing_axis_idx = None    # re-measure while the gripper is open
        self._cache_model_refs()
        self._settle()
        # read joints from the sim, not the obs dict: LIBERO publishes
        # robot0_joint_pos but stock robosuite tasks do not.
        self.reset_joint_pos = self.get_arm_joint_postions().copy()
        self.ik_solver = MujocoIKSolver(self, self.reset_joint_pos)

    @property
    def sim(self):
        """Always read through to the live MjSim — never cache it (see __init__)."""
        return self.env.sim

    ROBOT_BODY_PREFIXES = ("robot", "gripper", "mount")

    # sim steps between captured video frames; 1 = real time at 20 fps
    FRAME_EVERY = 5

    # Unit vector, in the end-effector frame, pointing the way the gripper
    # closes on an object. Upstream hard-codes +X because OmniGibson's Fetch
    # approaches along its local X; the Panda approaches along local +Z.
    # Measured on LIBERO: local +X maps to world [0.998, 0, -0.057] (sideways,
    # dz=-0.006) while local +Z maps to [-0.057, 0, -0.998] (straight down).
    GRASP_APPROACH_AXIS = np.array([0.0, 0.0, 1.0])

    # must exceed keypoint_proposer.num_candidates_per_mask (5) with margin
    min_segment_pixels = 40

    def _apply_init_state(self):
        """Pin the scene to one of LIBERO's recorded init states.

        No-op for backends that ship none (stock robosuite tasks), which is why
        this is a guarded hook rather than a call in the constructor.
        """
        states = getattr(self, "init_states", None)
        if states is None or not len(states):
            return
        # LIBERO recorded these against the PANDA model, so they are only
        # positionally valid for a Panda. On any other embodiment the robot
        # block is a different width and every object address shifts --
        # `set_state_from_flattened` reads positionally and would write drawer
        # positions into a bottle's quaternion. Identity for a Panda.
        state = remap_panda_init_state(
            states[self.init_state_id % len(states)], self.sim,
            warn=(lambda m: print(f"{bcolors.WARNING}{m}{bcolors.ENDC}")),
        )
        self._last_obs = self.env.set_init_state(state)

    def _fixture_reference(self):
        """Fixture poses from a PANDA scene with the same bddl and the same seed.

        Built once and cached. It costs one throwaway env, which is worth it:
        the alternative is a JSON that goes stale the first time anyone edits a
        bddl, and the failure would be a silently displaced cabinet.
        """
        from libero.libero.envs import OffScreenRenderEnv

        from . import fixtures as fx

        ref = OffScreenRenderEnv(
            bddl_file_name=self._bddl, robots=["Panda"],
            camera_heights=64, camera_widths=64,
            controller="OSC_POSE", camera_depths=False,
            camera_names=[CAM_NAMES[AGENTVIEW]], horizon=100,
        )
        try:
            np.random.seed(getattr(self, "reset_seed", 0))
            ref.reset()
            return fx.snapshot(ref.sim)
        finally:
            ref.close()

    def _pin_fixtures(self):
        """Force a non-Panda scene's furniture onto the Panda scene's placement.

        Seeding np.random before reset() is NOT sufficient across embodiments:
        robosuite draws `randn(n_arm_joints)` of initialization noise even at
        zero magnitude, so a 6-DOF arm leaves the stream one draw ahead of a
        7-DOF one and every fixture sampled afterwards moves. Measured at
        6.87 mm on libero_goal/0. See `fixtures.py`.
        """
        if getattr(self, "_robot_name", "Panda") == "Panda":
            return
        from . import fixtures as fx

        if getattr(self, "_fixture_ref", None) is None:
            self._fixture_ref = self._fixture_reference()
        fx.pin(self.sim, self._fixture_ref,
               warn=lambda m: print(f"{bcolors.WARNING}{m}{bcolors.ENDC}"))

    def _cache_model_refs(self):
        """Index lookups that are only valid for the current MjSim instance."""
        self.robot = self.env.robots[0]
        self._cache_robot_geoms()
        self.eef_site_id = self.sim.model.site_name2id(self.robot.controller.eef_name)
        self.arm_qpos_idx = np.array(self.robot._ref_joint_pos_indexes)
        self.arm_joint_limits = self.sim.model.jnt_range[self.robot._ref_joint_indexes]

        # robosuite reports `robot0_eef_pos` from the grip SITE but
        # `robot0_eef_quat` from the hand BODY, and on the Panda those frames
        # differ by a constant 90 deg. Every pose ReKep hands us is in the
        # observation convention, so the IK -- which reads the site's xmat --
        # needs this offset or it chases a phantom 90 deg rotation.
        site_mat = self.sim.data.site_xmat[self.eef_site_id].reshape(3, 3)
        obs_mat = T.quat2mat(self._last_obs["robot0_eef_quat"])
        self.eef_rot_offset = site_mat.T @ obs_mat

    # ------------------------------------------------------------------
    # perception
    # ------------------------------------------------------------------
    def _camera_geometry(self, cam_name):
        from robosuite.utils.camera_utils import (
            get_camera_intrinsic_matrix, get_camera_extrinsic_matrix,
        )
        K = get_camera_intrinsic_matrix(self.sim, cam_name, self.resolution, self.resolution)
        cam2world = get_camera_extrinsic_matrix(self.sim, cam_name)
        return K, cam2world

    def _depth_to_points(self, depth_raw, cam_name):
        """Backproject a full depth image to world-frame points, shape (H, W, 3).

        The vertical flip is required: MuJoCo renders with an OpenGL bottom-left
        origin while the intrinsic matrix assumes top-left. Flipping only the
        depth or only the RGB is worse than flipping neither -- both are flipped
        consistently here and in get_cam_obs.
        """
        from robosuite.utils.camera_utils import get_real_depth_map

        depth = get_real_depth_map(self.sim, depth_raw)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth[::-1]

        K, cam2world = self._camera_geometry(cam_name)
        h, w = depth.shape
        rows, cols = np.mgrid[0:h, 0:w]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        z = depth
        x = (cols - cx) / fx * z
        y = (rows - cy) / fy * z
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        pts_world = pts_cam @ cam2world.T
        return pts_world[..., :3]

    def _robot_pixel_mask(self, points):
        """Per-pixel mask of the robot's own body, from kinematics not vision.

        The keypoint proposer turns every distinct segmentation id into a mask
        group and samples keypoints from each, so leaving the arm in the mask
        yields keypoints floating on the robot. Those then get offered to the
        VLM, which cannot tell them from object keypoints -- and a grasp
        constraint bound to one can never be satisfied.
        """
        data, model = self.sim.data, self.sim.model
        flat = points.reshape(-1, 3)
        is_robot = np.zeros(len(flat), dtype=bool)
        for gid in self._robot_geom_ids:
            center, radius = data.geom_xpos[gid], model.geom_rbound[gid]
            is_robot |= np.linalg.norm(flat - center, axis=1) <= radius
        return is_robot.reshape(points.shape[:2])

    def robot_geom_mask(self, cam_id=AGENTVIEW):
        """Exact per-pixel robot mask, from the geom-id segmentation buffer.

        `_robot_pixel_mask` above tests `geom_rbound` spheres. That is the right
        shape of answer for keeping the keypoint proposer OFF the arm -- being
        over-inclusive costs nothing there. It is the wrong tool for REMOVING
        the arm from a point cloud: the hand's rbound is 119.6 mm, so it would
        delete the drawer handle the gripper is closed around, which is the one
        part of the scene we most need to keep.

        MuJoCo will render geom ids directly, which is exact, costs one extra
        render, and needs no radius at all. Returned flipped to the same
        top-left origin as `get_cam_obs`'s rgb and points.
        """
        from robosuite.utils.camera_utils import get_camera_segmentation

        seg = get_camera_segmentation(self.sim, CAM_NAMES[cam_id],
                                      self.resolution, self.resolution)
        robot = np.concatenate([self._robot_geom_ids, self._gripper_geom_ids])
        return np.isin(seg[:, :, 1], robot)

    def get_cam_obs(self):
        obs = self._last_obs
        out = {}
        for cam_id, cam_name in CAM_NAMES.items():
            rgb = np.asarray(obs[f"{cam_name}_image"])[::-1]
            seg = np.asarray(obs[f"{cam_name}_segmentation_instance"])[::-1, :, 0].copy()
            points = self._depth_to_points(obs[f"{cam_name}_depth"], cam_name)
            # Fold the arm into the background id. Background is then >50% of
            # the frame and the proposer's max_mask_ratio skips it outright,
            # so no keypoints are ever proposed on the robot.
            seg[self._robot_pixel_mask(points)] = 0
            # Masking the arm can leave slivers of a segment behind. The
            # proposer runs kmeans for `num_candidates_per_mask` clusters per
            # segment and raises "Cannot take a larger sample than population"
            # if a segment has fewer pixels than that, so fold the remnants
            # into the background too.
            uids, counts = np.unique(seg, return_counts=True)
            for uid, count in zip(uids, counts):
                if uid != 0 and count < self.min_segment_pixels:
                    seg[seg == uid] = 0
            out[cam_id] = {"rgb": rgb, "depth": obs[f"{cam_name}_depth"], "points": points, "seg": seg}
        return out

    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        """Signed distance grid from the depth point cloud. Positive = free.

        Depth-derived, so occluded geometry does not exist as far as collision
        avoidance is concerned. Acceptable for a table-top workspace with an
        overhead-ish agentview; would need real mesh SDFs for cluttered scenes.
        """
        pts = self.get_cam_obs()[AGENTVIEW]["points"].reshape(-1, 3)

        if exclude_robot:
            # The depth cloud contains the arm, so without this the robot is an
            # obstacle to itself and the solver refuses to move anywhere near
            # its own gripper. Drop points falling inside any robot geom's
            # bounding sphere -- ground truth from the model, no reliance on
            # mapping camera segmentation ids back to bodies.
            data, model = self.sim.data, self.sim.model
            centers = data.geom_xpos[self._robot_geom_ids]
            radii = model.geom_rbound[self._robot_geom_ids]
            keep = np.ones(len(pts), dtype=bool)
            for center, radius in zip(centers, radii):
                keep &= np.linalg.norm(pts - center, axis=1) > radius
            pts = pts[keep]

        if exclude_obj_in_hand:
            # A grasped object is rigidly attached to the gripper, so leaving it
            # in the obstacle map makes the robot collide with what it is
            # carrying -- the solver then refuses to move at all. ReKep's own
            # README calls this out: the SDF must ignore "robot arm and any
            # grasped objects".
            data, model = self.sim.data, self.sim.model
            for name in self._contacting_objects():
                for gid in self._object_geom_ids.get(name, []):
                    center, radius = data.geom_xpos[gid], model.geom_rbound[gid]
                    pts = pts[np.linalg.norm(pts - center, axis=1) > radius]

        shape = np.ceil((self.bounds_max - self.bounds_min) / resolution).astype(int)
        shape = np.maximum(shape, 1)

        inside = np.all((pts >= self.bounds_min) & (pts < self.bounds_max), axis=1)
        idx = ((pts[inside] - self.bounds_min) / resolution).astype(int)
        idx = np.clip(idx, 0, shape - 1)

        occupied = np.zeros(shape, dtype=bool)
        if len(idx):
            occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        if not occupied.any():
            return np.full(shape, 1.0, dtype=np.float32)

        # distance (in metres) from each free voxel to the nearest occupied one
        free = distance_transform_edt(~occupied, sampling=[resolution] * 3)
        # ...and negative depth inside the occupied region
        solid = distance_transform_edt(occupied, sampling=[resolution] * 3)
        return (free - solid).astype(np.float32)

    def _cache_robot_geoms(self):
        """Split geoms into robot vs scene, with bounding radii.

        Ground truth from the model beats guessing at segmentation instance ids,
        and it is what makes `exclude_robot` in get_sdf_voxels actually work.
        """
        model = self.sim.model
        robot_ids, gripper_ids = [], []
        for gid in range(model.ngeom):
            body_name = model.body_id2name(model.geom_bodyid[gid])
            if body_name is None:
                continue
            if body_name.startswith(self.ROBOT_BODY_PREFIXES):
                robot_ids.append(gid)
                if body_name.startswith("gripper") or "hand" in body_name:
                    gripper_ids.append(gid)
        self._robot_geom_ids = np.array(robot_ids, dtype=int)
        self._gripper_geom_ids = np.array(gripper_ids, dtype=int)
        self._gripper_geom_set = set(gripper_ids)

        # object name (as it appears in the obs dict) -> its geom ids, so a
        # grasped object can be removed from the SDF the same way the arm is.
        self._object_geom_ids = {}
        articulated = self._articulated_links()
        for name in self._object_poses():
            if name in articulated:
                # Fixture geoms are named after the whole fixture
                # (`wooden_cabinet_1_g29`), not the link that owns them, so a
                # name prefix match returns nothing at all. Body membership is
                # what actually separates one drawer from the next.
                ids = [gid for gid in range(model.ngeom)
                       if int(model.geom_bodyid[gid]) == articulated[name]]
            else:
                ids = [gid for gid in range(model.ngeom)
                       if (model.body_id2name(model.geom_bodyid[gid]) or "").startswith(name)]
            if ids:
                self._object_geom_ids[name] = np.array(ids, dtype=int)

    def finger_offset(self):
        """How far the fingertips sit beyond the ee site, along the approach axis.

        Needed by any grasp proposer: commanding the SITE to an object's
        mid-height puts the FINGERS above it, and they close on air.
        """
        model, data = self.sim.model, self.sim.data
        tips = [data.geom_xpos[g] for g in self._gripper_geom_ids
                if "tip" in (model.body_id2name(model.geom_bodyid[g]) or "")]
        if not tips:
            return 0.0
        R = T.quat2mat(self.get_ee_pose()[3:])
        approach_world = R @ self.GRASP_APPROACH_AXIS
        return float(np.dot(np.mean(tips, axis=0) - self.get_ee_pos(), approach_world))

    def _contacting_objects(self):
        """Object names whose geoms are currently touching a gripper geom.

        Ground truth from MuJoCo's contact list, which beats inferring a grasp
        from finger width alone -- that cannot tell you *which* object is held.
        """
        data, model = self.sim.data, self.sim.model
        geom2object = {}
        for name, ids in self._object_geom_ids.items():
            for gid in ids:
                geom2object[int(gid)] = name

        touching = set()
        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 in self._gripper_geom_set and g2 in geom2object:
                touching.add(geom2object[g2])
            elif g2 in self._gripper_geom_set and g1 in geom2object:
                touching.add(geom2object[g1])
        return touching

    def get_collision_points(self, noise=True):
        """Volumetric sample of the gripper, in world frame.

        ReKep centers these on the current ee pose and re-transforms them by each
        candidate pose, querying the scene SDF -- this is how the robot's own
        physical extent enters collision avoidance. Sampling each gripper geom's
        bounding box matters: body origins alone are a few dimensionless dots and
        would let the solver drive the fingers straight through an object.
        """
        model, data = self.sim.model, self.sim.data
        rng = np.random.default_rng(0)
        pts = []
        for gid in self._gripper_geom_ids:
            center = data.geom_xpos[gid]
            rot = data.geom_xmat[gid].reshape(3, 3)
            half = np.maximum(model.geom_size[gid], 1e-4)
            local = rng.uniform(-1.0, 1.0, size=(8, 3)) * half
            pts.append(center + local @ rot.T)
        if not pts:
            return np.zeros((0, 3))
        pts = np.concatenate(pts, axis=0)
        if noise:
            pts = pts + rng.normal(0, 0.002, pts.shape)
        return pts

    # ------------------------------------------------------------------
    # keypoints
    # ------------------------------------------------------------------
    def _articulated_links(self):
        """{body_name: body_id} for fixture parts that move — drawers, doors, lids.

        Joint type is the discriminator, and it is exact rather than heuristic:
        a FREE joint means a movable object, which LIBERO publishes in the
        observation dict; a SLIDE or HINGE joint means a part of a fixture,
        which it does not publish at all. A cabinet therefore has a jointless
        `_base` (skipped) and three sliding drawer bodies (kept).

        Without these, `register_keypoints` binds a keypoint on a drawer handle
        to the nearest MOVABLE object -- measured on libero_goal task 0, that is
        `plate_1` -- and `get_keypoint_positions` then drags the "handle"
        keypoint around as the plate moves. No VLM, however good, can express a
        drawer grasp through that mapping.
        """
        import mujoco

        model = self.sim.model
        out = {}
        for body_id in range(model.nbody):
            name = model.body_id2name(body_id) or ""
            # the robot's own links are hinge-jointed too
            if not name or name.startswith("robot") or name.startswith("gripper"):
                continue
            count = model.body_jntnum[body_id]
            if count == 0:
                continue
            adr = model.body_jntadr[body_id]
            types = [int(model.jnt_type[j]) for j in range(adr, adr + count)]
            if any(t == mujoco.mjtJoint.mjJNT_FREE for t in types):
                continue
            out[name] = body_id
        return out

    def _object_poses(self):
        """{object_name: (pos, xyzw)} for everything a keypoint may be bound to.

        LIBERO also publishes `<obj>_to_robot0_eef_pos`/`_quat`, which are
        end-effector-relative offsets rather than world poses. They match the
        same name pattern, so they must be excluded explicitly or keypoints get
        bound to a frame that is not an object at all.

        Articulated fixture parts are appended from the model, since the
        observation dict carries no entry for them at all.
        """
        out = {}
        for key in self._last_obs:
            if key.endswith("_pos") and not key.startswith("robot") and "_to_robot0" not in key:
                name = key[: -len("_pos")]
                if f"{name}_quat" in self._last_obs:
                    out[name] = (np.asarray(self._last_obs[f"{name}_pos"]),
                                 np.asarray(self._last_obs[f"{name}_quat"]))

        data = self.sim.data
        for name, body_id in self._articulated_links().items():
            w, x, y, z = data.xquat[body_id]     # MuJoCo stores wxyz
            out[name] = (np.asarray(data.xpos[body_id]).copy(),
                         np.array([x, y, z, w]))
        return out

    def _nearest_object(self, kp, poses):
        """Which object a keypoint belongs to, by distance to its GEOMETRY.

        Body origin is not a usable proxy. A cabinet's three drawer bodies sit
        at almost the same origin, so a keypoint on the middle drawer's handle
        binds to `cabinet_top` and then tracks the wrong body -- measured, the
        keypoint stayed put while the real handle moved 120 mm.

        Distance to the nearest geom surface (approximated by its bounding
        radius) fixes that, and is a strict improvement for movable objects
        too: a keypoint on a bowl's rim should bind by the rim, not by how far
        the rim happens to be from the bowl's centroid.
        """
        model, data = self.sim.model, self.sim.data
        best, best_dist = None, np.inf
        for name in poses:
            gids = self._object_geom_ids.get(name)
            if gids is None or not len(gids):
                dist = float(np.linalg.norm(poses[name][0] - kp))
            else:
                dist = float(np.min(np.linalg.norm(data.geom_xpos[gids] - kp, axis=1)
                                    - model.geom_rbound[gids]))
            if dist < best_dist:
                best, best_dist = name, dist
        return best

    def register_keypoints(self, keypoints):
        """Bind each keypoint to the nearest object, storing it in that object's frame."""
        self.keypoints = np.asarray(keypoints)
        self._keypoint_registry = {}
        self._keypoint2object = {}
        poses = self._object_poses()
        for idx, kp in enumerate(self.keypoints):
            if not poses:
                self._keypoint_registry[idx] = (None, kp.copy())
                self._keypoint2object[idx] = None
                continue
            name = self._nearest_object(kp, poses)
            pos, quat = poses[name]
            world2obj = T.pose2mat([pos, quat])
            local = np.linalg.inv(world2obj) @ np.append(kp, 1.0)
            self._keypoint_registry[idx] = (name, local[:3])
            self._keypoint2object[idx] = name

    def get_keypoint_positions(self):
        assert getattr(self, "_keypoint_registry", None) is not None, "keypoints not registered"
        poses = self._object_poses()
        out = []
        for idx, (name, local) in self._keypoint_registry.items():
            if name is None or name not in poses:
                out.append(local)
                continue
            pos, quat = poses[name]
            out.append((T.pose2mat([pos, quat]) @ np.append(local, 1.0))[:3])
        return np.array(out)

    def get_object_by_keypoint(self, keypoint_idx):
        return self._keypoint2object[keypoint_idx]

    # depth noise sits a millimetre or two off the true surface
    POINT_MARGIN = 0.003

    def _object_point_mask(self, points, name):
        """Which of `points` (any shape ending in 3) belong to `name`.

        Uses the object's COLLISION BOXES exactly, not `geom_rbound` spheres.
        Every LIBERO object is one visual mesh plus a box decomposition (40
        boxes for a bowl, 1 for the cookie box), and `geom_rbound` is the
        CIRCUMSCRIBED radius of the mesh -- 0.0841 m for a bowl. Once the scene
        settles and objects rest on the table, that sphere reaches well below
        the object and swallows the table surface: measured on libero_spatial
        task 0, 325 of cookies_1's 865 points and 517 of the bowl's 1541 lay
        within 5 mm of the table top at z=0.900.

        That contamination is not cosmetic. It inflated the analytic proposer's
        measured width from 0.060 m to 0.101 m -- turning a graspable object
        into one it rejects as too wide -- and it puts table pixels in the mask
        handed to Contact-GraspNet.

        Geometry rather than the segmentation image, still, so that the point
        cloud and the pixel mask are defined by the same predicate and the two
        proposers cannot disagree about what the object is.
        """
        return self.points_in_geoms(points, self._object_geom_ids.get(name, []))

    def points_in_geoms(self, points, geom_ids, margin=None):
        """`_object_point_mask` for an arbitrary geom set.

        Split out because fixtures are not in `_object_geom_ids` -- a cabinet is
        scene furniture, not a movable object -- yet articulated manipulation
        needs exactly this predicate over a drawer's or a handle's geoms.

        `margin` inflates every box. A grasp network needs surrounding context,
        not just the part: masking a drawer handle exactly yields too few points
        for Contact-GraspNet to propose anything at all, while the handle plus a
        few centimetres of drawer front is a region it can work with.
        """
        import mujoco

        if margin is None:
            margin = self.POINT_MARGIN
        flat = np.asarray(points).reshape(-1, 3)
        model, data = self.sim.model, self.sim.data
        gids = list(geom_ids)
        boxes = [g for g in gids if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        # the box decomposition IS the collision shape, so prefer it whole;
        # fall back to spheres only for an object that has no boxes at all
        use, exact = (boxes, True) if boxes else (gids, False)

        keep = np.zeros(len(flat), dtype=bool)
        for gid in use:
            rel = flat - data.geom_xpos[gid]
            if exact:
                local = rel @ np.asarray(data.geom_xmat[gid]).reshape(3, 3)
                half = model.geom_size[gid] + self.POINT_MARGIN
                keep |= np.all(np.abs(local) <= half, axis=1)
            else:
                keep |= np.linalg.norm(rel, axis=1) <= model.geom_rbound[gid]
        return keep.reshape(np.asarray(points).shape[:-1])

    def object_points(self, name):
        """Depth points belonging to one object — the input a grasp proposer needs."""
        pts = self.get_cam_obs()[AGENTVIEW]["points"]
        return pts.reshape(-1, 3)[self._object_point_mask(pts, name).reshape(-1)]

    def object_pixel_mask(self, name, cam_id=AGENTVIEW):
        """(H, W) bool mask of one object, aligned with `camera_view()`'s depth.

        Contact-GraspNet consumes a depth image plus a 2D segmentation mask
        rather than a point cloud, because it builds `pc_segments` itself and
        its `filter_thres` is calibrated against a cloud built that way.
        """
        return self._object_point_mask(self.get_cam_obs()[cam_id]["points"], name)

    def camera_pose_in_ee(self, cam_id=WRISTVIEW):
        """The camera's pose expressed in the end-effector frame.

        Derived from the live sim rather than from the robot XML, and both terms
        are read the same way they are read everywhere else, so whatever
        site-vs-body offset `eef_rot_offset` exists cancels in the inverse.
        """
        _, _, cam2world = self.camera_view(cam_id)
        ee = self.get_ee_pose()
        ee2world = np.eye(4)
        ee2world[:3, :3] = T.quat2mat(ee[3:])
        ee2world[:3, 3] = ee[:3]
        return np.linalg.inv(ee2world) @ cam2world

    def ee_pose_for_view(self, eye, lookat, cam_id=WRISTVIEW, up=(0.0, 0.0, 1.0)):
        """EE pose that puts camera `cam_id` at `eye`, looking at `lookat`.

        A grasp network can only propose grasps on surfaces it can see. The
        fixed agentview sits at (0.659, 0, 1.610) looking down -X/-Z, so a
        cabinet face pointing +Y is seen at grazing incidence and Contact-
        GraspNet returns nothing at all for a drawer handle -- while the SAME
        network, given a front view, lands 14-23 mm from it with the correct
        pull direction. The wrist camera is the one that can be aimed, so this
        computes where to put the wrist to aim it.
        """
        eye, lookat = np.asarray(eye, float), np.asarray(lookat, float)
        forward = lookat - eye
        forward /= np.linalg.norm(forward)
        up = np.asarray(up, float)
        if abs(float(forward @ up)) > 0.99:      # looking straight up or down
            up = np.array([0.0, 1.0, 0.0])
        up = up - forward * float(forward @ up)
        up /= np.linalg.norm(up)

        # OpenCV camera frame: +Z is the view direction, +Y is down, x = y X z
        z_axis, y_axis = forward, -up
        x_axis = np.cross(y_axis, z_axis)
        cam2world = np.eye(4)
        cam2world[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        cam2world[:3, 3] = eye

        ee2world = cam2world @ np.linalg.inv(self.camera_pose_in_ee(cam_id))
        return np.concatenate([ee2world[:3, 3], T.mat2quat(ee2world[:3, :3])])

    def look_at(self, target, standoffs=(0.30, 0.40, 0.50), direction=(0.0, 1.0, 0.0),
                cam_id=WRISTVIEW, up=(0.0, 0.0, 1.0)):
        """Move so the wrist camera faces `target`. Returns the standoff used.

        `direction` is the side to view from, in world coordinates -- +Y for a
        cabinet whose drawers face +Y. Standoffs are tried nearest-first and the
        first IK-feasible one wins, because the useful viewpoint is often the
        closest the arm can actually hold without colliding.
        """
        target = np.asarray(target, float)
        direction = np.asarray(direction, float)
        direction = direction / np.linalg.norm(direction)
        for standoff in standoffs:
            pose = self.ee_pose_for_view(target + direction * standoff, target,
                                         cam_id=cam_id, up=up)
            result = self.ik_solver.solve(T.convert_pose_quat2mat(pose))
            if not getattr(result, "success", False):
                continue
            self.execute_action(np.concatenate([pose, [self.get_gripper_null_action()]]),
                                precise=True)
            return standoff
        return None

    def camera_view(self, cam_id=AGENTVIEW):
        """(depth, K, cam2world) for one camera, in OpenCV convention.

        `depth` is metric and row-flipped to a top-left origin, matching both
        `K` and the RGB/seg images `get_cam_obs` returns (see `_depth_to_points`
        for why the flip is not optional). robosuite's extrinsic already carries
        the OpenGL->OpenCV axis correction, so +Z is the view direction and +Y
        is down -- exactly the frame Contact-GraspNet expects, with no further
        remapping.
        """
        from robosuite.utils.camera_utils import get_real_depth_map

        cam_name = CAM_NAMES[cam_id]
        depth = get_real_depth_map(self.sim, self._last_obs[f"{cam_name}_depth"])
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        K, cam2world = self._camera_geometry(cam_name)
        return depth[::-1], K, cam2world

    def gripper_closing_axis_idx(self):
        """Which ee-frame axis the fingers separate along (local Y on the Panda).

        Measured once, with the gripper OPEN, and cached — not re-measured per
        call. The finger geoms span the closing axis only while open: at reset
        the local spread is [0.0001, -0.0663, -0.0172], so Y leads Z by just
        3.9x, and that margin is the whole answer. Closing the fingers on an
        object collapses the Y term toward the *fixed* Z extent of the finger
        bodies, the argmax flips to Z, and Z is the approach axis — which
        `grasp.ee_rotation` correctly rejects, killing a run mid-episode with
        "closing axis cannot be the approach axis".

        The answer is a property of the gripper's kinematics, not of its
        current opening, so measuring it once is also the more honest
        computation.
        """
        if self._closing_axis_idx is None:
            model, data = self.sim.model, self.sim.data
            fingers = [data.geom_xpos[g] for g in self._gripper_geom_ids
                       if "finger" in (model.body_id2name(model.geom_bodyid[g]) or "")]
            if not fingers:
                return 1
            spread = np.array(fingers).max(axis=0) - np.array(fingers).min(axis=0)
            local = T.quat2mat(self.get_ee_pose()[3:]).T @ spread
            self._closing_axis_idx = int(np.argmax(np.abs(local)))
        return self._closing_axis_idx

    def set_object_pose(self, name, pos, quat_xyzw=None):
        """Teleport a free-floating object — the primitive disturbances need.

        MuJoCo stores a free joint as [x, y, z, qw, qx, qy, qz]; note the
        quaternion is **wxyz** here while robosuite and ReKep both use xyzw.
        """
        model, data = self.sim.model, self.sim.data
        body_id = model.body_name2id(f"{name}_main")
        joint_adr = model.jnt_qposadr[model.body_jntadr[body_id]]
        data.qpos[joint_adr:joint_adr + 3] = pos
        if quat_xyzw is not None:
            x, y, z, w = quat_xyzw
            data.qpos[joint_adr + 3:joint_adr + 7] = [w, x, y, z]
        self.sim.forward()

    # ------------------------------------------------------------------
    # robot state
    # ------------------------------------------------------------------
    def execute_joint_positions(self, q_target, steps=400, kp=400.0, kd=None,
                                tol=1e-3, capture_every=0, settle_vel=0.02):
        """Drive the arm to a joint configuration with a joint-space PD.

        FAITHFUL EXECUTION. PointWorld's action is named joint values through
        URDF forward kinematics -- no end effector appears in its contract, and
        "the end effector" has no canonical meaning on the dual-arm robot this
        deploys to. Planning in joint space and then round-tripping through the
        OSC POSE controller threw that away at the last step: measured, the
        planner commanded 3.3 mm and the arm moved 0.0 mm, because a pose
        target that small is inside the OSC convergence tolerance.

        This writes torques and steps the simulator directly, so LIBERO's own
        `step()` bookkeeping is bypassed and the observation is refreshed
        afterwards. `qfrc_bias` carries gravity and Coriolis, so the PD only has
        to supply the tracking effort rather than fight the arm's own weight.

        The gripper actuators are left holding whatever they were commanded --
        a grasp in progress must not be dropped by an arm motion.
        """
        data = self.sim.data
        self._pd_prev_err = None
        idx = self.arm_qpos_idx
        act = np.asarray(self.robot._ref_joint_actuator_indexes)
        kd = 2.0 * np.sqrt(kp) if kd is None else kd
        q_target = np.asarray(q_target, dtype=float)[:len(idx)]

        # Run to a TOLERANCE, not a fixed step count. At 80 steps this reached
        # only 42-64% of the commanded delta, and a planner facing a constant
        # undershoot keeps asking for motion it never gets -- a bias that looks
        # like a weak world model. `steps` is now a safety cap, not the target.
        for used in range(steps):
            q, dq = data.qpos[idx], data.qvel[idx]
            err = q_target - q
            data.ctrl[act] = kp * err + (-kd) * dq + data.qfrc_bias[idx]
            self.sim.step()
            self.step_counter += 1
            # This path bypasses `env.step()`, which is where frames are
            # normally captured -- so a planner driving joints directly renders
            # an EMPTY video unless it asks for frames here. Opt-in, because
            # rendering two cameras with depth and segmentation is not free and
            # a planning run does not otherwise need it.
            if capture_every and used % capture_every == 0:
                self._refresh_obs()
                self._frames.append(
                    np.asarray(self._last_obs[f"{CAM_NAMES[AGENTVIEW]}_image"])[::-1])
            if np.abs(err).max() < tol and np.abs(dq).max() < 10 * tol:
                break
            # STOP WHEN IT HAS SETTLED, not when it hits an absolute tolerance.
            # A PD holding a load settles at a NONZERO steady-state error, so
            # `tol` is simply unreachable while the arm pulls a damped joint and
            # the loop always burned its whole budget. Measured on the drawer:
            # all the motion happened in the first ~300 of 1500 steps and 86% of
            # the resulting video was a stationary arm.
            #
            # So the test is whether the error is still IMPROVING. Sampled every
            # 50 steps; if it has come down by less than 2% and the joints are
            # nearly stopped, more steps will not help.
            if used % 50 == 0:
                e = float(np.abs(err).max())
                prev = getattr(self, "_pd_prev_err", None)
                if (prev is not None and e > prev * 0.98
                        and float(np.abs(dq).max()) < settle_vel):
                    self._pd_prev_err = None
                    break
                self._pd_prev_err = e
        self._refresh_obs()
        self.last_joint_steps = used + 1
        return np.asarray(data.qpos[idx]).copy()

    def _refresh_obs(self):
        """Re-read observations after stepping the sim outside `env.step()`.

        `force_update=True` IS THE WHOLE POINT and leaving it off is silent.
        robosuite's `_get_observations()` only reads each observable's CACHED
        value; the cameras are re-rendered by `_update_observables`, which
        `env.step()` calls on its own schedule and nothing else does. So this
        method -- whose entire job is to refresh after stepping the sim by
        another route -- returned the images from before the motion.

        Measured: a planner run that pulled the drawer 140 mm produced a video
        in which 19 pixels changed, against 13770 for the same motion driven
        through `env.step()`. The same stale images feed `get_cam_obs()`, so
        this was not only a rendering artefact.
        """
        inner = self.env.env if hasattr(self.env, "env") else self.env
        for getter in ("_get_observations", "_get_observation"):
            if hasattr(inner, getter):
                fn = getattr(inner, getter)
                try:
                    self._last_obs = fn(force_update=True)
                except TypeError:      # older signature with no such argument
                    if hasattr(inner, "_update_observables"):
                        inner._update_observables(force=True)
                    self._last_obs = fn()
                return
        raise RuntimeError("cannot refresh observations; no _get_observations")

    def get_ee_pose(self):
        """The end-effector pose from LIVE sim data, not the cached obs dict.

        This used to read `self._last_obs`, which is only refreshed by
        `env.step()`. Anything that moved the simulator by another route --
        writing `qpos` to evaluate a hypothetical configuration, or stepping
        torques directly -- got a STALE pose back with no error. That one line
        produced three separate symptoms before it was found: joint
        sensitivity measured as 0 mm/rad for every joint, the planner's
        commanded step measured against the wrong origin, and a real 22 mm
        motion reported as 0.0 mm.

        Convention is preserved exactly: robosuite reports `robot0_eef_pos`
        from the grip SITE but `robot0_eef_quat` from the hand BODY, and
        `eef_rot_offset = site_mat.T @ obs_mat` relates them, so the site
        rotation is mapped back into the observation frame every caller
        already expects.
        """
        data = self.sim.data
        site_mat = np.asarray(data.site_xmat[self.eef_site_id]).reshape(3, 3)
        return np.concatenate([np.asarray(data.site_xpos[self.eef_site_id]).copy(),
                               T.mat2quat(site_mat @ self.eef_rot_offset)])

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:]

    def get_arm_joint_postions(self):  # noqa: N802 - name must match ReKep's caller
        return np.asarray(self.sim.data.qpos[self.arm_qpos_idx])

    def is_grasping(self, candidate_obj=None):
        """Whether the gripper is holding `candidate_obj` (or anything, if None).

        ReKep calls this per keypoint via `_update_keypoint_movable_mask`, so it
        has to answer about a *specific* object -- a finger-width heuristic
        cannot, and would mark every keypoint movable as soon as the gripper
        closed on anything.
        """
        if self.last_gripper_action <= 0:
            return False
        touching = self._contacting_objects()
        if not touching:
            return False
        if candidate_obj is None:
            return True
        return candidate_obj in touching

    # ------------------------------------------------------------------
    # actuation
    # ------------------------------------------------------------------
    def get_gripper_open_action(self):
        return -1.0

    def get_gripper_close_action(self):
        return 1.0

    def get_gripper_null_action(self):
        return 0.0

    def get_last_og_gripper_action(self):
        return self.last_gripper_action

    def open_gripper(self):
        # Record the command BEFORE stepping: LIBERO raises EpisodeFinished the
        # instant it reports done, and for a place task that happens INSIDE
        # this call. Setting the flag afterwards leaves it stale at "shut",
        # which made the proprioceptive grasp test read a 71.6 mm OPEN jaw as
        # a held object.
        self.last_gripper_action = -1.0
        if self.last_gripper_action == -1.0:
            return
        for _ in range(15):
            self._step(self._hold(-1.0))
        self.last_gripper_action = -1.0

    def close_gripper(self):
        if self.last_gripper_action == 1.0:
            return
        for _ in range(15):
            self._step(self._hold(1.0))
        self.last_gripper_action = 1.0

    def compute_target_delta_ee(self, target_pose):
        ee = self.get_ee_pose()
        return (np.linalg.norm(ee[:3] - target_pose[:3]),
                angle_between_quats(ee[3:], target_pose[3:]))

    def _step(self, action):
        # ReKep's loop has no notion of episode termination -- it runs until its
        # own stages are exhausted. Stop cleanly the moment LIBERO says done,
        # rather than letting robosuite raise "executing action in terminated
        # episode" several waypoints later.
        if getattr(self, "last_done", False):
            raise EpisodeFinished(f"episode terminated at step {self.step_counter}")
        # Advance any active disturbance before stepping physics, mirroring
        # upstream environment.py:482. Without this the whole reactive-recovery
        # story ReKep is built on cannot be exercised at all.
        if self.disturbance_seq is not None:
            try:
                next(self.disturbance_seq)
            except StopIteration:
                self.disturbance_seq = None
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=float))
        self._last_obs = obs
        self.step_counter += 1
        self.last_reward = reward
        self.last_done = done
        # Every 5th step by default: rendering is not free and a planning run
        # does not need a smooth video. Set `env.FRAME_EVERY = 1` for one that
        # plays at real time -- `save_video` writes 20 fps, which is LIBERO's
        # control_freq, so any other capture rate is a speed-up.
        if self.step_counter % self.FRAME_EVERY == 0:
            self._frames.append(np.asarray(obs[f"{CAM_NAMES[AGENTVIEW]}_image"])[::-1])
        return obs

    def _move_to_waypoint_joint(self, target_pose, pos_threshold=0.02,
                                rot_threshold=3.0, max_steps=10):
        """Reach an absolute pose through IK and JOINT_POSITION control.

        OSC_POSE descends locally in task space, and on a 6-DOF arm that
        descent gets stuck: the scripted drawer test misses its pre-grasp by
        83 mm on the UR5e and closes on air. cuRobo IK says every one of those
        poses is reachable (45/45), so the poses were never the problem -- the
        descent was. This solves the configuration globally and then drives the
        joints to it, which is the same division of labour a planner provides,
        without the collision model.
        """
        from .arm_ik import solve_ik

        q_remote = self._solve_ik_remote(target_pose)
        if q_remote is not None:
            for _ in range(max_steps):
                q = self.get_arm_joint_postions()
                dq = q_remote - q
                if np.abs(dq).max() < 0.005:
                    break
                self._step(np.concatenate([np.clip(dq / 0.05, -1, 1),
                                           [self.last_gripper_action]]))
            return

        sim = self.sim
        idx = self.arm_qpos_idx
        limits = self.arm_joint_limits
        # get_ee_pose reports mat2quat(site_mat @ eef_rot_offset), so going
        # the other way needs the TRANSPOSE -- same as MujocoIKSolver:86.
        target_mat = T.quat2mat(target_pose[3:]) @ self.eef_rot_offset.T
        q_goal, pos_err, _rot_err, _it = solve_ik(
            sim, self.eef_site_id, idx, target_pose[:3], target_mat,
            q_init=self.get_arm_joint_postions(), joint_limits=limits)
        if pos_err > 0.05:
            # Do NOT fall back to OSC here. An OSC action is a task-space
            # delta and this env is under JOINT_POSITION, where the action is
            # one entry per joint -- feeding one to the other raises
            # `invalid action dimension` at best and drives nonsense at worst.
            # Drive the best solution we have and let the caller see the miss.
            print(f"{bcolors.WARNING}ik: {pos_err * 1000:.1f} mm short of "
                  f"{np.round(target_pose[:3], 4)}{bcolors.ENDC}")
        for _ in range(max_steps):
            q = self.get_arm_joint_postions()
            dq = q_goal - q
            if np.abs(dq).max() < 0.005:
                break
            # JOINT_POSITION input is normalised over output_max = 0.05 rad
            action = np.concatenate([np.clip(dq / 0.05, -1, 1),
                                     [self.last_gripper_action]])
            self._step(action)

    def _solve_ik_remote(self, target_pose):
        """Collision-aware IK from the cuRobo service, or None if unavailable.

        The world is rebuilt from DEPTH on every call: `world_export` is a
        snapshot and the drawer moves as it opens, so a world uploaded once is
        stale by the second waypoint.
        """
        if not self._ik_socket:
            return None
        try:
            if self._ik_client is None:
                import sys as _sys
                _sys.path.insert(0, os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                from curobo_bridge.protocol import Client
                self._ik_client = Client(self._ik_socket)
            from .frames import flange_from_obs, inv, pose_to_curobo
            from .world_depth import perceived_world

            world, _info = perceived_world(self, warn=None)
            m, d = self.sim.model, self.sim.data
            b = m.body_name2id("robot0_base")
            T_wb = np.eye(4)
            T_wb[:3, :3] = d.body_xmat[b].reshape(3, 3)
            T_wb[:3, 3] = d.body_xpos[b]
            goal = inv(T_wb) @ flange_from_obs(target_pose[:3], target_pose[3:])
            r = self._ik_client.solve(self.get_arm_joint_postions().tolist(),
                                      pose_to_curobo(goal), world, [])
            if not r or not r.get("ok"):
                print(f"{bcolors.WARNING}curobo-ik: no solution "
                      f"({r.get('pos_err', 0) * 1000:.1f} mm, "
                      f"{r.get('error', '')[:40]}){bcolors.ENDC}")
                return None
            return np.asarray(r["q"][:len(self.arm_qpos_idx)], dtype=float)
        except Exception as exc:  # noqa: BLE001 - degrade to local IK, loudly
            print(f"{bcolors.WARNING}curobo-ik unavailable: "
                  f"{type(exc).__name__}: {str(exc)[:60]}{bcolors.ENDC}")
            self._ik_socket = None
            return None

    def _move_to_waypoint(self, target_pose, pos_threshold=0.02, rot_threshold=3.0, max_steps=10):
        """Absolute-pose move, routed by whichever controller the env was built with."""
        if getattr(self, "_controller", "OSC_POSE") == "JOINT_POSITION":
            return self._move_to_waypoint_joint(target_pose, pos_threshold,
                                                rot_threshold, max_steps)
        return self._move_to_waypoint_osc(target_pose, pos_threshold,
                                          rot_threshold, max_steps)

    def _move_to_waypoint_osc(self, target_pose, pos_threshold=0.02, rot_threshold=3.0, max_steps=10):
        """OSC_POSE takes deltas; iterate until the absolute target is reached."""
        rot_threshold = np.deg2rad(rot_threshold)
        for _ in range(max_steps):
            ee = self.get_ee_pose()
            dpos = target_pose[:3] - ee[:3]
            # axis-angle delta from current to target orientation
            dmat = T.quat2mat(target_pose[3:]) @ T.quat2mat(ee[3:]).T
            drot = T.quat2axisangle(T.mat2quat(dmat))
            if np.linalg.norm(dpos) < pos_threshold and np.linalg.norm(drot) < rot_threshold:
                break
            # OSC_POSE input is normalised to [-1, 1] over the controller's limits
            action = np.concatenate([
                np.clip(dpos / 0.05, -1, 1),
                np.clip(drot / 0.5, -1, 1),
                [self.last_gripper_action],
            ])
            self._step(action)

    # Upstream uses 0.03 m for "precise" moves, sized for the OmniGibson Fetch.
    # On the Panda that is larger than the grasp itself: a move can stop 2 cm
    # short of the bowl, report success, and close the fingers on air. Measured
    # exactly that -- requested z=0.961, reached z=0.982, bowl centre at 0.970.
    PRECISE_POS_THRESHOLD = 0.005
    PRECISE_ROT_THRESHOLD = 2.0
    COARSE_POS_THRESHOLD = 0.05
    COARSE_ROT_THRESHOLD = 5.0

    def execute_action(self, action, precise=True):
        """action = (x, y, z, qx, qy, qz, qw, gripper_action), pose absolute in world frame."""
        pos_threshold, rot_threshold = (
            (self.PRECISE_POS_THRESHOLD, self.PRECISE_ROT_THRESHOLD) if precise
            else (self.COARSE_POS_THRESHOLD, self.COARSE_ROT_THRESHOLD))
        action = np.array(action).copy()
        assert action.shape == (8,)
        target_pose, gripper_action = action[:7], action[7]

        if np.any(target_pose[:3] < self.bounds_min) or np.any(target_pose[:3] > self.bounds_max):
            print(f"{bcolors.WARNING}[environment_libero | {get_clock_time()}] target out of bounds, clipping{bcolors.ENDC}")
            target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

        current_pose = self.get_ee_pose()
        pos_close = np.linalg.norm(current_pose[:3] - target_pose[:3]) < self.interpolate_pos_step_size
        rot_close = angle_between_quats(current_pose[3:7], target_pose[3:7]) < self.interpolate_rot_step_size
        if pos_close and rot_close:
            pose_seq = np.array([target_pose])
        else:
            n = get_linear_interpolation_steps(current_pose, target_pose,
                                               self.interpolate_pos_step_size,
                                               self.interpolate_rot_step_size)
            pose_seq = linear_interpolate_poses(current_pose, target_pose, n)

        for pose in pose_seq[:-1]:
            self._move_to_waypoint(pose, self.COARSE_POS_THRESHOLD, self.COARSE_ROT_THRESHOLD)
        # a tighter tolerance needs more OSC iterations to actually converge
        self._move_to_waypoint(pose_seq[-1], pos_threshold, rot_threshold,
                               max_steps=120 if precise else 30)

        pos_error, rot_error = self.compute_target_delta_ee(target_pose)

        if gripper_action == self.get_gripper_open_action():
            self.open_gripper()
        elif gripper_action == self.get_gripper_close_action():
            self.close_gripper()
        elif gripper_action != self.get_gripper_null_action():
            raise ValueError(f"Invalid gripper action: {gripper_action}")
        return pos_error, rot_error

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def reset(self):
        # Seed BEFORE reset, not after. A pinned init state restores `qpos`,
        # which covers the robot and every free-jointed object -- but a cabinet
        # base is a FIXED body whose pose lives in the model, not the state, and
        # robosuite's placement sampler randomises it during reset() through
        # np.random. So the init state alone still left the drawer handle
        # wandering ~2 mm between runs (down from ~12 mm, which is exactly the
        # kind of partial fix that reads as success).
        np.random.seed(getattr(self, "reset_seed", 0))
        self._last_obs = self.env.reset()
        self._pin_fixtures()
        self._apply_init_state()
        self._closing_axis_idx = None    # re-measure while the gripper is open
        self._cache_model_refs()
        self._settle()  # the MjSim is rebuilt by reset
        self.step_counter = 0
        self.last_gripper_action = -1.0
        self._frames = []
        self.last_reward, self.last_done = 0.0, False
        return self._last_obs

    def is_success(self):
        return bool(getattr(self, "last_done", False) and getattr(self, "last_reward", 0.0) > 0)

    def sleep(self, seconds):
        """No-op in sim: stepping physics is what advances time."""
        return

    def save_video(self, save_path=None):
        import imageio

        if not self._frames:
            return None
        save_path = save_path or os.path.join("videos", f"rekep_{int(time.time())}.mp4")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        imageio.mimsave(save_path, self._frames, fps=20)
        return save_path
