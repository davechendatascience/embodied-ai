"""Run ReKep's closed loop against LIBERO.

Subclasses upstream `Main` and replaces only its OmniGibson-specific
`__init__`. Everything below that -- stage sequencing, backtracking, subgoal
and path solving, grasp/release -- is upstream's code unmodified, so the loop
logic has a single source of truth.

    MUJOCO_GL=egl .venv/bin/python -m rekep_libero.runner --task-id 0
"""

import argparse
import os
import sys
import time

import numpy as np

from . import add_rekep_to_path
from .config import load_config
from .environment_libero import ReKepLiberoEnv, EpisodeFinished, AGENTVIEW, WRISTVIEW
from .grasp import AnalyticGraspProposer
from .environment_robosuite import ReKepRobosuiteEnv

add_rekep_to_path()

import torch  # noqa: E402
import transform_utils as T  # noqa: E402
from main import Main  # noqa: E402  (upstream ReKep loop)
from utils import print_opt_debug_dict, bcolors  # noqa: E402
from keypoint_proposal import KeypointProposer  # noqa: E402
from constraint_generation import ConstraintGenerator  # noqa: E402
from subgoal_solver import SubgoalSolver  # noqa: E402
from path_solver import PathSolver  # noqa: E402


class StepBudgetExceeded(RuntimeError):
    """Upstream `_execute` is an unbounded `while True`; this bounds it."""


class BacktrackLoop(RuntimeError):
    """Stage kept regressing — a path constraint is unsatisfiable, not merely unmet.

    Worth its own error because it is also a performance trap: `_update_stage`
    resets `first_iter = True`, so every backtrack forces a cold
    dual_annealing solve instead of the ~29ms warm-started one.
    """


def budgeted(base_cls):
    """Wrap any backend with a hard sim-step budget (upstream loops unbounded)."""
    class _Budgeted(base_cls):
        def __init__(self, *args, max_steps=3000, **kwargs):
            self.max_steps = max_steps
            super().__init__(*args, **kwargs)

        def _step(self, action):
            if self.step_counter >= self.max_steps:
                raise StepBudgetExceeded(f"exceeded {self.max_steps} sim steps")
            return super()._step(action)
    _Budgeted.__name__ = f"Budgeted{base_cls.__name__}"
    return _Budgeted


BACKENDS = {"libero": budgeted(ReKepLiberoEnv), "robosuite": budgeted(ReKepRobosuiteEnv)}


class ReKepLibero(Main):
    def __init__(self, config, task_suite="libero_spatial", task_id=0,
                 robot="Panda", resolution=256, max_steps=3000, max_backtracks=12,
                 backend="libero", workspace=None, grasp_proposer=None):
        self.max_backtracks = max_backtracks
        self._backtracks = 0
        self.config = config["main"]
        self.bounds_min = np.array(self.config["bounds_min"])
        self.bounds_max = np.array(self.config["bounds_max"])
        # upstream's Visualizer needs open3d, which has no aarch64 wheel
        self.visualize = False

        seed = self.config["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.keypoint_proposer = KeypointProposer(config["keypoint_proposer"])
        self.constraint_generator = ConstraintGenerator(config["constraint_generator"])

        env_config = dict(config["env"])
        if workspace is not None:   # backend-specific workspace overrides main
            self.bounds_min = np.array(workspace["bounds_min"])
            self.bounds_max = np.array(workspace["bounds_max"])
            self.config = dict(self.config)
            self.config["bounds_min"] = list(workspace["bounds_min"])
            self.config["bounds_max"] = list(workspace["bounds_max"])
            for sec in ("subgoal_solver", "path_solver", "keypoint_proposer"):
                config[sec] = dict(config[sec])
                config[sec]["bounds_min"] = list(workspace["bounds_min"])
                config[sec]["bounds_max"] = list(workspace["bounds_max"])
        env_config["bounds_min"] = self.config["bounds_min"]
        env_config["bounds_max"] = self.config["bounds_max"]
        env_config["interpolate_pos_step_size"] = self.config["interpolate_pos_step_size"]
        env_config["interpolate_rot_step_size"] = self.config["interpolate_rot_step_size"]

        env_cls = BACKENDS[backend]
        if backend == "robosuite":
            self.env = env_cls(env_config, robot=robot, resolution=resolution, max_steps=max_steps)
        else:
            self.env = env_cls(env_config, task_suite=task_suite, task_id=task_id,
                               robot=robot, resolution=resolution, max_steps=max_steps)
        # ReKep's solvers only need the four-attribute IK contract
        self.subgoal_solver = SubgoalSolver(config["subgoal_solver"], self.env.ik_solver, self.env.reset_joint_pos)
        self.path_solver = PathSolver(config["path_solver"], self.env.ik_solver, self.env.reset_joint_pos)

        # Which proposer supplies the grasp's 6 DOF. Resolved here rather than
        # per-call so the network's weights load once, not once per grasp
        # stage, and so an unavailable checkpoint degrades to PCA at startup
        # instead of mid-episode.
        grasp_cfg = config.get("grasp", {})
        proposer = grasp_proposer or grasp_cfg.get("proposer", "analytic")
        self._cgn_look = bool(grasp_cfg.get("aim_wrist", True))
        self._cgn = None
        if proposer == "contact_graspnet":
            from . import grasp_cgn
            if grasp_cgn.available():
                self._cgn = grasp_cgn.ContactGraspNetProposer(
                    finger_offset=self.env.finger_offset())
                grasp_cgn.estimator()
            else:
                print(f"{bcolors.WARNING}grasp: no Contact-GraspNet checkpoint at "
                      f"{grasp_cgn.CHECKPOINT_DIR} — using PCA{bcolors.ENDC}")

    # ------------------------------------------------------------------
    # Upstream assumes the Fetch's local +X approach axis in two places. Both
    # are overridden here rather than patched into third_party/, since the axis
    # is a property of the embodiment, not of ReKep.
    # ------------------------------------------------------------------
    def _approach_offset(self, rot_mat, distance):
        return rot_mat @ (self.env.GRASP_APPROACH_AXIS * distance)

    def _update_stage(self, stage):
        prev = getattr(self, "stage", None)
        if prev is not None and stage < prev:
            self._backtracks = getattr(self, "_backtracks", 0) + 1
            if self._backtracks > self.max_backtracks:
                raise BacktrackLoop(
                    f"backtracked {self._backtracks} times (last {prev}->{stage}); "
                    f"a stage-{prev} path constraint is never satisfied")
        return super()._update_stage(stage)

    def _grasp_orientation(self, keypoint_idx):
        """Wrist rotation that actually closes the fingers on the object.

        ReKep's subgoal constraint is position-only -- `norm(end_effector -
        keypoints[i])` says nothing about orientation, so the solver returns
        whatever rotation is feasible. Measured on robosuite Lift that was
        92.9 deg from the alignment the cube needs, so the fingers closed
        across its wide axis and the grasp failed even with a correct keypoint
        and a 1.3x gripper margin. Position-only constraints cannot specify a
        grasp; this supplies the missing DOF from the object's own geometry.
        A learned 6-DOF proposer would substitute here.
        """
        name = self.env.get_object_by_keypoint(keypoint_idx)
        if name is None:
            return None

        # Learned first when configured. It falls back rather than raising,
        # because "no grasp fits" is a real answer about the object, and PCA
        # answering a box well is better than no grasp at all.
        if self._cgn is not None:
            # Select by proximity to the grasp keypoint, as upstream does with
            # AnyGrasp -- the keypoint is what carries the task's intent about
            # WHERE on the object to grip, which a confidence score does not.
            keypoint = None
            if getattr(self, "keypoints", None) is not None and keypoint_idx < len(self.keypoints):
                keypoint = np.asarray(self.keypoints[keypoint_idx])

            # Aim the wrist camera before asking. The fixed agentview sits at
            # (0.659, 0, 1.610) looking down -X/-Z, so any surface facing
            # sideways is seen at grazing incidence -- measured on the
            # libero_goal cabinet, that yields ZERO grasps for a drawer handle,
            # while a front view of the same handle gives 14-23 mm with the
            # correct pull direction.
            #
            # Viewing direction is "from where the arm already is", which is
            # both reachable by construction and usually unoccluded, rather than
            # a per-object normal we have no way to know.
            cam_id = AGENTVIEW
            if self._cgn_look and keypoint is not None:
                direction = self.env.get_ee_pos() - keypoint
                if np.linalg.norm(direction) > 1e-6:
                    if self.env.look_at(keypoint, direction=direction) is not None:
                        cam_id = WRISTVIEW

            mask = self.env.object_pixel_mask(name, cam_id)
            result = self._cgn.propose_from_mask(self.env, mask, cam_id=cam_id,
                                                 keypoint=keypoint)
            if result is not None:
                pos, quat, width = result
                gap = "" if keypoint is None else \
                    f", {np.linalg.norm(pos - keypoint) * 1000:.0f}mm from keypoint"
                print(f"grasp: {name} via contact-graspnet, {width:.3f}m opening "
                      f"({len(self._cgn.last_candidates)} candidates{gap})")
                return pos, quat
            print(f"{bcolors.WARNING}grasp: contact-graspnet proposed nothing usable "
                  f"for {name} — falling back to PCA{bcolors.ENDC}")

        points = self.env.object_points(name)
        if len(points) < 10:
            return None
        proposer = AnalyticGraspProposer(finger_offset=self.env.finger_offset())
        pos, quat, width = proposer.propose(points, self.env.GRASP_APPROACH_AXIS,
                                          self.env.gripper_closing_axis_idx())
        if not proposer.fits(width):
            print(f"{bcolors.WARNING}grasp: {name} is {width:.3f}m across, wider than the "
                  f"{proposer.max_opening:.3f}m opening — keeping the solver's pose{bcolors.ENDC}")
            return None
        return pos, quat

    def _get_next_subgoal(self, from_scratch):
        subgoal_pose, debug_dict = self.subgoal_solver.solve(
            self.curr_ee_pose, self.keypoints, self.keypoint_movable_mask,
            self.constraint_fns[self.stage]["subgoal"],
            self.constraint_fns[self.stage]["path"],
            self.sdf_voxels, self.collision_points, self.is_grasp_stage,
            self.curr_joint_pos, from_scratch=from_scratch,
        )
        if self.is_grasp_stage:
            # Take BOTH position and orientation from object geometry. A keypoint
            # is a point on the visible SURFACE, so the solver's position-only
            # target sat 17.7mm off a cube of 21.7mm half-width -- the fingers
            # gripped near the edge. The proposer's centroid-based point is
            # 8.1mm off. Once both come from geometry the VLM's constraint
            # selects WHICH object and geometry supplies the full 6-DOF pose,
            # which is how ReKep's own real-robot system used AnyGrasp.
            grasp = self._grasp_orientation(self.program_info["grasp_keypoints"][self.stage - 1])
            if grasp is not None:
                subgoal_pose[:3], subgoal_pose[3:] = grasp[0], grasp[1]
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        if self.is_grasp_stage:
            # back off half the grasp depth to leave room to close in
            subgoal_pose[:3] += self._approach_offset(
                subgoal_pose_homo[:3, :3], -self.config["grasp_depth"] / 2.0)
        debug_dict["stage"] = self.stage
        print_opt_debug_dict(debug_dict)
        return subgoal_pose

    def _execute_grasp_action(self):
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += self._approach_offset(
            T.quat2mat(pregrasp_pose[3:]), self.config["grasp_depth"])
        self.env.execute_action(
            np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]]), precise=True)


def make_disturbance_seq(target_object, shift=np.array([0.0, -0.10, 0.0]), num_steps=30):
    """Slide `target_object` sideways mid-stage, then leave it there.

    This is the LIBERO analogue of upstream's pen-holder disturbance. It exists
    to exercise the property ReKep is actually built for: keypoints are re-read
    every iteration, so the solver should re-plan toward the object's NEW
    position rather than continuing to its stale one.
    """
    def build(env):
        start = env._object_poses()[target_object][0].copy()
        path = [start + shift * (i + 1) / num_steps for i in range(num_steps)]
        counter = 0
        while True:
            if counter < len(path):
                env.set_object_pose(target_object, path[counter])
            counter += 1
            yield

    return build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="libero", choices=["libero", "robosuite"])
    ap.add_argument("--suite", default=None)
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--robot", default=None)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--disturb", default=None,
                    help="object name to shove mid-execution, e.g. akita_black_bowl_1")
    ap.add_argument("--disturb-stage", type=int, default=1)
    ap.add_argument("--program-dir", default=None,
                    help="reuse previously generated constraints instead of querying the VLM")
    ap.add_argument("--grasp", default=None, choices=["analytic", "contact_graspnet"],
                    help="which proposer supplies the grasp pose (default: configs/)")
    ap.add_argument("--vlm", default=None,
                    help="constraint_generator backend override, e.g. openai_compat")
    ap.add_argument("--vlm-model", default=None)
    args = ap.parse_args()

    config = load_config()
    if args.vlm:
        config["constraint_generator"] = dict(config["constraint_generator"])
        config["constraint_generator"]["backend"] = args.vlm
        if args.vlm_model:
            config["constraint_generator"]["model"] = args.vlm_model
    lib = config["libero"]
    suite = args.suite or lib["task_suite"]
    task_id = args.task_id if args.task_id is not None else lib["task_id"]
    robot = args.robot or lib["robot"]

    workspace = config.get("robosuite_workspace") if args.backend == "robosuite" else None
    runner = ReKepLibero(config, task_suite=suite, task_id=task_id,
                         robot=robot, resolution=lib["resolution"], max_steps=args.max_steps,
                         backend=args.backend, workspace=workspace,
                         grasp_proposer=args.grasp)
    instruction = runner.env.instruction
    print(f"\n=== {suite} task {task_id} | {robot} ===\n{instruction}\n")

    t0 = time.time()
    status = "ok"
    try:
        disturbance = ({args.disturb_stage: make_disturbance_seq(args.disturb)}
                       if args.disturb else None)
        runner.perform_task(instruction, rekep_program_dir=args.program_dir,
                            disturbance_seq=disturbance)
    except BacktrackLoop as exc:
        status = f"backtrack loop: {exc}"
    except EpisodeFinished as exc:
        status = f"episode ended: {exc}"
    except StepBudgetExceeded as exc:
        status = f"step budget: {exc}"
    except Exception as exc:  # noqa: BLE001 - report rather than traceback-and-die
        status = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.time() - t0
        success = runner.env.is_success()
        print(f"\n=== result ===")
        print(f"  success : {success}")
        print(f"  status  : {status}")
        print(f"  steps   : {runner.env.step_counter}")
        print(f"  wall    : {elapsed:.1f}s")
        path = runner.env.save_video(os.path.join("videos", f"{args.backend}_{suite if args.backend=='libero' else 'lift'}_{task_id}_{robot}.mp4"))
        if path:
            print(f"  video   : {path}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
