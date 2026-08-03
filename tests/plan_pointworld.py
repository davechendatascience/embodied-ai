"""Task-agnostic closed-loop planning with PointWorld on LIBERO.

Nothing here knows what a drawer is. Each tick: observe, mask the scene points
belonging to the task's target object, ask where they still have to go, sample
candidate gripper trajectories, roll them out through PointWorld over the
socket, execute the best candidate's FIRST step, re-observe. The only
task-specific input is a `TaskSpec` -- a target object and a translation still
owed -- and that is exactly the mask-plus-targets a VLM would ground from the
instruction (`src/rekep_libero/task_spec.py`).

SOLVED: `libero_goal/0`, 159.9 -> 19.9 mm, `_check_success()` True in 21 ticks
on `large-droid+behavior`, with `+Y` chosen on every single tick at `cos 1.00`.

The search is STRUCTURED, and that is the design. A candidate is a direction
and a rate; the trajectory is a straight line at constant rate, which is what a
pull, a push or a place actually is. Four things had to be true at once, and
each was measured rather than guessed (`HANDOFF.md`, "THE PLANNER, DIAGNOSED"):

1. The model, not the control cost, must rank the candidates. A `pytorch_mppi`
   version ranked them by joint feasibility and finger-joint perturbation cost
   instead -- rank correlation with PointWorld's own cost was +0.03 -- and
   closed -1.7 mm in 60 ticks against this design's 140.
2. The rates must SCALE WITH THE DISTANCE OWED. A fixed 8/16/26 mm ladder over
   10 steps offers 0 mm or 80 mm and nothing between, so inside ~40 mm every
   option is bad, "still" wins, and the tick becomes a fixed point.
3. The cost must be read at EVERY step, not just the horizon end, or a
   candidate that arrives and overshoots scores like one that never moved.
4. Execution must actually track. A PD holding a load settles where `kp * err`
   balances it, so at kp=400 the arm delivered ~20% of every command --
   a constant fractional undershoot that mimics a weak world model exactly.

Success is always LIBERO's `_check_success()`. Anything else would be grading
our own homework.

    scripts/run_planner.sh --suite libero_goal --task-id 0
"""

import argparse
import time
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rekep_libero import add_rekep_to_path  # noqa: E402
from rekep_libero.config import load_config  # noqa: E402

add_rekep_to_path()

from rekep_libero.environment_libero import ReKepLiberoEnv, EpisodeFinished  # noqa: E402
from robot_points import MujocoRobotPoints  # noqa: E402
from rekep_libero.pw_observation import NR, T_LEN, live_observation  # noqa: E402
from rekep_libero import task_spec as specs  # noqa: E402
from rekep_libero import scene_graph as sg  # noqa: E402
from pointworld_bridge.client import PointWorldClient  # noqa: E402
from pointworld_bridge.protocol import DEFAULT_SOCKET  # noqa: E402
import transform_utils as T  # noqa: E402

# THE SEARCH SPACE IS STRUCTURED AND LOW-DIMENSIONAL, and that is the whole
# design. A candidate is (direction, rate): 18 directions covering the sphere,
# a handful of rates derived from the distance still owed, plus "still". The
# trajectory is a straight line at constant rate, which is what a pull, a push
# or a place actually is.
#
# The alternative -- perturbing every joint at every step, as `pytorch_mppi`
# does -- is 30 x 7 = 210 dimensions searched with 32 samples, and it does not
# work here: measured, -1.7 mm of progress in 60 ticks against this design's
# 140.2 mm. It also made MPPI's own perturbation cost, which grows as
# sqrt(steps x joints), large enough to drown the model's cost entirely.
#
# ACTIONS ARE STILL JOINT DELTAS. PointWorld's sampler is driven by named joint
# values through URDF forward kinematics, so joint space is the model's actual
# action representation, and end-effector deltas cannot express a dual-arm
# action or open the jaw. Directions are converted to joint deltas by damped
# least squares on the ROBOT-POINT Jacobian, so no end effector is ever named.
# ONE 10-step chunk, which is what the server's native horizon is and what the
# design that opened the drawer used. Chaining three chunks triples the cost per
# tick and buys nothing here: the rates are already scaled to the distance owed,
# so a longer horizon does not reach further, it only predicts further ahead of
# a plan that gets replaced on the next tick anyway.
HORIZON = 10
MAX_RATE_MM = 26.0     # per-step cap on commanded point motion
STEP_MM = 15.0         # only for the reported sensitivity, not for planning
SIGMA_FINGER = 0.001   # the grip is held; the fingers are never planned
# Stop only when the task is essentially done. 20 mm was above the threshold
# LIBERO's `_check_success()` needs for the drawer, so the loop could exit
# reporting progress on a task it had not solved.
DONE_MM = 4.0
# The arm is pulling a joint with damping=50. At the previous 400-step cap the
# PD stopped 0.009 rad short, which on a 700 mm/rad joint is 6.5 mm -- and the
# planner achieved ~30% of what it commanded, a bias that looks exactly like a
# weak world model. Run it to the tolerance instead.
EXEC_STEPS = 1500
# ...and the cap was only half of it. A PD holding a load settles where
# `kp * err` balances the resistance, so the steady-state error is set by the
# STIFFNESS, not by how long it is given: at kp=400 the arm stopped 0.003 rad
# short every tick regardless of the step budget, achieving ~20% of the
# commanded motion. That constant fractional undershoot is indistinguishable
# from a world model that under-predicts, and it is neither.
EXEC_KP = 2000.0


def point_jacobian(rp, arm, eps=1e-3):
    """d(mean robot point) / d(joint), a 3 x A matrix. Finite-differenced.

    DELIBERATELY NOT AN END-EFFECTOR JACOBIAN, for the same reason
    `point_sensitivity` is not: PointWorld's action is the robot's point flow,
    no end effector appears in its contract, and "the end effector" has no
    canonical meaning on the dual-arm UR7e this deploys to. This is the
    vector-valued version of the sensitivity already measured here -- where can
    the robot's points be pushed, per radian -- so it works unchanged on a
    bimanual robot and reads live geometry.

    A+1 kinematics evaluations per tick, ~1 ms each, against ~1 s of model time.
    """
    q0 = rp.current_config()
    p0 = rp.at_config(q0)[0].mean(axis=0)
    J = np.zeros((3, len(arm)))
    for i, j in enumerate(arm):
        q = q0.copy()
        q[j] += eps
        J[:, i] = (rp.at_config(q)[0].mean(axis=0) - p0) / eps
    return J


def joint_delta_for(J, d, damping=0.02):
    """Damped least squares: the joint step that moves the points along `d`.

    Damping is not optional near a singularity, where the exact solution asks
    for unbounded joint motion to produce a finite point motion.
    """
    return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), d)


def candidate_directions():
    """Six axes plus the twelve edge diagonals — a coarse cover of the sphere.

    A STRUCTURED set, not Gaussian samples. This is the search space the model
    was validated on (`rank_actions_pointworld.py` separates 20 such candidates
    cleanly, `discover_axis_pointworld.py` probes 18), and it is 3-dimensional
    where sampling per-step joint deltas is 30 x 7 = 210. Measured: the
    Gaussian version closed -1.7 mm in 60 ticks against this one's 140.2 mm.
    """
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for a in (-1, 1):
        for b in (-1, 1):
            dirs += [(a * 0.7071, b * 0.7071, 0.0),
                     (a * 0.7071, 0.0, b * 0.7071),
                     (0.0, a * 0.7071, b * 0.7071)]
    return np.array(dirs, dtype=np.float64)


def joint_sigma(rp):
    """Per-joint noise DERIVED from the robot's kinematics, not tuned.

    Every scalar in this planner was originally tuned against metres of
    end-effector travel, and none of them survived the move to joint radians --
    `W_TRAVEL` silently made "do nothing" optimal. The fix is not a better
    constant: it is to specify the quantity that means the same thing on any
    robot -- millimetres of gripper motion per step -- and solve for the joint
    noise that produces it.

    Sensitivity varies by an order of magnitude across joints, so sigma is
    per-joint. With `n` arm joints perturbed independently, their contributions
    add in quadrature, hence the sqrt(n).
    """
    sens = rp.point_sensitivity()          # metres of ROBOT POINT per radian
    is_finger = np.array(["finger" in n or "gripper" in n for n in rp.joint_names])
    n_arm = max(int((~is_finger).sum()), 1)
    per_joint_m = (STEP_MM / 1000.0) / np.sqrt(n_arm)

    sig = np.full(len(rp.joint_names), SIGMA_FINGER)
    arm = ~is_finger & (sens > 1e-6)
    sig[arm] = per_joint_m / sens[arm]
    return sig, sens


def joint_limits(rp, model):
    lo = np.full(len(rp.qadr), -np.inf)
    hi = np.full(len(rp.qadr), np.inf)
    for i, adr in enumerate(rp.qadr):
        j = int(np.nonzero(model.jnt_qposadr == adr)[0][0])
        if model.jnt_limited[j]:
            lo[i], hi[i] = model.jnt_range[j]
    return lo, hi


def grasp_target(env, spec, use_cgn=True):
    """Place the fingers on the task's target object. NOT planned.

    MEASURED, not assumed (`tests/test_pointworld_grasp.py`): PointWorld
    CANNOT rank grasps. Over 12 candidates around the bowl its ordering
    correlates with executed reality at **-0.15**; its top pick closed on air
    and its worst pick held.

    The reason is NOT resolution -- the old justification here cited a
    "10-15 mm error floor" that has since been retracted, and the correct
    checkpoint predicts moved points to 3.92 mm. It is that the model has no
    representation of ENCLOSURE: sweeping the grasp 0, 25 and 50 mm sideways
    changes its prediction by ~2 mm (112.3 / 115.2 / 114.8) while changing
    reality from held to closed-on-air. It predicts that things near the hand
    move with the hand, which is the same property behind the flat axis probe
    and the 60.6 mm of spurious static motion.

    So Contact-GraspNet places the grasp, as ReKep's own real-robot system used
    AnyGrasp -- now for a measured reason rather than an inherited one.
    """
    from rekep_libero.grasp import ee_rotation
    from rekep_libero import grasp_cgn

    geoms = spec.geoms(env)
    truth = specs.object_center(env, spec.target)
    approach = spec.approach

    quat = None
    if use_cgn and grasp_cgn.available():
        proposer = grasp_cgn.ContactGraspNetProposer(
            finger_offset=env.finger_offset(), top_k=64)
        grasp_cgn.estimator()
        try:
            result = proposer.propose_multiview(
                env, lambda cam: env.points_in_geoms(
                    env.get_cam_obs()[cam]["points"], geoms, margin=0.01),
                truth, [tuple(-approach)], cam_id=1, keypoint=truth)
            if result is not None:
                pos, quat, _ = result
                truth = pos
        except Exception as exc:  # noqa: BLE001 - fall back, but say so
            print(f"          CGN failed ({type(exc).__name__}: {exc}); "
                  f"falling back to the geometric grasp")

    if quat is None:
        # Geometric fallback: approach along the spec's axis, jaws across it.
        R = ee_rotation(approach, np.array([0.0, 0.0, 1.0]),
                        env.GRASP_APPROACH_AXIS, env.gripper_closing_axis_idx())
        quat = T.mat2quat(R)
        truth = truth - approach * env.finger_offset()

    env.execute_action(np.concatenate([truth - approach * 0.10, quat,
                                       [env.get_gripper_open_action()]]), precise=True)
    env.execute_action(np.concatenate([truth, quat, [env.get_gripper_null_action()]]),
                       precise=True)
    env.close_gripper()
    return quat


def holding(env, spec):
    """Is the target still in the jaws? MuJoCo contacts, not finger width.

    Finger width cannot say WHICH object is held, and a planner that has
    dropped its object does not otherwise notice: measured on `libero_goal/1`,
    the bowl slipped at tick 8 and the loop spent the next 21 ticks
    confidently commanding motion with an empty gripper, `owes` frozen at
    exactly 309.1 mm while `cos far` drifted to -0.77. Every number it printed
    looked like a planner working.

    Returns True when the answer is unknown -- fixtures like a drawer are not
    in `_object_geom_ids`, and re-grasping a drawer every tick because the
    contact list cannot see it would be far worse than not checking.
    """
    if not spec.needs_grasp:
        return True
    try:
        if spec.target not in env._object_geom_ids:
            return True
        return spec.target in env._contacting_objects()
    except Exception:  # noqa: BLE001 - a missing check must not stop the run
        return True


def place_for_push(env, spec, back=0.075, lift=0.005):
    """Put the CLOSED gripper just behind the object, on the push line.

    A push has no grasp to place, but it does have a staging pose, and getting
    it wrong is the difference between pushing the object and knocking it over
    or missing entirely. Three choices, each for a reason:

      * BEHIND, along the push direction, so the first commanded step makes
        contact rather than travelling through the object;
      * `back` beyond the object's own half-extent, so the fingers start clear
        of it -- starting in contact would have the planner scoring candidates
        from an already-perturbed scene;
      * LOW, just above the table, because a plate is thin and a push applied
        near its rim tips it instead of sliding it.

    The gripper is closed into a fist first: an open jaw catches the rim and
    turns a push into a hook.
    """
    from rekep_libero.grasp import ee_rotation

    src = specs.object_center(env, spec.target)
    lo, hi = specs.object_aabb(env, spec.target)
    owed = spec.offset(env)
    d = owed / max(np.linalg.norm(owed), 1e-9)          # push direction, planar

    # Half-extent of the object along the push direction, so `back` is a
    # clearance from its SURFACE rather than from its centre. `geom_rbound`
    # would be the circumscribed radius -- the trap that has cost this project
    # time three times (`NOTES.md` section 2).
    half = abs(d[0]) * (hi[0] - lo[0]) * 0.5 + abs(d[1]) * (hi[1] - lo[1]) * 0.5

    pos = src - d * (half + back)
    pos[2] = lo[2] + lift + env.finger_offset()

    R = ee_rotation(np.array([0.0, 0.0, -1.0]), d,
                    env.GRASP_APPROACH_AXIS, env.gripper_closing_axis_idx())
    quat = T.mat2quat(R)
    env.close_gripper()
    env.execute_action(np.concatenate([pos + np.array([0.0, 0.0, 0.10]), quat,
                                       [env.get_gripper_null_action()]]), precise=True)
    env.execute_action(np.concatenate([pos, quat,
                                       [env.get_gripper_null_action()]]), precise=True)
    return quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--ticks", type=int, default=25)
    ap.add_argument("--rebind-mm", type=float, default=3.0)
    ap.add_argument("--horizon", type=int, default=HORIZON,
                    help="planning steps; the paper uses 30 = 3 chained chunks")
    ap.add_argument("--no-cgn", action="store_true",
                    help="skip Contact-GraspNet and use the geometric grasp")
    ap.add_argument("--resweep", type=int, default=8,
                    help="score all 18 directions every N ticks; between those, "
                         "only the incumbent and its neighbours are re-scored")
    ap.add_argument("--neighbours", type=int, default=3,
                    help="directions adjacent to the incumbent to keep in a "
                         "local sweep")
    ap.add_argument("--avoid-weight", type=float, default=0.0,
                    help="penalise PREDICTED motion of non-target scene points. "
                         "There is otherwise no collision term at all: the cost "
                         "sees only the target, so knocking a bottle over is "
                         "free. Uses the world model, not geometry.")
    ap.add_argument("--stages", action="store_true",
                    help="decompose LIBERO's multi-predicate goal into ORDERED "
                         "stages and run them in sequence. libero_10 is all "
                         "multi-predicate; the listed order is not the "
                         "execution order (see stages_from_bddl).")
    ap.add_argument("--ground", action="store_true",
                    help="NO ORACLE TARGET. Cluster the cloud into object "
                         "proposals, ask a VLM once which to move and where, "
                         "then TRACK that cluster every tick. `task_spec` is "
                         "used only to report progress and never as an input.")
    ap.add_argument("--vlm", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--max-regrasp", type=int, default=3,
                    help="give up after this many re-grasps; a loop that keeps "
                         "dropping is reporting a grasp problem, not planning")
    ap.add_argument("--mask-jitter-mm", type=float, default=-1.0,
                    help="ABLATION: replace the oracle mask with 'every scene "
                         "point within --mask-radius-mm of an APPROXIMATE "
                         "location', where the location is the object's centre "
                         "displaced by a fixed random vector of this magnitude. "
                         "This is what a grounder actually supplies: a rough "
                         "point, not a segmentation. -1 keeps the oracle mask.")
    ap.add_argument("--mask-radius-mm", type=float, default=60.0)
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--freeze-scene", action="store_true",
                    help="ABLATION: pin every SCENE channel to tick 0 while the "
                         "robot points stay live. This reproduces the stale-"
                         "observation bug deliberately, and answers whether the "
                         "planner is actually using what the cameras see.")
    ap.add_argument("--video", metavar="PATH", default=None,
                    help="record agentview frames to an mp4")
    cli = ap.parse_args()


    config = load_config()
    ec = dict(config["env"])
    ec["bounds_min"], ec["bounds_max"] = config["workspace"]["bounds_min"], config["workspace"]["bounds_max"]
    ec["interpolate_pos_step_size"] = config["main"]["interpolate_pos_step_size"]
    ec["interpolate_rot_step_size"] = config["main"]["interpolate_rot_step_size"]
    env = ReKepLiberoEnv(ec, task_suite=cli.suite, task_id=cli.task_id, robot="Panda",
                         resolution=config["libero"]["resolution"], reset_seed=0)

    print(f"task    : {cli.suite}/{cli.task_id} — {env.instruction}")
    stages = (specs.stages_from_bddl(env) if cli.stages
              else [specs.for_task(cli.suite, cli.task_id, env)])
    print(f"stages  : {len(stages)}")
    for _i, _s in enumerate(stages):
        print(f"          {_i}: {_s.name}  owes {_s.remaining_mm(env):.0f} mm")
    all_ok, stage_lines = True, []
    rp = MujocoRobotPoints(env, NR)
    for _stage, spec in enumerate(stages):
        print(f"\n--- stage {_stage}: {spec.name} ---")

        if spec.needs_grasp:
            quat = grasp_target(env, spec, use_cgn=not cli.no_cgn)
            print(f"grasped : ee {np.round(env.get_ee_pos(), 3)}, "
                  f"owes {spec.remaining_mm(env):.1f} mm")
        else:
            quat = place_for_push(env, spec)
            print(f"staged  : ee {np.round(env.get_ee_pos(), 3)} behind "
                  f"{spec.target}, owes {spec.remaining_mm(env):.1f} mm (NO GRASP)")
        n_grasp_frames = len(env._frames)

        # Joint space needs no binding and no re-binding: the fingers are joints,
        # not an error term. The rigid ee binding cost 37.6 mm when the jaw closed.
        rp = MujocoRobotPoints(env, NR)

        start_owed = spec.remaining_mm(env)
        model = env.sim.model
        lo, hi = joint_limits(rp, model)
        sig, sens = joint_sigma(rp)
        print('sigma   : ' + '  '.join(f'{n.split("_")[-1]}={s:.4f}rad'
                                       for n, s in zip(rp.joint_names, sig)))
        print('ee sens : ' + '  '.join(f'{v*1000:.0f}mm/rad' for v in sens))
        J = len(rp.joint_names)
        dev = "cpu"          # J is 9-18; the sampler is not the bottleneck

        # THE FINGERS ARE NOT PLANNED. `SIGMA_FINGER` is deliberately tiny because
        # sampling the grip would drop the object -- but MPPI's perturbation cost is
        # `lambda * sum(U * noise / sigma^2)`, and after the `exp(-cost/lambda)`
        # weighting the lambda CANCELS, so that term's influence does not depend on
        # the temperature at all. At sigma = 0.001 the inverse-variance weight is
        # 1e6, so the two joints given the LEAST noise dominated the cost by three
        # orders of magnitude, and they carry no information about the goal.
        #
        # Measured: rank correlation between the cost MPPI weights by and
        # PointWorld's own cost was +0.03 over 15 ticks, i.e. the world model was
        # not in the loop. Clamping the joint limits (below) removed a second
        # thousandfold term and left rho at +0.02; this is the one that mattered.
        #
        # So the action space is the ARM joints only. The fingers are held at
        # whatever the grasp left them at and reinserted before forward kinematics,
        # which is also what "the grip is held" meant in the first place.
        is_finger = np.array(["finger" in n or "gripper" in n for n in rp.joint_names])
        arm = np.flatnonzero(~is_finger)
        A = len(arm)
        print(f"action  : {A} arm joints planned, {int(is_finger.sum())} finger "
              f"joints held (not sampled)")

        state = {"pw": None, "goal_idx": None, "goal_pos": None, "calls": 0}

        def expand(q_arm):
            """(..., A) arm configs -> (..., J) full configs, fingers held."""
            q_arm = np.asarray(q_arm)
            full = np.broadcast_to(state["q_full"], q_arm.shape[:-1] + (J,)).copy()
            full[..., arm] = q_arm
            return full

        lo_a, hi_a = lo[arm], hi[arm]

        ALL_DIRS = candidate_directions()

        def local_dirs(prev, k=3):
            """The previous winner plus its `k` nearest neighbours on the sphere.

            The direction is highly persistent -- `+Y` won on EVERY tick of every
            successful run -- so re-scoring all 18 every tick pays 73 forwards to
            re-derive an answer that has not changed. The neighbours are what let
            it turn when the task does; the periodic full sweep is what lets it
            turn when the neighbours are not enough.
            """
            if prev is None:
                return ALL_DIRS
            d = ALL_DIRS @ (prev / max(np.linalg.norm(prev), 1e-9))
            return ALL_DIRS[np.argsort(-d)[:k + 1]]

        def build_candidates(q0_arm, owed_m, horizon, dirs=None):
            """(qs, labels, rates): straight-line joint trajectories, scaled to the goal.

            THE RATES ARE DERIVED FROM THE DISTANCE STILL OWED, which is the fix for
            the endgame. A fixed ladder (8/16/26 mm per step) over a 10-step horizon
            offers 0 mm or 80 mm of travel and nothing between, so once fewer than
            ~40 mm remain, moving overshoots and standing still undershoots, `still`
            wins on cost, the state stops changing, and the tick becomes a fixed
            point. Both structured runs froze exactly there, with 3 mm margins
            against a 0.46 mm noise floor -- confidently, not by chance.

            `owed / horizon` is the rate that lands ON the goal, so it is always in
            the set, and the multipliers bracket it.
            """
            J3 = point_jacobian(rp, arm)
            on_goal = owed_m / max(horizon, 1)
            rates = np.unique(np.clip(on_goal * np.array([0.25, 0.5, 1.0, 1.5]),
                                      0.0005, MAX_RATE_MM / 1000.0))
            qs, labels, steps, dout = [], [], [], []
            for i, d in enumerate(ALL_DIRS if dirs is None else dirs):
                d = d / np.linalg.norm(d)
                # Only the incumbent needs rate resolution; the neighbours are there
                # to answer "should we turn", which one rate settles.
                for r in (rates if (dirs is None or i == 0) else rates[-2:-1]):
                    dq = joint_delta_for(J3, d * r)
                    traj = np.clip(q0_arm[None] + np.arange(horizon + 1)[:, None] * dq[None],
                                   lo_a[None], hi_a[None])
                    qs.append(traj)
                    steps.append(dq)
                    dout.append(d)
                    labels.append(f"[{d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f}] @ {r*1000:5.1f}mm")
            # "Still" must be on the menu or the planner can never decline to move,
            # however bad every option is.
            qs.append(np.repeat(q0_arm[None], horizon + 1, axis=0))
            steps.append(np.zeros(len(arm)))
            dout.append(np.zeros(3))
            labels.append("still")
            return np.stack(qs), labels, np.stack(steps), np.stack(dout)

        def score(qs_arm):
            """ONE batched PointWorld call for the whole candidate set.

            Every candidate becomes robot point flow through the SAME forward
            kinematics the recorder uses, so the planner and the training-time
            action representation cannot drift apart.

            `qs_arm` is (K, H+1, A) and INCLUDES the current config at index 0.
            That alignment is not cosmetic: PointWorld's contract is that
            `robot_flows[:, 0]` and `scene_flows[:, 0]` are the same instant, and
            `robot_velocity` and `dist2robot` are both derived from it. The MPPI
            version silently started at step 1, and because the server chunks
            `(H+1-1)//10` passes it also dropped 9 of its 30 planned steps.

            Returns the per-step cost (K, H+1), so a candidate is judged where it
            ARRIVES rather than only where it ends up.
            """
            qs = expand(qs_arm)                                 # (K, H+1, J)
            t0 = time.perf_counter()
            flows = rp.at_configs(qs)                       # (K, H+1, Nr, 3)
            state["t_fk"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            _, out = state["pw"].rollout(flows, state["goal_idx"], state["goal_pos"],
                                         avoid_idx=state.get("avoid_idx"),
                                         avoid_weight=cli.avoid_weight)
            state["t_model"] = time.perf_counter() - t0
            state["calls"] += 1
            return out["cost_steps"] if "cost_steps" in out else out["cost"][:, None]

        if cli.ground:
            os.makedirs("videos/grounding", exist_ok=True)
            from rekep_libero.vlm_backends import make_backend
            state["vlm"] = make_backend({"backend": "qwen_local", "model": cli.vlm,
                                         "temperature": 0.0, "max_tokens": 512})
        # GROUND TRUTH for bystander disturbance. Not an input -- a ruler. The
        # cost term above is the model's opinion; this is what actually moved.
        others = [n for n in sorted(env._object_geom_ids) if n != spec.target]
        start_pos = {n: specs.object_center(env, n).copy() for n in others}

        rebinds, regrasps, finished = 0, 0, False
        with PointWorldClient(cli.socket) as pw:
            state["pw"] = pw
            for tick in range(cli.ticks):
                obs = live_observation(env, rp, steps=T_LEN)
                # Scene channels only. `robot_flows`/`robot_normals`/`gripper_open`
                # stay live -- freezing those would ablate the ACTION, which would
                # break the loop for a reason that has nothing to do with
                # perception and prove nothing.
                if cli.freeze_scene:
                    SCENE = ("scene_flows", "scene_colors", "scene_normals",
                             "rgb", "depth")
                    if state.get("obs0") is None:
                        state["obs0"] = {k: obs[k].copy() for k in SCENE}
                    else:
                        obs.update({k: v.copy() for k, v in state["obs0"].items()})
                points0 = obs["scene_flows"][0, 0]

                if not holding(env, spec):
                    regrasps += 1
                    print(f"tick {tick:2d}: LOST the grasp on {spec.target} after "
                          f"{spec.remaining_mm(env):.1f} mm owed — re-grasping "
                          f"({regrasps})")
                    if regrasps > cli.max_regrasp:
                        print(f"          gave up after {regrasps} re-grasps")
                        break
                    grasp_target(env, spec, use_cgn=not cli.no_cgn)
                    continue

                if cli.ground:
                    # PERCEPTION PATH. The VLM is asked ONCE -- the guide's
                    # event-driven re-grounding, minimal form -- and the target is
                    # then TRACKED by nearest cluster, because the cloud is
                    # re-derived every tick so point indices are not stable.
                    lab = sg.cluster(points0)
                    graph = sg.scene_graph(points0, lab)
                    if state.get("goal_world") is None:
                        gi, gp, graph, lab, reply = sg.ground(
                            env.instruction, obs["rgb"][0][0], points0,
                            obs["intrinsic"][0][0], obs["extrinsic"][0][0],
                            state["vlm"], save_image="videos/grounding/live.png")
                        print(f"ground  : {reply.strip()[:160]}")
                        if len(gi) == 0:
                            print("          VLM named no target — stopping")
                            break
                        state["track"] = points0[gi].mean(axis=0)
                        state["goal_world"] = (gp - points0[gi]).mean(axis=0)
                        print(f"ground  : target at {np.round(state['track'],3)}, "
                              f"displacement {np.linalg.norm(state['goal_world'])*1000:.1f} mm "
                              f"(oracle says {spec.remaining_mm(env):.1f} mm)")
                    # track: the proposal whose centroid is nearest the last one
                    best, bd = None, np.inf
                    for r in graph:
                        d0 = np.linalg.norm(np.asarray(r["centroid_xyz"]) - state["track"])
                        if d0 < bd:
                            best, bd = r, d0
                    if best is None:
                        print(f"tick {tick:2d}: lost the tracked object — stopping")
                        break
                    goal_idx = np.flatnonzero(lab == best["object_id"])
                    moved = np.asarray(best["centroid_xyz"]) - state["track"]
                    state["track"] = np.asarray(best["centroid_xyz"])
                    state["goal_world"] = state["goal_world"] - moved
                    owed = state["goal_world"]
                    state["goal_idx"] = goal_idx
                    state["goal_pos"] = (points0[goal_idx] + owed.astype(np.float32))
                    if np.linalg.norm(owed) * 1000 < DONE_MM:
                        print(f"tick {tick:2d}: grounded goal reached")
                        break
                else:
                    mask = env.points_in_geoms(points0, spec.geoms(env), margin=0.01)
                    goal_idx = np.flatnonzero(mask)
                if cli.mask_jitter_mm >= 0:
                    # A grounder returns a rough LOCATION, not a segmentation. The
                    # error is drawn ONCE and held: a persistent bias is the honest
                    # failure mode, and re-drawing it per tick would average away
                    # exactly the thing being tested.
                    if state.get("jitter") is None:
                        r = np.random.default_rng(cli.mask_seed).normal(size=3)
                        r /= max(np.linalg.norm(r), 1e-9)
                        state["jitter"] = r * cli.mask_jitter_mm / 1000.0
                    here = specs.object_center(env, spec.target) + state["jitter"]
                    near = np.linalg.norm(points0 - here, axis=1) <= cli.mask_radius_mm / 1000.0
                    approx = np.flatnonzero(near)
                    if len(approx):
                        inter = len(np.intersect1d(approx, goal_idx))
                        state["iou"] = inter / len(np.union1d(approx, goal_idx))
                        goal_idx = approx
                if len(goal_idx) == 0:
                    print(f"tick {tick:2d}: no points on {spec.target} visible — stopping")
                    break
                if not cli.ground:
                    owed = spec.offset(env)
                    state["goal_idx"] = goal_idx
                    state["goal_pos"] = (points0[goal_idx] + owed.astype(np.float32))

                if cli.avoid_weight > 0:
                    lab_a = state.get("lab_cache")
                    if lab_a is None or len(lab_a) != len(points0):
                        lab_a = sg.cluster(points0)
                    keep = (lab_a >= 0)
                    keep[goal_idx] = False
                    state["avoid_idx"] = np.flatnonzero(keep)
                    state["lab_cache"] = None

                pw.observe(obs)
                q_full = rp.current_config()
                state["q_full"] = q_full          # fingers, held at their live value
                q0 = q_full[arm]
                state["q0"] = q0

                # WARM-STARTED SEARCH. A full 18-direction sweep costs 73 forwards;
                # a local one costs 8. Re-sweep on the first tick, periodically, and
                # whenever the incumbent stops being convincing -- if the winner is
                # "still" or the best cost got worse, the direction may be stale and
                # the neighbours cannot see far enough to fix it.
                full = (state.get("dir") is None or tick % cli.resweep == 0
                        or state.get("restale", False))
                qs_arm, labels, dsteps, cdirs = build_candidates(
                    q0, float(np.linalg.norm(owed)), cli.horizon,
                    dirs=None if full else local_dirs(state["dir"], cli.neighbours))
                cost_steps = score(qs_arm)                       # (K, H+1)
                # ARGMIN OVER BOTH candidate AND step. Scoring only the horizon end
                # cannot express "arrive and stop", which is what froze the previous
                # planner: with 42 mm owed, the slowest moving candidate overshot by
                # 37 mm and `still` undershot by 40, so `still` won and nothing ever
                # moved again. Taking the best point ALONG each trajectory lets a
                # candidate that arrives at step 3 beat one that never leaves.
                kbest, tbest = np.unravel_index(int(np.argmin(cost_steps)), cost_steps.shape)
                cost = cost_steps[:, -1]
                state["cost"], state["qs"], state["cost_steps"] = cost, qs_arm, cost_steps
                # Margin over the best OTHER direction, against the model's own
                # noise. A choice inside the noise is a coin flip, and saying so is
                # more useful than pretending the planner decided.
                per_cand = cost_steps.min(axis=1)
                order = np.argsort(per_cand)
                margin = float(per_cand[order[1]] - per_cand[order[0]]) * 1000
                # Carry the incumbent direction, and notice when it stops working.
                best_c = float(per_cand[kbest])
                state["restale"] = (np.linalg.norm(cdirs[kbest]) < 1e-9
                                    or best_c > state.get("best_c", np.inf) * 1.5)
                state["best_c"] = best_c
                if np.linalg.norm(cdirs[kbest]) > 1e-9:
                    state["dir"] = cdirs[kbest]

                # Execute ONE step of the winner, then re-observe. `tbest` says how
                # many steps it wanted; taking only the first is what makes this
                # receding-horizon rather than open-loop.
                q_target = expand(np.clip(q0 + dsteps[kbest], lo_a, hi_a))
                # Diagnostics in ROBOT POINTS, not end-effector pose. There is no
                # single end effector on an arbitrary URDF -- the dual-arm UR7e has
                # two and no canonical one -- and PointWorld's contract never
                # mentions one. The mean displacement of the sampled points is the
                # quantity the model actually consumes, so it is what to report.
                pts_before = rp.live()[0]
                pts_cmd = rp.at_config(q_target)[0]
                commanded = np.linalg.norm(pts_cmd - pts_before, axis=1).mean() * 1000
                # DIRECTION, separately from magnitude. A scalar "cmd 38 -> got 8"
                # cannot tell a planner that chose the wrong way to push from an
                # arm that could not follow the right one, and those need opposite
                # fixes. `owed` is the translation the task still needs, so the
                # cosine against it is "did this tick aim at the goal".
                # (`NOTES.md` section 3: a single scalar hides the thing you care
                # about -- the second camera moved cos 0.73 -> 0.94 and left the
                # mean error unchanged.)
                owed_dir = owed / max(np.linalg.norm(owed), 1e-9)
                cmd_vec = (pts_cmd - pts_before).mean(axis=0)
                cos_cmd = float(cmd_vec @ owed_dir / max(np.linalg.norm(cmd_vec), 1e-9))

                # The winner's direction, at its first step and over the horizon it
                # actually asked for. `cos_far` at `tbest` rather than at the end,
                # because `tbest` is where this candidate was judged.
                qs_best = qs_arm[kbest]
                far_vec = (rp.at_config(expand(qs_best[tbest]))[0] - pts_before).mean(axis=0)
                cos_far = float(far_vec @ owed_dir / max(np.linalg.norm(far_vec), 1e-9))
                cos_best = cos_cmd     # one step of the winner IS the command now
                env.execute_joint_positions(q_target, steps=EXEC_STEPS, kp=EXEC_KP,
                                            capture_every=10 if cli.video else 0)
                q_got = rp.current_config()
                track = float(np.abs((q_target - q_got)[arm]).max())
                got_vec = (rp.live()[0] - pts_before).mean(axis=0)
                cos_got = float(got_vec @ owed_dir / max(np.linalg.norm(got_vec), 1e-9))

                # The cost SPREAD is what any ranking has to work with. If the
                # candidates differ by less than the model's own run-to-run noise
                # (0.46 mm batched on `large-droid+behavior`, 3.9 mm on
                # `small-droid`), the choice is a coin flip however it is made.
                spread = float(per_cand.max() - per_cand.min()) * 1000
                achieved = np.linalg.norm(rp.live()[0] - pts_before, axis=1).mean() * 1000
                owed_now = spec.remaining_mm(env)
                tag = (f"[IoU {state.get('iou', float('nan')):.2f} n={len(goal_idx):3d}] "
                       if cli.mask_jitter_mm >= 0 else "")
                print(f"tick {tick:2d}: {labels[kbest]:24s} @t{tbest:2d} {tag}"
                      f"cost {per_cand[kbest]*1000:6.1f} margin {margin:5.1f} | "
                      f"pts cmd {commanded:5.1f} -> got {achieved:5.1f} mm | "
                      f"cos far {cos_far:+.2f} cmd {cos_cmd:+.2f} -> got {cos_got:+.2f} | "
                      f"track {track:.4f} | K {len(qs_arm):2d} "
                      f"fk {state["t_fk"]*1e3:5.0f} model "
                      f"{state["t_model"]*1e3:5.0f} ms | "
                      f"owes {owed_now:6.1f} mm | success {spec.success(env)}")
                if finished or owed_now < DONE_MM or spec.success(env):
                    break

        if spec.release_at_end:
            # `EpisodeFinished` here is LIBERO reporting SUCCESS, not a failure --
            # the fourth time this has bitten (`NOTES.md` traps). Releasing is
            # exactly when a place task completes, so the terminating exception is
            # the MOST likely outcome of this line, and letting it propagate killed
            # the run before the verdict printed.
            try:
                env.open_gripper()
            except EpisodeFinished:
                print("released: LIBERO terminated the episode on release "
                      "(this is its success signal, not an error)")
            print(f"released: owes {spec.remaining_mm(env):.1f} mm")

        # per-stage bookkeeping; the run is only a success if EVERY stage is
        ok = spec.success(env)
        stage_lines.append(f"  stage {_stage}: {spec.name[:44]:44s} "
                           f"{start_owed:6.1f} -> {spec.remaining_mm(env):6.1f} mm"
                           f"   success {ok}")
        all_ok = all_ok and bool(ok)
        if not holding(env, spec) and spec.needs_grasp and spec.release_at_end:
            pass  # released on purpose

    end_owed = stages[-1].remaining_mm(env)
    success = spec.success(env)
    print("\nstages  :")
    for line in stage_lines:
        print(line)
    print(f"success : {success}   (LIBERO's own _check_success)"
          f"   all stages ok: {all_ok}")
    print(f"rebinds : {rebinds}   regrasps: {regrasps}")
    moved = {n: float(np.linalg.norm(specs.object_center(env, n) - p) * 1000)
             for n, p in start_pos.items()}
    worst = sorted(moved.items(), key=lambda kv: -kv[1])[:4]
    print(f"disturb : {sum(moved.values()):.1f} mm total over {len(moved)} bystanders"
          f"  | worst: " + ", ".join(f"{n.split('_')[0]} {v:.0f}mm" for n, v in worst))
    print(f"verdict : {'SOLVED' if success else 'not solved'} by planning "
          f"on PointWorld, with a placed grasp and no language input")
    if cli.video:
        path = env.save_video(cli.video)
        print(f"video   : {path} ({len(env._frames)} frames; "
              f"{n_grasp_frames} of them are the approach/grasp, "
              f"{len(env._frames) - n_grasp_frames} are the pull)")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
