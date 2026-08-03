"""What the planner needs to know about a task, and nothing more.

PointWorld consumes a MASK over scene points and TARGET POSITIONS for them.
Everything task-specific in a LIBERO rollout collapses into producing those
two arrays, so this module is the whole surface a new task has to implement.

Nearly all of LIBERO turns out to be two shapes:

    move_to(A, B)      A's points end up at B          pick and place
    slide(A, axis, d)  A's points travel along an axis drawers, doors, buttons

Both reduce to "these points, translated by this vector", which is why the
planner itself has no idea which kind of task it is running.

**This is the stub the VLM replaces.** A `TaskSpec` is a target object, a
destination, and a tolerance -- a few structured fields, not the executable
Python with staged sub-goal and path constraints that ReKep's prompt asks a
VLM to emit. Grounding "put the bowl on the stove" to
`move_to("akita_black_bowl_1", "flat_stove_1")` is a far smaller ask than
generating code, and it fails in ways you can inspect. Resolving the spec from
geometry here, rather than from language, keeps a planner bug distinguishable
from a grounding bug.

Success is always LIBERO's own `_check_success()`. Inventing a per-task
criterion would let us grade our own homework.

WHAT IS PRIVILEGED HERE, AND MUST NOT BE MISTAKEN FOR PERCEPTION
----------------------------------------------------------------
Everything in this module reads the SIMULATOR, not sensors. On a real robot
none of it exists. Listing it explicitly because a planner that scores well on
oracle inputs and a planner that works are different claims, and the gap is
invisible from the success rate:

  the MASK        `env.points_in_geoms(points, spec.geoms(env))` selects the
                  target's points using MuJoCo collision geometry. This is the
                  big one -- a real robot has a point cloud and an image, and
                  must segment. Replaceable: PointWorld's scene encoder
                  already computes 128-d DINOv3 features PER POINT (measured
                  std 1.55), which is exactly what a segmentation head or a
                  text/image query would consume. Nothing new has to be built
                  to have features; something has to be built to query them.
  the GOAL        `parsed_problem["goal_state"]` is LIBERO's own answer key.
                  A robot has an instruction instead, which is the VLM's job.
  the AXIS        `model.jnt_axis` gives the drawer's kinematic direction.
                  A robot cannot look this up -- but it may not need to:
                  PointWorld already ranks `-Y` (into the cabinet) worst and
                  `+Y` best by 60 mm (`NOTES.md` section 4), so the constraint
                  direction is DISCOVERABLE by rolling out candidates. That is
                  a real use for a world model and it is untested.
  the POSE        `object_aabb` / `object_center` read geom poses.

Legitimate by contrast: `env.robot_geom_mask()` removes the arm from the scene
cloud using the robot's own kinematics, which any robot has about itself.

So: this module is an ORACLE, used to test the planner in isolation. Treat a
success here as "the planner can act given a correct target", never as "the
system can do the task".
"""

import numpy as np


def object_geoms(env, name):
    """Geom ids for a movable object OR an articulated fixture link.

    `_object_geom_ids` already handles both, including the body-membership
    rule that keeps a cabinet's three drawers apart -- binding by name prefix
    returns nothing for fixtures, and binding by body origin puts all three
    drawers at the same place (`NOTES.md` section 2).
    """
    if name in env._object_geom_ids:
        return list(env._object_geom_ids[name])

    # FALLBACK: any MuJoCo body, by name. `_object_geom_ids` is built from
    # movable objects plus fixture links that have a JOINT, so a STATIC fixture
    # part is invisible to it -- and that is most placement surfaces. Measured
    # on `libero_goal/1`: the stove's `button` has a hinge and is registered,
    # its `burner_plate` -- the actual cook surface the task requires -- is
    # jointless and is not, so nothing could name the goal. The task was aimed
    # at the control knob instead, 155 mm away, and could not have passed.
    #
    # Resolved here rather than in `_cache_robot_geoms` deliberately: adding
    # static bodies to the env's registry would change what `_contacting_objects`
    # and the keypoint binder consider an "object", and those have their own
    # reasons for listing only movables.
    model = env.sim.model
    ids = [g for g in range(model.ngeom)
           if (model.body_id2name(model.geom_bodyid[g]) or "") == name]
    if ids:
        return ids
    raise KeyError(f"{name!r} is neither a registered object nor a body. "
                   f"Registered: {sorted(env._object_geom_ids)}")


def placement_surface(env, region):
    """The body to place ON for a BDDL region, e.g. `flat_stove_1_cook_region`.

    LIBERO names regions semantically (`cook`, `top`, `heating`) while the model
    names bodies structurally (`burner_plate`), so the label often matches no
    body at all -- `_resolve_region` reduces `flat_stove_1_cook_region` to
    `flat_stove_1`, which is not a body with geoms either.

    "On a fixture" means on the broadest upward-facing surface, so when the
    label does not resolve, take the sub-body with the largest HORIZONTAL AREA,
    tie-broken by height. Area, not height: measured on this stove,

        flat_stove_1_burner        top 0.930   361 cm2
        flat_stove_1_burner_plate  top 0.930   338 cm2
        flat_stove_1_button        top 0.965   107 cm2   <- tallest, and wrong

    the control KNOB is the tallest part, so a highest-top-z rule picks it and
    aims the bowl 155 mm from the hob. A thing you place an object on is wide;
    a thing you turn is not.
    """
    fixture, label = _resolve_region(env, region)
    model, data = env.sim.model, env.sim.data

    def extent(body_id):
        """(xy area, top z) of a body, or None when it carries no geoms."""
        gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body_id]
        if not gs:
            return None
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for g in gs:
            c = np.asarray(data.geom_xpos[g])
            e = (np.abs(np.asarray(data.geom_xmat[g]).reshape(3, 3))
                 @ np.abs(np.asarray(model.geom_size[g][:3])))
            lo = np.minimum(lo, c - e)
            hi = np.maximum(hi, c + e)
        return float((hi[0] - lo[0]) * (hi[1] - lo[1])), float(hi[2])

    exact = f"{fixture}_{label}" if label else fixture
    for cand in (region, exact, fixture):
        for i in range(model.nbody):
            if (model.body_id2name(i) or "") == cand and extent(i) is not None:
                return cand

    best, best_key = None, (-np.inf, -np.inf)
    for i in range(model.nbody):
        n = model.body_id2name(i) or ""
        if not n.startswith(fixture):
            continue
        e = extent(i)
        if e is not None and e > best_key:
            best, best_key = n, e
    if best is None:
        raise KeyError(f"no body for region {region!r} (fixture {fixture!r})")
    return best


def object_aabb(env, name):
    """(min, max) world AABB of an object, from its geom origins and sizes.

    `geom_rbound` is deliberately not used: it is the CIRCUMSCRIBED radius and
    has produced a wrong answer three times in this project. `geom_size` with
    the geom's own rotation is loose for meshes but never swallows the table.
    """
    model, data = env.sim.model, env.sim.data
    lo, hi = [], []
    for gid in object_geoms(env, name):
        c = np.asarray(data.geom_xpos[gid])
        R = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
        # half-extent of the rotated box that bounds this geom
        ext = np.abs(R) @ np.abs(np.asarray(model.geom_size[gid][:3]))
        lo.append(c - ext)
        hi.append(c + ext)
    return np.min(lo, axis=0), np.max(hi, axis=0)


def object_center(env, name):
    lo, hi = object_aabb(env, name)
    return (lo + hi) * 0.5


class TaskSpec:
    """A target object, where its points should go, and when to stop.

    `offset(env)` returns the translation STILL OWED, recomputed every tick.
    Returning the remainder rather than a fixed goal is what makes the loop
    receding-horizon: the model under-predicts displacement 3.6x, so a plan
    computed once would be systematically short, and re-deriving the remainder
    each tick absorbs that (`NOTES.md` section 4).
    """

    def __init__(self, name, target, offset_fn, needs_grasp=True,
                 release_at_end=False, approach=(0.0, -1.0, 0.0)):
        self.name = name
        self.target = target
        self._offset = offset_fn
        self.needs_grasp = needs_grasp
        self.release_at_end = release_at_end
        self.approach = np.asarray(approach, dtype=np.float64)

    def geoms(self, env):
        return object_geoms(env, self.target)

    def offset(self, env):
        return np.asarray(self._offset(env), dtype=np.float64)

    def remaining_mm(self, env):
        return float(np.linalg.norm(self.offset(env)) * 1000.0)

    @staticmethod
    def success(env):
        """LIBERO's own criterion. Never invent a substitute for it."""
        try:
            return bool(env.env.env._check_success())
        except Exception:  # noqa: BLE001 - absence is informative, not fatal
            return None


def slide(target, axis, distance, joint=None, stop_qpos=None, approach=(0.0, -1.0, 0.0)):
    """Drag an articulated part `distance` along `axis`.

    SIGN TRAP: `libero_goal/0`'s drawer OPENS by travelling +Y while its slide
    joint runs NEGATIVE, from +0.001 to -0.16 (`NOTES.md` section 5). So the
    joint reading and the world direction disagree, and both are supplied here
    rather than derived from each other. Getting it backwards asks the object
    to stay where it is, and the planner then correctly chooses "still"
    forever -- which looks exactly like a broken planner.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    def offset(env):
        if joint is None or stop_qpos is None:
            return axis * distance
        model, data = env.sim.model, env.sim.data
        q = float(data.qpos[model.jnt_qposadr[model.joint_name2id(joint)]])
        return axis * max(0.0, abs(q - stop_qpos))

    return TaskSpec(f"slide {target}", target, offset, needs_grasp=True,
                    release_at_end=False, approach=approach)


def move_to(target, destination, clearance=0.03, approach=(0.0, 0.0, -1.0)):
    """Carry `target` until it sits `clearance` above `destination`'s top face.

    The goal is a translation of the object's OWN points, not of the gripper,
    which is what makes it expressible to PointWorld at all.
    """
    def offset(env):
        src = object_center(env, target)
        lo_t, hi_t = object_aabb(env, target)
        lo_d, hi_d = object_aabb(env, destination)
        dest = np.array([(lo_d[0] + hi_d[0]) * 0.5,
                         (lo_d[1] + hi_d[1]) * 0.5,
                         hi_d[2] + (hi_t[2] - lo_t[2]) * 0.5 + clearance])
        return dest - src

    return TaskSpec(f"move {target} to {destination}", target, offset,
                    needs_grasp=True, release_at_end=True, approach=approach)


def region_centre(env, region):
    """World-frame centre of a BDDL region, from LIBERO's own definition.

    Regions with `ranges` are axis-aligned boxes in the TABLE's local xy frame
    (`[x0, y0, x1, y1]`); regions without them are attached to a fixture and
    are just that body's extent. Reading the problem file rather than typing
    coordinates keeps the target LIBERO's answer, not ours -- the same reason
    `from_bddl` exists.
    """
    pp = env.env.env.parsed_problem
    spec = pp["regions"].get(region)
    if spec is None or not spec.get("ranges"):
        name, _ = _resolve_region(env, region)
        lo, hi = object_aabb(env, name)
        return (lo + hi) * 0.5
    x0, y0, x1, y1 = spec["ranges"][0]
    model, data = env.sim.model, env.sim.data
    bid = model.body_name2id("table")
    origin = np.asarray(data.xpos[bid])
    return np.array([origin[0] + (x0 + x1) * 0.5,
                     origin[1] + (y0 + y1) * 0.5,
                     origin[2]])


def push_to(target, region):
    """Slide `target` across the table to a region, WITHOUT grasping it.

    THIS IS THE TASK THAT CAN FALSIFY "grasped things follow the hand".
    Everything measured so far -- the drawer opening, the axis probe, the
    perception ablation -- is consistent with a model that believes the target
    simply tracks the gripper, because with the object held that belief is
    RIGHT. A push breaks it in three ways the drawer cannot:

      * contact makes and breaks, so the object moves only while touched;
      * it keeps sliding after the gripper stops, and stops on friction;
      * a gripper that misses by a centimetre moves nothing at all.

    Predicting any of that requires the scene, which is exactly the channel
    `--freeze-scene` showed contributes 0.3 mm on the drawer.

    The push is planar by construction: the goal is a translation in xy at the
    object's own height. Asking for a z component would be asking for a lift,
    which is a different task and not one a push can achieve.
    """
    def offset(env):
        src = object_center(env, target)
        dst = region_centre(env, region)
        return np.array([dst[0] - src[0], dst[1] - src[1], 0.0])

    return TaskSpec(f"push {target} to {region}", target, offset,
                    needs_grasp=False, release_at_end=False, approach=None)


def _resolve_region(env, region):
    """`wooden_cabinet_1_middle_region` -> (fixture name, label).

    LIBERO's goal predicates refer to REGIONS, which are neither objects nor
    geoms. Every region name is `<fixture or object>_<label>_region`, and the
    fixture part must be matched against the scene rather than guessed, since
    labels like `middle` and `top` recur across fixtures.
    """
    stem = region[:-len("_region")] if region.endswith("_region") else region
    pp = env.env.env.parsed_problem
    names = [n for group in list(pp["fixtures"].values()) + list(pp["objects"].values())
             for n in group]
    for name in sorted(names, key=len, reverse=True):
        if stem == name:
            return name, ""
        if stem.startswith(name + "_"):
            return name, stem[len(name) + 1:]
    return stem, ""


def _articulated_part(env, fixture, label):
    """The moving link's registered name and its joint, for `Open <region>`."""
    part = next((k for k in env._object_geom_ids
                 if k.startswith(fixture) and label and label in k), None)
    model = env.sim.model
    joint = next((model.joint_id2name(j) for j in range(model.njnt)
                  if (model.joint_id2name(j) or "").startswith(fixture)
                  and label and label in (model.joint_id2name(j) or "")), None)
    if part is None or joint is None:
        raise KeyError(f"no articulated part for {fixture!r}/{label!r}; "
                       f"known parts {sorted(env._object_geom_ids)}")
    return part, joint


def open_articulated(env, region):
    """`(Open <region>)` -> a spec, with the axis read from MuJoCo.

    The joint's own axis and range are the ground truth for which way "open"
    is, so nothing here hardcodes a direction or a threshold. That also
    dissolves the sign trap in `NOTES.md` section 5 -- `libero_goal/0` travels
    +Y while its qpos runs NEGATIVE, and the axis-times-signed-remainder
    arithmetic gets that right without anyone having to notice it.
    """
    fixture, label = _resolve_region(env, region)
    part, joint = _articulated_part(env, fixture, label)

    model, data = env.sim.model, env.sim.data
    jid = model.joint_name2id(joint)
    adr = model.jnt_qposadr[jid]
    lo, hi = model.jnt_range[jid]
    q_now = float(data.qpos[adr])
    # "Open" is the end of travel further from where it starts (closed).
    stop = lo if abs(q_now - lo) > abs(q_now - hi) else hi
    axis_local = np.asarray(model.jnt_axis[jid], dtype=np.float64)
    bid = int(model.jnt_bodyid[jid])
    axis_world = np.asarray(data.xmat[bid]).reshape(3, 3) @ axis_local

    def offset(env_):
        q = float(env_.sim.data.qpos[adr])
        return axis_world * (stop - q)

    return TaskSpec(f"open {part} (joint {joint} -> {stop:+.3f})", part, offset,
                    needs_grasp=True, release_at_end=False,
                    approach=-_dominant_axis(axis_world))


def close_articulated(env, region):
    """`(Close <region>)` -- the mirror of `open_articulated`.

    Same joint, same axis arithmetic; only the target stop flips to the end of
    travel NEARER where a closed part sits. Sharing the derivation is the point:
    the sign trap in `NOTES.md` section 5 (travel +Y while qpos runs negative)
    is handled once, not twice.
    """
    fixture, label = _resolve_region(env, region)
    part, joint = _articulated_part(env, fixture, label)

    model, data = env.sim.model, env.sim.data
    jid = model.joint_name2id(joint)
    adr = model.jnt_qposadr[jid]
    lo, hi = model.jnt_range[jid]
    q_now = float(data.qpos[adr])
    stop = lo if abs(q_now - lo) < abs(q_now - hi) else hi
    axis_local = np.asarray(model.jnt_axis[jid], dtype=np.float64)
    bid = int(model.jnt_bodyid[jid])
    axis_world = np.asarray(data.xmat[bid]).reshape(3, 3) @ axis_local

    def offset(env_):
        q = float(env_.sim.data.qpos[adr])
        return axis_world * (stop - q)

    return TaskSpec(f"close {part} (joint {joint} -> {stop:+.3f})", part, offset,
                    needs_grasp=True, release_at_end=False,
                    approach=-_dominant_axis(axis_world))


def _one_predicate(env, pred):
    """One BDDL goal predicate -> one TaskSpec."""
    kind = pred[0].lower()
    if kind == "open":
        return open_articulated(env, pred[1])
    if kind == "close":
        return close_articulated(env, pred[1])
    if kind in ("on", "in"):
        # `in` and `on` differ in where the object ENDS UP RESTING, not in what
        # the robot does: both carry it above the destination and release, and
        # gravity settles it onto a surface or into a container. `move_to`
        # already aims `clearance` above the destination's top face, which is
        # exactly the release pose for both.
        return move_to(pred[1], placement_surface(env, pred[2]))
    raise NotImplementedError(
        f"goal predicate {kind!r} is not handled yet (saw {pred})")


def stages_from_bddl(env):
    """LIBERO's goal predicates -> an ORDERED list of TaskSpecs.

    Multi-predicate tasks are the whole of `libero_10`, and the listed order is
    NOT the execution order. `libero_10/3` is

        [['close', 'white_cabinet_1_bottom_region'],
         ['in', 'akita_black_bowl_1', 'white_cabinet_1_bottom_region']]

    i.e. "close the drawer" is listed first, but closing it before putting the
    bowl in makes the second predicate unreachable. The rule is therefore:
    **articulated CLOSE goes last**, because you close a container after
    filling it, and OPEN goes first, because you open it before reaching in.
    Everything else keeps its listed order.

    This is ordering knowledge, not perception -- it belongs to the task
    layer, and a language model asked for a plan would have to supply the same
    thing.
    """
    goals = env.env.env.parsed_problem["goal_state"]
    order = {"open": 0, "close": 2}
    preds = [[str(x) for x in g] for g in goals]
    ranked = sorted(enumerate(preds), key=lambda kv: (order.get(kv[1][0].lower(), 1),
                                                      kv[0]))
    return [_one_predicate(env, p) for _, p in ranked]


def _dominant_axis(v):
    """The signed unit axis `v` points most strongly along."""
    i = int(np.argmax(np.abs(v)))
    out = np.zeros(3)
    out[i] = np.sign(v[i])
    return out


def from_bddl(env):
    """Derive the spec from LIBERO's OWN goal predicate. No hand specification.

    `parsed_problem["goal_state"]` is machine-readable at runtime:

        [['open', 'wooden_cabinet_1_middle_region']]
        [['on', 'akita_black_bowl_1', 'flat_stove_1_cook_region']]

    which is the same information a VLM would have to ground out of the
    instruction, except already correct. Using it means the planner can be run
    on any LIBERO task without anyone writing a target, and it separates two
    questions that would otherwise be tangled: can the planner act, and can a
    VLM say what to act on.
    """
    stages = stages_from_bddl(env)
    if len(stages) != 1:
        raise NotImplementedError(
            f"{len(stages)} goal predicates. Use `stages_from_bddl(env)` and "
            f"run them in order; `from_bddl` is the single-stage shortcut.")
    return stages[0]


# Kept only as an override for experiments; `from_bddl` is the default and
# needs no entry here.
REGISTRY = {
    ("libero_goal", 0): slide(
        "wooden_cabinet_1_cabinet_middle", axis=(0, 1, 0), distance=0.16,
        joint="wooden_cabinet_1_middle_level", stop_qpos=-0.16,
        approach=(0.0, -1.0, 0.0)),
    # NOT the button: LIBERO's predicate is
    # ('on', akita_black_bowl_1, flat_stove_1_cook_region) and the cook surface
    # is `burner_plate`. The button is the control knob, 155 mm away, and the
    # task could not pass while aimed at it. Resolved from the region now.
    ("libero_goal", 1): None,
    # "push the plate to the front of the stove". LIBERO's goal predicate is
    # ('on', plate_1, main_table_stove_front_region), which `from_bddl` would
    # read as a pick-and-place -- and CARRYING the plate would restore exactly
    # the follows-the-hand regime this task exists to break. The override is
    # the whole point, not a shortcut.
    ("libero_goal", 5): push_to("plate_1", "main_table_stove_front_region"),
    ("libero_goal", 2): move_to("wine_bottle_1", "wooden_cabinet_1_cabinet_top"),
    ("libero_object", 0): move_to("alphabet_soup_1", "basket_1"),
    ("libero_object", 1): move_to("cream_cheese_1", "basket_1"),
    ("libero_object", 2): move_to("salad_dressing_1", "basket_1"),
}


def for_task(suite, task_id, env=None):
    """A hand-written override if one exists, otherwise LIBERO's own predicate.

    `from_bddl(env)` was written to make hand-typed targets unnecessary and
    then never called, because this function raised instead of falling back.
    The one hand-written entry that existed for `libero_goal/1` was WRONG --
    aimed at the stove's knob rather than its cook surface -- which is exactly
    the failure mode the fallback exists to prevent.
    """
    key = (suite, task_id)
    if REGISTRY.get(key) is not None:
        return REGISTRY[key]
    if env is not None:
        return from_bddl(env)
    if key not in REGISTRY or REGISTRY[key] is None:
        raise KeyError(
            f"no spec for {suite}/{task_id}. Add one to REGISTRY -- this is "
            f"the table a VLM would produce from the instruction. Known: "
            f"{sorted(REGISTRY)}")
    return REGISTRY[key]
