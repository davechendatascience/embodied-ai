# LIBERO-Variations: a generated benchmark

Status: design, 2026-09-26. Nothing built yet.

## Why

LIBERO's own suites fix each task's scene: across a task's 50 initial states every object moves within about 3 cm
per axis and the robot always starts in the same pose. Success on them shows a policy solves each task's canonical
scene, not that it handles variation. Perturbing LIBERO's scenes after the fact failed on theory: whether a
perturbed scene is still valid (its relations, its instruction's meaning) and still doable could not be decided from
the task files -- LIBERO's own evaluator finds half of spatial 0's table-region relations false in its stored
initial state 0, and a soft reset drops fixture-hosted objects on the floor
(AXM-libero-soft-reset-misplaces-hosted-objects).

When we generate the task ourselves we own its ground truth, so validity, meaning and doability can hold by
construction instead of being inferred.

## What it is

- **A module** (`screwhead/variations/`) that assembles scenes from LIBERO's assets -- its fixtures (stove,
  microwave, cabinets, fridge, faucet), groceries, bowls, plates, mugs, books, caddy, trays, shelves -- plans a task
  that can be done in each, writes LIBERO's task-file format, and builds it with LIBERO's environment, so success is
  LIBERO's predicate evaluator on a task file of our own.
- **A metadata file** that *is* the benchmark: its name and version, the generator's revision, the scene families,
  the object pool, the task templates and their weights, the action space, the seeds of each split, episodes per
  task and step limits. Nothing else is stored: every scene and task regenerates from the metadata. Two runs of one
  metadata agree task for task (measured, by a digest per generated task).

## Generation, and what holds by construction

1. **Action space.** A region of the table, defined once per robot from its kinematics (the tool reaching every
   point in it with the approaches the teacher uses), stored with the metadata. Placement only checks containment --
   no inverse kinematics per task.
2. **Scene.** A scene family (table type, a subset of fixtures at their places), then objects drawn from the pool and
   placed, by LIBERO's samplers, in regions we define inside the action space, with clearance between footprints.
   Checked after the scene settles: every object at rest where it was placed, every initial relation of our own
   task file true.
3. **Task.** A template (put A on B; put A in C; stack A on B; open or close D; turn on E; two of these in sequence),
   instantiated only with objects that meet its preconditions: A movable and graspable (its category has an
   affordance entry), B a support or C a container with room for A's footprint, every object and target point
   inside the action space, the goal false at the start.
4. **Instruction.** Written from the goal and the scene as generated: each object named by its category, and where
   two of one category are present, by the relation that singles it out as placed ("the left bowl", "the bowl on
   the plate") -- correct because the generator knows where everything is.

"Doable" here means those preconditions hold; it is necessary, not sufficient. A task the teacher fails with its
preconditions true is evidence against the teacher, or against the generator's notion of doable -- deciding which
is the diagnosis step below.

## Evaluation and co-evolution

- Rounds on fresh seeds, never reused for scoring; the bar is the teacher's: every task template and scene family
  above 95% in three rounds in a row.
- Each failure is diagnosed in three layers: a premise (an axiom about LIBERO or MuJoCo) measured false; the
  implementation not doing what its branch says; or the design -- the teacher's skill failing with its
  preconditions true (fix the teacher) or the task not doable after all (fix the generator, bump the metadata
  version). Results are tied to a metadata version; old ones are kept against theirs.

## First version

One family -- the kitchen tabletop without fixtures -- with LIBERO's bowls, plates, mugs and groceries, and the
templates put A on B, put A in C (basket, tray, bowl), stack A on B. Then fixtures (drawers, microwave, stove) and
two-step templates.

## Decisions (2026-09-27, the user's)

- **Action space: a top-down reach map.** A grid over the table, computed once per robot and stored with the
  metadata by digest: a point is in when the tool, pointing down, reaches it at each of a declared set of yaws and
  heights (table + 2 cm to + 25 cm) as a reachable pose (DEF-reachable-pose), with a 5 cm border removed. Front
  approaches join when fixtures do.
- **Capacity: category and measured fit.** The affordance table (screwhead/teacher/affordances.yaml) decides which
  categories can hold or support which; the objects' footprints, measured from the compiled model, decide whether
  this object fits, with a clearance margin.
- **LIBERO's standard suites** stay as a secondary reference, run now and then so the VLA's numbers remain
  comparable with published ones; LIBERO-Variations is the target.

## Open

- The yaws and heights of the reach map; the clearance margins; how many objects per scene in v0.
