# embodied_ai

An experimental project: a vision-language-action policy (VLA) whose action head can be handed an
arbitrary robot URDF, so that moving a trained policy to another arm is a change of kinematics
rather than a new dataset.

The head emits an embodiment-free body twist and a gripper aperture; fixed, URDF-parameterized
layers (product-of-exponentials kinematics, damped least-squares IK, null-space control) decode it
for the arm at hand. The plan is to train on LIBERO with a Panda and then swap the arm. Before a
transfer result can mean anything, the policy has to be shown to act on what it sees, which is why
most current work is on the demonstrations it learns from.

## Gates

1. **Action head** -- the conversion, Jacobian, IK and null-space layers match a reference to
   machine precision on five arms. Done.
2. **Grounding** -- a policy trained under randomized layouts and start poses beats a copy with its
   cameras zeroed. A DINOv2 patch-token VLA reached 82% on libero_spatial against 4.5% blind.
   The current work: a demonstration teacher for every task of LIBERO's five suites.
3. **Transfer** -- swap the arm. Not started.

## The skill teacher

A geometry-driven demonstration policy (`screwhead/teacher/`) that reads a task's goal from its
bddl and the scene's geometry from the simulator, and labels any state a student reaches (DAgger).
Status at v48 (commit 40c3a81) on LIBERO's protocol -- each task's 50 initial states in order, 600
steps (libero_10 800) -- in `runs/skill_v48`:

| suite | at 600 steps (libero_10 800) | within LIBERO's step limit |
|---|---|---|
| libero_spatial (limit 220) | 495 / 500 | 479 / 500 |
| libero_object (limit 280) | 500 / 500 | 500 / 500 |
| libero_goal (limit 300) | 500 / 500 | 461 / 500 |
| libero_10 (limit 520) | 481 / 500 | 477 / 500 |
| libero_90 (limit 400) | 4468 / 4500 | 4437 / 4500 |
| all 130 tasks | 6444 / 6500 (99.1%) | 6354 / 6500 (97.8%) |

122 of the 130 tasks succeed in at least 48 of 50. Below that: libero_90 32 (35), libero_10 1 and 8 (43),
libero_90 24 (44), libero_spatial 4 (45), libero_10 3, libero_90 5 and 8 (47).

LIBERO scores the first step its goal holds. Scored instead after the teacher has finished -- the
placed object released, at rest, and no more tipped than LIBERO's own human demonstrations leave it
(`CTR-teacher-settled`, `tools/teacher_settled.py`, 20 episodes per task at seed 557) -- the teacher at
v48 settled 2479 of 2600 episodes (95.3%; v41 2454). libero_90 86 settles 13 of 20 though all 20 succeed: a book released at
the crossing pitch can tip or still be moving when scored. Three tasks cannot settle as scored: a book resting upright in the
desk caddy's back compartment lies 3 mm below LIBERO's region box (libero_10 5, libero_90 77), and in
libero_90 89 the humans' demonstrations end holding the book mid-insertion, so the tilt they leave is
not a resting one.

The goal is at least 95% on every task within LIBERO's limits. What works and what is missing,
task by task and against LIBERO's own human demonstrations, is in
[`docs/skill_teacher_task_notes.md`](docs/skill_teacher_task_notes.md).

### Beyond LIBERO's own scenes

LIBERO's 50 initial states per task are near-copies (each object within about 3 cm per axis, the robot's start
fixed), so they show the teacher solves each task's canonical scene, not that it handles variation. Perturbing
LIBERO's scenes after the fact could not be made valid or doable from the task files, and the first attempt
(2026-09-26, fresh soft-reset layouts with widened starts, 634/650) is void: a soft reset dropped objects hosted on
fixtures to the floor (AXM-libero-soft-reset-misplaces-hosted-objects). The teacher push moves to LIBERO-Variations, a
benchmark generated from LIBERO's assets whose scenes and tasks are valid and doable by construction
([`docs/design_libero_variations.md`](docs/design_libero_variations.md)).

LIBERO-Variations v0 (`screwhead/variations/`, the benchmark is `benchmarks/v0.yaml`): the kitchen table without
fixtures; put A on a plate, cookie box or ramekin, or in the basket; A, its target and 1-3 other objects from 16
LIBERO categories, placed inside the Panda's reach map; one episode per generated task, a fresh split seed per
round (`tools/lv_eval.py --seed S --tasks 40`).

| round (split seed) | success | put on | put in | settled |
|---|---|---|---|---|
| 1 (3101) | 40/40 | 23/23 | 17/17 | 37/40 |
| 2 (3102) | 40/40 | 20/20 | 20/20 | 40/40 |
| 3 (3103) | 40/40 | 27/27 | 13/13 | 38/40 |

Settled: after LIBERO's first success the teacher lets go and retreats; the goal still holds, the object is at rest
and upright within 10 deg. The five unsettled are all tall bottles (bbq sauce, milk, salad dressing) put on a ramekin
or the cookie box and ending on their side -- four still scored as successes by LIBERO's On. One bottle slipped
from a pinch high on its neck during the lift and was re-picked lying; one stood on the 62 mm cookie box and
toppled after release.

## Changelog

Features by date, newest first. Numbers are measured at the stated commit.

- **2026-09-27** -- LIBERO-Variations v0 built and scored (39492ec, e0b6861). A top-down reach map for the Panda
  over the kitchen table (1279 of 3111 grid points; jaw lines served by either direction, as the grasp planner
  offers both -- fixed tool yaws confined it to one side, 690 points), a shape catalog measured from collision
  geometry, a generator that keeps a task only if its scene, preconditions and instruction check out at rest, and an
  evaluation that regenerates each task from the metadata and a seed (reproduced by digest, 6 of 6, three
  generations). The teacher: 120/120 over three fresh rounds, 115/120 settled (put on 65/70, put in 50/50) -- the
  unsettled are tall bottles on small supports. Building it found that the model fingerprint left out the model's
  names and options, so two objects of a category could have swapped bodies unseen (fixed). Theory: four branches
  (action space, scene validity, doable tasks, unambiguous instructions), three proven so far, and an axiom that a
  MuJoCo contact depends on its pair alone (AXM-mujoco-contacts-pairwise, d8eaa16).

- **2026-09-27** -- Theory and code cleanup. Pruned the dormant optimization teacher (search on the task
  loss, verdicts, state restore; 7 branches, a lemma, a component with 5 contracts, 20 files), SimArm's lean
  step, the unused intended-contact screen and the retired randomized set (b312e00, 6a0e0de). A liveness audit
  found ten skill-teacher branches stating designs the code does not implement as stated; each was restated to
  what the code does, its shortfalls named rather than cited as met (d71552c, 456388c), and two live but
  undeclared designs were declared (clear-before-opening, trial provenance). Two defects fixed on the way:
  skill_eval recorded the wrong seed and episode index under --split, so no split episode could be re-run from its
  trial (34d1df5; now 4 of 4 re-run exactly), and the teacher could read the previous episode's released and
  last-held objects at an episode's first step (bce70d4; episode-identical on 30 episodes). Ledger: 47 proven
  (was 43), 23 open (was 34), 4 refuted (was 8) -- the four refuted are lemmas of the planned regression planner.
- **2026-09-26** -- The Panda VLA paused; the teacher pushed to widened randomized starts. P0 (Qwen3-VL-2B on
  LIBERO's demonstrations) stopped at step 5000, before any evaluation (docs/vla_comparisons.md). VLA-JEPA
  (lerobot/VLA-JEPA-LIBERO) on libero_spatial: 476/500 from LIBERO's starts, 140/200 (tasks 0-3) from this
  project's randomized set, 72% of its failures precision misses (the right bowl set down 3.1-5.8 cm from the
  plate's centre, where every success ended within 3.0 cm). The comparison protocol is a checklist with measured
  contracts (reproduction, placement, the checkpoint training ended with); pairs recorded from the teacher are
  declared equivalent to LIBERO's for training (BRN-vla-teacher-pairs-as-demo-pairs, proven). The teacher at 1.5x
  the randomized set's range: 450/475 on libero_spatial's other nine tasks, task 4 (the drawer) 50/164, as poor
  with its own gripper as with the VLA's two-valued one (11 vs 10 of 30 paired starts). A first round from fresh
  LIBERO layouts with widened starts (634/650) was void -- a soft reset drops fixture-hosted objects to the floor
  (AXM-libero-soft-reset-misplaces-hosted-objects) -- and the approach was dropped for LIBERO-Variations, a generated
  benchmark (docs/design_libero_variations.md).
- **2026-09-26** -- Servo transfer, the teacher on other arms (docs/design_servo_transfer.md). Another arm
  now starts in the Panda's scene: its tool at the pose LIBERO's Panda was recorded at, the fixtures drawn
  from the Panda's random stream (2c371d4; before, robosuite's own start poses knocked objects on 42 of 160
  resets). On the 40 benchmark tasks at init 0, with the Panda gripper: UR5e 37, IIWA 38, Kinova3 35, Jaco 36
  of 40 (146 of 160). The reach screen no longer probes fixed heights over the pre-grasp (562db34): those
  poses sat at the edge of the Kinova3's and the Jaco's reach and rejected every handle grasp of the moka
  pot -> 149 of 160; the Panda's protocol sweep scores 6444 of 6500, as v48 did (10 episodes differ). A
  screen graded by intended contact was measured (146 -> 144) and withdrawn. The UR5e starts in its home
  family, the shoulder panned to the far side of the base (4d3c216; calibrated on held-out libero_90 tasks,
  85 -> 90 of 90) -> 152 of 160. Over initial states 0-4 of the same 40 tasks (200 episodes per arm,
  4d3c216): UR5e 195 (184 before the home family), IIWA 186, Kinova3 180, Jaco 179; never solved: IIWA
  libero_goal 7, Kinova3 and Jaco libero_spatial 4 and libero_10 9. **Cross-embodiment tuning of the
  teacher is paused here.** Next: the Panda VLA on Qwen3-VL-2B, trained on LIBERO's human demonstrations
  replayed through the servo (docs/design_VLA_qwen.md).
- **2026-09-25** -- LIBERO's protocol for all 130 tasks (each task's 50 initial states in order):
  5746 of 6500 at v26 -> 6413 at v41 -> 6433 at v44 -> 6444 at v48; settled 2150 -> 2454 -> 2479 of 2600 (v26 -> v41 -> v48).
  - v48: the moka pot is held by the top of its handle -- the handle finder had taken the spout, 49 mm out
    on the other side, for a handle (libero_10 8: 39 -> 43; libero_10 9 49 -> 48); a laid book crosses
    the shelf's face pitched up to 0.5 rad, the humans' 23 deg median, so the hand stays high over the
    book standing in front (libero_90 86: 42 -> 50). The reach screen keeps the step's IK and contact
    answers, keyed on the whole simulator state: the teacher's planning had doubled since v41
    (libero_10 9: 17-20 -> 36-41 ms per step), and the sweep's libero_90 went 2261 -> 1417 s,
    episodes identical (60 of 60 compared step for step against v44).
  - v44: the face pinch is also offered leaned 0.1-0.2 rad toward the robot's base, as the humans
    take a bottle standing close in (libero_90 62: 46 -> 50); the microwave door's swing table keeps
    only rows backed by the mode's demonstrations (libero_90 33 46 -> 50, libero_10 9 44 -> 49); a
    roofed entry goes in shallow only where the grasp was screened there (v42: libero_90 42 43 -> 49).
    No task lost an episode against v42.
  - Grasps leaned off the vertical where no upright one passes the reach screen, as LIBERO's humans
    lean theirs (a rim pinch about the rim's tangent, a handle pinch about its jaw); after a drawer is
    hooked the hand rises clear of its bar before anything else; the pick turns its wrist before it
    comes down, not on the way (libero_90 8, open the drawer and put the bowl in it: 12 -> 47 of 50).
  - A roofed target with no spot open from above is entered only 2 cm past its face, and the grasp is
    screened where that entry lets go; the yellow-and-white mug is also held by its handle, offered
    after its rim (libero_10 9, the mug into the microwave: 0 -> 43 of 50).
  - The hook takes the jaw direction the wrist can reach (libero_90 23: 33 -> 50); a face pinch that
    would touch a neighbour is turned up to 30 deg off square, as the humans turn theirs on the cream
    cheese (libero_10 1: 37 -> 43).
  - Regressions at v41, open: libero_90 32 43 -> 35 and 42 49 -> 43, libero_spatial 4 48 -> 45.
  - A container's open test is read as mid-drag only while its drive is running
    (BRN-open-precondition-reads-the-running-drive; libero_90 2 42 -> 49, 24 41 -> 44, libero_10 3
    45 -> 47).
  - Roofed targets entered past loose objects (libero_90 42 29 -> 49); a carried object turned to fit;
    an object laid along a tilted region (libero_90 86 37 -> 42).
  - A standing object too tall for its region is tipped over into it (libero_90 32 1 -> 43); into a
    region the object is aligned before it is lowered (the caddy: libero_90 75, 80, 83 46/39/40 -> 50);
    the leave from under a roof counts only what lies below the leg's start (libero_90 35 back to 50).
  - A laid book goes in steep, front first, as the humans do (libero_90 89 0 -> 50); the hand leaves a
    roof level before it rises (libero_90 23 13 -> 33).
  - DEF-skill-contract's start gate: a skill does not start while the hand is still on what an earlier
    one released (libero_10 2 26 -> 50).
  - Drawer, caddy, moka-pot and shelf-exit laws (libero_10 8 0 -> 39, libero_90 5 11 -> 47, the caddy
    tasks 75, 80, 81, 83 21-24 -> 39-50).
  - Roofed targets entered level, as the humans do (libero_90 40, 88 0 -> 50, 42 0 -> 30, 86 0 -> 37);
    the microwave's door swung without a grasp (libero_90 35 0 -> 50, 33 27 -> 46).
- **2026-09-24**
  - Success is also scored after the teacher has finished (`CTR-teacher-settled`): on episode 0 of all
    130 tasks, 93 of the 104 placed objects were still held, falling or rocking when LIBERO scored
    them. Three fixes it found: a level object is let go once it rests on the physical surface under
    it, not LIBERO's region box, which can lie below it (libero_object 2 and 4, libero_90 48, 69, 70:
    pressed into the floor to the horizon, 0 -> 10 of 10 settled); the drop point tests the footprint
    as it will be carried -- turned, centred on its box -- where the old test's differs; a tight fit is
    squared rather than tolerated at 15 deg (books into the caddy: libero_90 73 8 -> 19, 78 2 -> 18 of
    20; libero_90 1505 -> 1537, the 30 benchmark tasks unchanged, libero_10 357 -> 356).
  - A support is moved before anything is set on it (libero_90 63 and 64, stack a bowl on another and
    put them in the tray: 0 -> 20 of 20 each; libero_90 1465 -> 1505).
  - The stamp-monitor MCP server is registered (read-only audit of both ledgers).
  - A held object is turned about the vertical during the carry when that makes its footprint fit the
    target region (libero_90's book into the caddy's front compartment: 73, 78, 81 0 -> 9, 4, 8 of 20;
    libero_90 1444 -> 1465).
  - libero_90 1346 -> 1444 of 1800: frying pans and moka pots held by the handle, as the humans hold
    them (pan tasks 18, 21, 41: 4-5 -> 20 of 20; libero_10 2: 24 -> 32 of 50); a drawer with another's
    bar over it opened with a front hook, as the humans do with no grasp (6: 0 -> 20); a container
    nothing goes into is closed before others are opened (23: 0 -> 6); an object whose collision
    geoms sit on a child body is found (62: 0 -> 20).
  - libero_10 182 -> 349 of 500: a container the goal also closes is filled first and then closed
    (a goal-less open step had the teacher reaching for an open drawer's handle); openness is read
    from the joint; a place is done only once the object is let go, and a released object counts as
    delivered only within 3 cm above its target; drop points are open over the whole footprint; the
    grasp screen probes the arm where it lets go; sliding drawers are closed by pressing the bar, as
    the humans do (BRN-push-closes-sliding-drawer). Tasks 0 and 7 0 -> 50, 3 0 -> 42.
  - First libero_90 baseline: 1346 of 1800 (20 per task); 61 of 90 tasks at 20 of 20, 19 at 0-4.
  - skill_eval runs a rolling pool instead of waves (a libero_90 sweep had kept 3 of 10 cores busy).
  - Human-demo survey over libero_10 and libero_90: 5000 demos, 4985 meeting their goal at the last
    recorded state.
  - Rim pinches halfway between the palm's depth and the usual one, before the usual (libero_spatial 4
    48 -> 49; 1498 -> 1499 of 1500).
  - libero_goal 3 the humans' way: the drawer is opened first and the bowl picked beside it when it
    keeps its rim pinch with the drawer open (DEF-skill-contract's clear_of, which the code had
    approximated by a footprint test); a held object under an overhang moves level to a clear
    column before it rises (BRN-lift-leaves-overhang). Goal 3 within 300 steps 0 -> 12 of 50
    (1423 -> 1435).
  - The grasp approach is probed every 2 cm: the hand's side wings slipped between the probes into a
    wine-rack bar (1495 -> 1498).
  - The servo scales its lead uniformly (BRN-servo-lead-scaled-uniformly) by default: clipped per
    joint, a fast descent tilted the tool 18.6 deg, the elbow reached its stop and the arm stood
    still for 550 steps (1482 -> 1495).
  - Execution is anchored at every reset by default: robosuite's finger target survived resets, so
    an episode depended on the ones its process ran before. `skill_eval --interleave N` runs a
    task's episodes on N workers, episode for episode identical to one worker.
  - Clearing spots are searched on a 12 x 32 grid (libero_goal 3 41 -> 50 of 50).
  - The support ray starts above an object's bottom: started below, it began inside the table and
    met its underside 49 mm down, and the cream cheese's fingertips went 4.3 mm into the table
    (libero_goal 6 within 300 steps 36 -> 50).
- **2026-09-23**
  - Transport at human speeds (lift/lower 0.15, carry 0.18 m/s) and a 10 cm/s push (1466 -> 1475;
    within LIBERO's limits 1343 -> 1394).
  - Carry and crossing heights clear what actually stands near the path: the room's walls had
    put every carry at the cap (1433 -> 1466 of 1500; within LIBERO's limits 1258 -> 1343).
  - Drawer hook: the top drawer is opened with open jaws from above, as the human demos do
    (opens in ~55 steps instead of ~290; libero_goal 3: 9 -> 18 of 50).
  - Push skill: a dish is pushed with open jaws caging its rim, as LIBERO's human demos do
    (libero_goal 5: 3 -> 50 of 50).
  - Execution bounds the commanded acceleration at 0.5 m/s^2 while the jaws hold something, 2
    otherwise (libero_goal 9, the bottle on the rack: 1 -> 50 of 50).
  - Human-demo survey over all 1500 LIBERO demonstrations of the three suites, with each demo's
    own fixture poses; per-task notes comparing them with the teacher.
  - Mid-air drops and release heights measured on every episode (`grip_watch.py`); drops
    turned out to be hidden by success on the bowl tasks.
  - The retreat from a driven handle stops where the handle clears the fingers.
  - Refusals: a precondition with no witness ends the episode and is counted apart from failures.
  - Evaluations stamp the teacher revision when they start; sweeps run in bounded waves.
- **2026-09-22** -- design pass for a trained teacher (task loss, RL objective, optimization
  teacher); RL abandoned after a PPO pilot; parameters must come from a declared source.
- **2026-09-21** -- the skill teacher's foundations: affordances per object category, grasp
  mechanics measured in the simulator, gripper and support scanned rather than declared, smooth
  execution (a joint ramp and a bounded twist change), planner designs by regression over skill
  contracts.
- **2026-09-18** -- held-out tasks: train on seven, evaluate three never seen.
- **2026-09-17** -- gripper as a classification over program apertures; DAgger pipeline options.
- **2026-09-15/16** -- gripper servo (the policy names an aperture); patch-token VLA with DAgger,
  82% under randomization against 4.5% blind; one execution path from demonstration to
  evaluation.
- **2026-09-14** -- randomized layouts and start poses; per-task scripted demonstrations (294/300
  on libero_spatial); DAgger distillation; the distilled VLA beats its blind control.
- **2026-09-07/08** -- action-head layers verified across arms and grippers; embodiment-free
  proprioception; backbones compared (SigLIP, Qwen2-VL, DINOv2); the finding that LIBERO's fixed
  initial states do not require vision.
- **2026-08-03 .. 08-20** -- the previous project: cross-embodiment by keypose following
  (UR5e, Robotiq and Rethink grippers on LIBERO) and grounding probes on pi-0.5 and GR00T. Its
  scaffolding was removed on 2026-09-14.
