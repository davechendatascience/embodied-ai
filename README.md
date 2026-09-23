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
   The current work: a demonstration teacher for every task of libero_spatial, object and goal.
3. **Transfer** -- swap the arm. Not started.

## The skill teacher

A geometry-driven demonstration policy (`screwhead/teacher/`) that reads a task's goal from its
bddl and the scene's geometry from the simulator, and labels any state a student reaches (DAgger).
Status at v21, 50 episodes per task at seed 555 unless stated (`runs/skill_v21`, `runs/skill_l10_v7`,
`runs/skill_l90_v3`):

| suite | at 600 steps | within LIBERO's step limit |
|---|---|---|
| libero_spatial (limit 220) | 498 / 500 | 478 / 500 |
| libero_object (limit 280) | 500 / 500 | 496 / 500 |
| libero_goal (limit 300) | 500 / 500 | 462 / 500 |
| the three | 1498 / 1500 | 1436 / 1500 |
| libero_10 (limit 520; at 800 steps) | 357 / 500 | 354 / 500 |
| libero_90 (limit 400; 20 per task) | 1465 / 1800 | 1459 / 1800 |

The goal is at least 95% on every task within LIBERO's limits. What works and what is missing,
task by task and against LIBERO's own human demonstrations, is in
[`docs/skill_teacher_task_notes.md`](docs/skill_teacher_task_notes.md).

## How the work is kept honest

- `consistency.yaml` -- the design as axioms, definitions, lemmas and branches, verified by
  falsification trials (the consistency-belief MCP server). A design change is proposed and
  verified before it is built.
- `belief.yaml` -- components, contracts and tests; measured evidence is ingested per revision
  (the component-belief MCP server). A trial is stamped with the teacher revision that ran it.
- Both load from git HEAD: a declaration does nothing until it is committed.

## Layout

| path | what |
|---|---|
| `screwhead/sim` | LIBERO environment wrappers, the twist servo, the gripper servo, contacts, scene geometry |
| `screwhead/geometry` | kinematics and frames shared by the teacher and the student |
| `screwhead/teacher` | the skill teacher: task spec, skills, grasp planner, reach screen, affordances |
| `screwhead/student`, `screwhead/scripted` | the VLA and the earlier per-task scripted teachers |
| `tools/` | evaluation (`skill_eval.py`), reports and ledger ingest (`teacher_report.py`), the human-demo survey (`demo_survey.py`) |
| `tests/` | unit tests (`.venv-libero/bin/python -m pytest tests -q`) |
| `docs/` | design notes, the project brief (zh-TW), the per-task notes |

## Changelog

Features by date, newest first. Numbers are measured at the stated commit.

- **2026-09-24**
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
