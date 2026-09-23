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
Status at commit `85e65e9`, 50 episodes per task at seed 555 (`runs/skill_v4`):

| suite | at 600 steps | within LIBERO's step limit |
|---|---|---|
| libero_spatial (limit 220) | 478 / 500 | 357 / 500 |
| libero_object (limit 280) | 496 / 500 | 493 / 500 |
| libero_goal (limit 300) | 449 / 500 | 408 / 500 |
| all | 1423 / 1500 | 1258 / 1500 |

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

- **2026-09-23**
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
