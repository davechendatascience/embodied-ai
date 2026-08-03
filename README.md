# PointWorld manipulation on LIBERO

A closed-loop manipulation stack built around **PointWorld**, NVIDIA's
action-conditioned 3D world model. PointWorld predicts how a scene's points
move given a robot's motion; it is not a policy and it does not read language.
Everything that turns it into a robot — the planner, the grounding, the
controller — is built here.

Two LIBERO tasks are solved end to end, by planning through the model:

| task | result |
|---|---|
| `libero_goal/0` "open the middle drawer" | 159.9 → 19.9 mm, `_check_success()` **True**, 13–21 ticks |
| `libero_goal/1` "put the bowl on the stove" | 265.4 → 51.7 mm, `_check_success()` **True**, 45 ticks |
| `libero_goal/1` **with no oracle target** (`--ground`) | `_check_success()` **True** — goal from clustering + one VLM call |

Success is always LIBERO's own `_check_success()`. Anything else would be
grading our own homework.

**The honest limit:** with `--ground` the target comes from perception (point-cloud
object proposals plus one Qwen3-VL call) and one task is solved that way; without
it the target comes from the simulator. Articulated tasks still need the oracle.
See *What is still an oracle* below.
Videos of both runs are in `videos/`.

---

## Start here

`HANDOFF.md` is the current state and the next action. `NOTES.md` is the
accumulated findings and the traps that cost real time. Read them in that
order; this file is only the map.

---

## Two environments, never merge them

| venv | stack | for |
|---|---|---|
| `.venv` | mujoco 3.1.6, robosuite 1.4.1, numpy 1.26.4 | ReKep / LIBERO / Contact-GraspNet |
| `.venv-pw` | torch 2.11+cu130, numpy 2.5, spconv, flash-attn | PointWorld |

They cannot share a process, and **neither may reference the other at all** —
not its interpreter, not its paths, not its environment variables. A client
that can launch its server has the server's configuration baked into it, and
then rebuilding one venv breaks the other for no visible reason.

Two interfaces cross, and nothing else:

| | for |
|---|---|
| `.npz` on disk | offline work — recording episodes, scoring them |
| UNIX socket | the control loop — `scripts/pointworld_serve.sh` |

**Orchestration lives in shell**, which belongs to neither venv. Any shell that
imports spconv needs `CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0`.

---

## Running it

```bash
# solve a task (service up, plan, service down)
POINTWORLD_CKPT=small-droid \
  scripts/run_planner.sh --suite libero_goal --task-id 0 --video videos/out.mp4

# the bridge, on its own
scripts/pointworld_serve.sh          # service, in .venv-pw
scripts/run_bridge_test.sh           # up, round-trip test from .venv, down

# score the model against a recorded episode
CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \
  .venv-pw/bin/python tests/run_pointworld_on_episode.py
```

Useful flags on the planner, all of which exist because they answered a
question rather than because they were convenient:

| flag | what it measures |
|---|---|
| `--freeze-scene` | pins the scene to tick 0 — does perception matter at all? |
| `--mask-jitter-mm` | replaces the oracle mask with a rough location — how good must grounding be? |
| `--video PATH` | records the episode; two bugs this session were visible ONLY here |
| `--resweep`, `--neighbours` | how much of the direction sweep to redo per tick |
| `--ground` | no oracle target — clustering + one VLM call, then tracked |
| `--avoid-weight` | penalise predicted motion of non-target points (42% less disturbance) |

`POINTWORLD_CKPT` selects the checkpoint (`small-droid`,
`large-droid+behavior`). `MAX_BATCH` sets the service batch size.

---

## Architecture

```
RGB-D x2 ─► depth ─► scene points ─┐
                                   ├─► PointWorld service (.venv-pw, GPU)
robot URDF/FK ─► robot points ─────┘        observe: DINOv3, once per tick
                                            rollout: PTv3 + heads, per candidate
                                            features: per-point DINOv3, exposed
                                   ▲                    │ cost (K, H+1)
                            UNIX socket                 ▼
                                   └──────────  planner (.venv)
                                                18 directions x rates, argmin
                                                execute 1 step, re-observe
```

The planner searches a **structured, low-dimensional** space: a candidate is a
direction and a rate, and the trajectory is a straight line at constant rate.
Directions become joint deltas by damped least squares on the robot-point
Jacobian, so no end effector is ever named — which is what makes it portable to
the dual-arm robot this targets.

An MPPI version (`pytorch_mppi`, 30 steps × 7 joints sampled with 32 samples)
was tried and is measurably worse: **−1.7 mm of progress in 60 ticks** against
this design's 140. `HANDOFF.md` has the diagnosis.

Timing, steady state, `small-droid`, 2789 points: `observe` 59 ms, candidate FK
9 ms, rollout 108 ms at K=8 — about **5.7 Hz**.

---

## What is still an oracle

`src/rekep_libero/task_spec.py` reads the SIMULATOR, not sensors. On a real
robot none of it exists:

| input | today | on a robot |
|---|---|---|
| **mask** over the target's points | MuJoCo collision geometry | must segment |
| **goal** position | LIBERO's own BDDL predicate | instruction → VLM |
| **axis** of an articulated part | `model.jnt_axis` | not lookup-able |

So the claim is **"the planner can act given a correct target"**, never "the
system does the task". Three measurements this session show how load-bearing
that oracle is, and they are why the grounding work is scoped the way it is:

* freeze the entire visual scene for a whole episode → **still succeeds**
  (0.3 mm difference)
* hand it a mask with **IoU 0.00**, containing zero points of the target →
  **still succeeds**
* a global DINOv3 feature-similarity mask tops out at **0.29–0.67 IoU** even
  when the query is derived from the answer

Together: on these tasks the goal *vector* is doing nearly all the work, and
the mask almost none. That is the opposite of the intuition, and it sets the
order of what to build.

Legitimate by contrast: `env.robot_geom_mask()` removes the arm from the scene
cloud using the robot's own kinematics, which any robot has about itself.

---

## Layout

| path | what |
|---|---|
| `src/pointworld_bridge/` | the service, protocol, client, GPU feature assembly — **ports to a robot unchanged** |
| `src/rekep_libero/` | LIBERO environment, task specs, grasping, grounding |
| `src/robot_points/` | robot surface points as a function of joint config |
| `tests/` | every measurement in `HANDOFF.md` is reproducible from here |
| `scripts/` | setup and orchestration; lifecycle lives in shell |
| `third_party/` | PointWorld, DINOv3, ReKep, KUDA (cloned, not installed) |

---

## Rules this project runs on

Each was paid for, and they are in `NOTES.md` with the evidence:

* **A fallback that silently changes experimental conditions is worse than a
  crash.** Warn loudly or fail.
* **"Flaky model" is a hypothesis of last resort.** Prove reproducibility first.
* **When a fix shrinks an error instead of eliminating it**, the mechanism is
  only partly understood. Chase the residual.
* **A counterfactual needs a control that reproduces the factual**, or a broken
  generator is indistinguishable from a bad model.
* **A single scalar hides the thing you care about.** Score direction and
  location, not just magnitude.
* **Check the checkpoint before characterising the model.** Days of "model
  limitations" once described a misconfiguration.
* **Watch the video.** Two defects this session — stale camera observations and
  an arm standing still for 86% of every tick — moved no number the planner
  printed.
