# Cross-embodiment manipulation on LIBERO

Can a policy trained on one robot drive a different one?

LIBERO is a Franka Panda benchmark, and modern VLAs are very good at it —
**VLA-JEPA** (ECCV 2026) reports state-of-the-art numbers there. Beating that
score is not the goal and could not be the contribution. The question here is
narrower and currently unanswered: *the same policy, the same scene, a
different arm.*

The bet is that a policy's **spatial intent** transfers even when its **motor
output** does not, so the pipeline splits them:

```
RGB + language ──► VLA ──► SE(3) intent ──► planner (URDF, joint limits,
                                            collisions) ──► joint commands
```

The second embodiment is a **UR5e**, running LIBERO's Panda scenes with the
Panda gripper so that a pose means the same thing on both arms.

---

## What works, with numbers

Every result is LIBERO's own `_check_success()` or a direct physical
measurement. Reproduce commands are in `NOTES.md` §7–§8.

**The UR5e opens the drawer.** Same scene, same gripper, same waypoints — only
the executor differs:

| arm | executor | drawer |
|---|---|---|
| Panda | OSC (task-space descent) | **141.2 mm — OPENED** |
| UR5e | OSC | 0.0 mm — never touches it |
| UR5e | IK + joint control | **140.4 mm — OPENED** |

cuRobo IK predicted this: it solves **45/45** of the waypoints OSC missed,
including the pre-grasp OSC missed by 83 mm. The poses were always reachable;
the local descent was not. Videos in `videos/`.

**The UR5e scene is provably the Panda's.** 16 bodies compared, worst drift
**0.0000 mm**, robot bases coincident, TCP `[0, 0, 0.097]` on both arms.

**cuRobo agrees with the simulator.** URDF generated from the live MuJoCo model
rather than fetched from a vendor, so forward kinematics matches to
**0.0002 mm** over 200 random configurations on both arms.

**Where the keypose abstraction holds — and where it does not.** Compressing a
dense VLA rollout to sparse SE(3) keyposes and replaying them:

| task | contact | keyposes | result |
|---|---|---|---|
| `libero_object/0` pick-and-place | prehensile | 7–8 (16–19x) | **4/4 pass** |
| `libero_goal/0` open drawer | non-prehensile | 11 (11x) | fails |

The drawer fails even though every keypose is reached within 5 mm: the policy
**hooks** the handle with the gripper open, and a list of poses cannot encode
"still hooked". A grasp attaches the object to the wrist and the waypoints
carry it. So the boundary is *prehensile vs not* — and it is a property of the
solution the policy chose, not of the task: the scripted stack opens the same
drawer by grasping it.

---

## What is NOT done

Stated plainly, because the pieces above are easy to over-read:

* **The headline experiment has not run.** The UR5e drawer success used
  *scripted* waypoints, not policy keyposes. VLA-JEPA → keyposes → UR5e is
  still ahead.
* **cuRobo has not yet planned anything.** IK and FK are verified; collision
  spheres are fitted but unvalidated, and no trajectory has been generated or
  executed.
* **Naive transfer fails and is not yet fully explained.** VLA-JEPA's dense
  deltas on the UR5e: 0/3. Reach transfers (first 75 steps agree within
  ~21 mm, commands at cos 0.999); the contact phase does not.
* **Nothing is at benchmark scale.** Results are 3–4 episodes of single tasks.
  The published protocol is 50 trials × 10 tasks per suite.
* **The obstacle world is privileged**, read from MuJoCo geometry. The
  depth-derived version is designed but unbuilt.

---

## Environments — five, and none may reference another

| venv | stack | for |
|---|---|---|
| `.venv` | mujoco 3.1.6, robosuite 1.4.1, numpy 1.26.4 | LIBERO, ReKep, Contact-GraspNet, all sim-side work |
| `.venv-pw` | torch 2.11+cu130, spconv, flash-attn | PointWorld |
| `.venv-jepa` | torch 2.11+cu130, transformers 4.57, flash-attn | VLA-JEPA policy server |
| `.venv-curobo` | torch 2.13+cu130, cuda-core | cuRobo planning |
| `.venv-gam` | torch 2.13+cu130 | GAM (set up, not used) |

They cannot share a process and **neither may reference the other at all** — not
its interpreter, not its paths. Orchestration lives in shell. What crosses
between them is a file or a socket, nothing else:

| interface | between |
|---|---|
| `.jsonl` traces | policy rollout → keypose extraction → replay |
| `.json` world + URDF | MuJoCo scene → cuRobo |
| websocket | VLA-JEPA policy ↔ LIBERO sim |
| UNIX socket | PointWorld service ↔ planner |

Rebuild any of them with `scripts/setup_*.sh`; each carries its version pins
and the reason for them. Two are load-bearing and non-obvious: `.venv-jepa` is
pinned to torch 2.11 because the only aarch64 flash-attn wheel is built against
it and the policy hardcodes `flash_attention_2`; cuRobo needs
`cuda-core[cu13]`, which its own error message misreports as `cu12`.

---

## Running it

```bash
# the drawer, both arms
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_drawer_open.py \
    --robot Panda --video
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_drawer_open.py \
    --robot UR5e --joint-control --video

# scene invariance, frame math, cuRobo FK
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_ur5e_scene.py
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_frame_math.py
PYTHONPATH=src .venv-curobo/bin/python tests/test_curobo_fk.py \
    --samples /tmp/fk_ur5e.json --urdf $PWD/configs/curobo/ur5e_pandagripper.urdf

# VLA-JEPA rollout (policy server in .venv-jepa, sim in .venv)
scripts/setup_vla_jepa.sh            # once
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/run_vla_jepa_rollout.py \
    --suite libero_object --task-id 0 --episodes 4 --video

# keypose extraction + E1 replay
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_e1_keypose_replay.py \
    --trace <trace.jsonl> --suite libero_object --task-id 0 --mode keypose
```

---

## The experiment matrix

| | arm | intent | executor | isolates |
|---|---|---|---|---|
| E0 | Panda | VLA dense | OSC | harness sanity |
| E1 | Panda | keypose | OSC waypoints | **information lost by the abstraction** — done |
| E2 | UR5e | VLA dense | OSC | naive transfer — done, 0/3 |
| E3a | UR5e | keypose | cuRobo, oracle world | kinematic feasibility |
| E3b | UR5e | keypose | cuRobo, depth world | the headline, perception included |

E1 gates everything: it failed on the drawer and passed on pick-and-place, so
E3 targets `libero_object/0`, not `libero_goal/0`.

---

## Prior work in this repo

**PointWorld** — a closed-loop planner built around NVIDIA's action-conditioned
3D world model, solving `libero_goal/0` and `/1` by planning through the model,
one of them with no oracle target. That line is complete and documented in
`HANDOFF.md`; `src/pointworld_bridge/` ports to a robot unchanged.

**ReKep + Contact-GraspNet** — the scripted grasp-and-pull stack, 3/3
deterministic on the drawer. It is what the UR5e result above is measured
against, and what supplies the waypoints.

---

## Start here

`HANDOFF.md` is the current state and the next action. `NOTES.md` is the
accumulated findings and the traps — §7 is the UR5e scene, §8 is the keypose
boundary and cuRobo. Read them in that order; this file is only the map.

---

## Rules this project runs on

Each was paid for, and each is in `NOTES.md` with the evidence:

* **A fallback that silently changes experimental conditions is worse than a
  crash.** Warn loudly or fail.
* **"Flaky model" is a hypothesis of last resort.** Prove reproducibility first.
* **Copy the conversion, do not restate it.** Paraphrasing a gripper mapping as
  "close when > 0.5" inverted it and read as a broken checkpoint: 0/2 became
  2/3 once the original function was used verbatim.
* **When a fix shrinks an error instead of eliminating it**, the mechanism is
  only partly understood. Chase the residual.
* **Check the checkpoint before characterising the model.** Days of "model
  limitations" once described a misconfiguration.
* **Watch the video.** Defects that moved no printed number were visible there.
