# Start here

Current state and the next action. `NOTES.md` holds the long-form findings and
the full experimental record; this file is only what a decision depends on.

Two lines of work have run in this repo:

| | status |
|---|---|
| **Cross-embodiment transfer of a frozen VLA** (branch `gripper-transfer`) | current |
| **PointWorld + ReKep** world-model planning | parked, task solved |

---

# CURRENT (2026-08-04) — cross-embodiment VLA transfer

**Tomorrow: try a different VLA.** Rationale below — the evidence says this
checkpoint, not the method, is the binding constraint.

Branch `gripper-transfer`, pushed, 3 commits (`f3679e3`, `c495ed6`, `eb0e773`).
Method write-up: `METHOD.zh-TW.md`.
PR: https://github.com/davechendatascience/embodied-ai/pull/new/gripper-transfer

## One line

Swapping the **arm** transfers (11/12, zero parameters). Swapping the **gripper**
transfers weakly — **1/3** — and only with the wrist-camera transform restored.
Report it; do not build on it.

## Results

`libero_spatial/0`, all rows with start-pose alignment:

| config | result |
|---|---|
| UR5e + PandaGripper (control) | **3/3** |
| UR5e + Robotiq85 | 0/3 |
| UR5e + Robotiq85 + vector camera alignment | **1/3** |
| UR5e + RethinkGripper | 0/3 |
| UR5e + RethinkGripper + vector camera alignment | **1/3** |
| any of the above + learned corrector | 0/3 |
| Robotiq85 + Panda fingertip friction | 0/3 |
| RethinkGripper + data-derived constant action bias | 0/3 |

`libero_object/0`, Robotiq85 + camera alignment: 0/3.

Earlier, established: arm swap with the same gripper is 11/12 across four
suites, from start-pose alignment alone — no planner, no keypose abstraction, no
learned adapter. The apparent embodiment gap was **configuration OOD, not
appearance OOD**: every LIBERO demo starts from the same home pose, so the policy
has zero training signal for any other initial configuration.

## Established

**The policy is a visual servo that ignores proprioception** (state swap between
arms → cos 1.000). It drives the *camera*; the TCP follows only because
camera→TCP was rigid in training. In the camera frame:

| config | camera→TCP (mm) |
|---|---|
| Panda + PandaGripper | `[0, -50, -97]` |
| UR5e + PandaGripper | `[0, -50, -97]` ← identical; this is *why* the arm swap works |
| UR5e + RethinkGripper | `[0, -50, -109]` |
| UR5e + Robotiq85 | `[0, -50, -145]` |

A gripper swap perturbs **depth only**. Restoring the whole vector (not its
length) is the only intervention that has changed an outcome.

**Geometry is not what makes a gripper transfer — appearance is.** The Rethink is
geometrically closest (12 mm) and behaviourally furthest: identity baseline
cos **−0.1158**, gripper-channel agreement **44.4%**, against Robotiq85's 0.9498
and 95.4%.

**The bottleneck is the gripper's own appearance in the wrist image.** Pixels
differing from the Panda view: 45.8% unaligned, 33.9% aligned. Gripper occupancy
26.8% (Panda) → 35.5% (Robotiq85) → 16.4% (aligned) — pushing the camera forward
to fix the transform pushes the fingers out of frame. **The transform and the
appearance cannot both be matched**, because the gripper's physical extent
differs.

## Ruled out by measurement — do not re-litigate

- **Friction.** robosuite's PandaGripper has a dedicated fingertip pad at sliding
  friction 2.0; Robotiq85 is 1.0, Rethink 0.0. Bowl is 0.95, condim 3, MuJoCo
  takes the max — a Panda grips this object with **twice** the friction of any
  replacement. An asset property, not an embodiment one. Equalising it: **0/3**,
  worse than the 1/3 without.
- **Gripper-channel pass-through.** Disagreement rises 1/450 (0.2%) → 22/480
  (4.6%). Predicting the channel lifts offline agreement 44.4% → 97.7% on
  Rethink and still gives 0/3.
- **Command sign convention.** `+1` closes all three, measured by true fingertip
  separation. `gripper_qpos` is **not** a comparable width proxy across grippers
  (Panda reports finger displacement, Robotiq85 a driver angle where larger means
  *more closed*) — that proxy wrongly reported the convention as inverted.
- **Constant action bias.** Derived from the pair data, not swept: median
  (source − target) is ~0 for Panda (a sanity check that passes), small for
  Robotiq85, `[-0.138, -0.264, -0.110]` for Rethink. 0/3 at full scale and at
  0.25/0.5/0.75, including on the one init that *is* winnable with camera
  alignment. Only 41% of the needed correction is systematic, and the right
  offset is task- and phase-dependent — a constant cannot express it.
- **Language steering.** Same image and state, a *completely wrong* instruction
  ("close the microwave door") rotates the action by only cos 0.80; a gripper
  hint moves it 0.028 against a mean command magnitude of 0.63 (~3× sampling
  noise). Prompt injection is a nudge, not a steering channel.

## Two bugs that produced false results — watch for the pattern

1. **`env.reset()` rebuilds the model from XML.** A `cam_pos` written before the
   episode loop is silently discarded on every reset. Every `--align-camera`
   number produced before `c495ed6` ran with alignment *inactive* and looked like
   a legitimate failure. Model mutations belong **inside** the loop, after reset.
2. **Matching a vector's length is not matching the vector.** The first camera
   fix rescaled the whole offset, repairing depth while sliding the grasp point
   14 mm across the image plane.

Also: trajectory filenames now carry the gripper. They did not, so a Robotiq85
run overwrote the PandaGripper trajectory for the same suite/init — and
`--align-start` reads those files, so a later run would have aligned to a
**failed** rollout's start pose.

## The recurring lesson

**Offline action agreement does not predict closed-loop success. Three times
now.** The corrector improves every offline metric (Rethink frac<0.9
72.3% → 16.9%) and is worse in rollout every time. Stop using offline cosine as
the decision metric.

Diagnosed cause: pairs are labelled at **matched** poses; rollouts occupy
**drifted** ones, so the correction is applied off-manifold — right sign, wrong
scale (overshoots by 32 mm and stops short of the object).

## Why a different VLA is the right next move

Three independent signs that this checkpoint is unusually overfit, so the method
is being judged against a weak instrument:

- It ignores proprioception entirely (cos 1.000 under a state swap).
- It barely attends to language (wrong instruction → cos 0.80).
- Its 11/12 arm-transfer result came from start-pose alignment alone, i.e. it had
  **zero** training signal for any initial configuration but one.

A policy that reads state and language would let interface-level interventions
actually bite. When swapping, re-run in this order, cheapest first:
(1) proprioception sensitivity, (2) instruction sensitivity, (3) start-pose
alignment, (4) gripper swap. The first two take minutes and decide whether the
rest is worth running.

## Then, in order

1. **DAgger relabelling** — the one untried fix that addresses the diagnosed
   cause: relabel on rollout-visited states, not another round of matched-pose
   pairs. Everything else has been tried.
2. **Visual prompting conditioned on gripper geometry** (MOKA / PIVOT /
   Set-of-Mark; ReKep is already in `third_party/`). Promising *because* vision
   dominates — but the action head saw only unmarked LIBERO frames, so it trades
   one OOD for another. Cheap pre-test: draw a mark, check the action changes at
   all.
3. **Open item:** `diag_grasp_basin.py` reports the Robotiq85 failing at every
   lateral and vertical offset, yet the same gripper lifts the bowl 45.3 mm under
   the policy. That probe measures the protocol, not the gripper. Fix or discard
   — **do not cite it**.
4. **Written but never run:** `diag_close_time.py` (closure timing vs the
   policy's lift budget).
5. **Older open item:** `libero_goal` init7, the single failure in the original
   11/12.

## Data and tooling

    pairs/panda/       450 pairs, no `geom` column (reads as Panda geometry = a true zero)
    pairs/robotiq85/   480 pairs, geom [145.0, 153.4] mm
    pairs/rethink/     480 pairs, geom [109.0, 119.9] mm
    pairs/traj/        reference trajectories, gripper in the filename
    pairs/traj/legacy/ pre-alignment runs, NOT comparable to anything in traj/

    examples/collect_gripper_pairs.sh <Gripper> <outdir>   collect for any gripper
    examples/diag_camera_frame.py    camera->TCP per gripper
    examples/diag_wrist_view.py      what the wrist camera sees
    examples/diag_gripper.py         per-step rollout comparison
    examples/diag_grasp_basin.py     scripted grasp sweep (SUSPECT, see above)
    examples/diag_close_time.py      closure timing (WRITTEN, NEVER RUN)

`eval_corrected.py`: `--gripper --align-start --align-camera --act-bias
--grip-friction --corrector`.

## Standing constraint

Interventions belong in the **action interface** — transform what the policy
outputs and what it is told its actions did. Editing simulator geometry (camera
extrinsics) is **diagnosis only, not a fix**: it does not exist on real hardware.
`--align-camera` is kept because it *identified* the invariant, but a result that
depends on it is not a transferable result.

---

# PARKED — PointWorld + ReKep (through 2026-08-03)

Full record in `NOTES.md`. Only load-bearing conclusions kept here.

## Environments — never merge them

| venv | stack | for |
|---|---|---|
| `.venv` | mujoco 3.1.6, robosuite 1.4.1, numpy 1.26.4, torch 2.13+cu130 | ReKep / LIBERO / VLA-JEPA / Contact-GraspNet |
| `.venv-pw` | torch 2.11+cu130, numpy 2.5, spconv, flash-attn | PointWorld |

**Neither may reference the other at all** — not its interpreter, not its paths,
not its environment variables. A client that can launch its server has the
server's configuration baked in, and rebuilding one venv then breaks the other
for no visible reason. Two interfaces only: `.npz` on disk for offline work, a
UNIX socket for the control loop. **Orchestration lives in shell**, which belongs
to neither.

Any shell importing spconv needs `export CUMM_CUDA_VERSION=13.0
CUMM_CUDA_ARCH_LIST=12.0`.

## What works

**The task is solved, with perception rather than an oracle.**

```bash
POINTWORLD_CKPT=small-droid scripts/run_planner.sh \
    --suite libero_goal --task-id 1 --ground --ticks 45
# success: True (LIBERO's own _check_success), videos/planner_bowl_grounded.mp4
```

The goal comes from the point cloud plus one VLM call; `task_spec` only prints
progress. What made it work was **object proposals by connectivity**
(`scene_graph.py`), not keypoints and not feature search — cluster IoU 0.72–0.89
against the oracle, beating the global-DINOv3 ceiling (0.35–0.67) on every
free-standing object. Grounding measured alone: goal error **24.9 mm, cos +1.00**
with the scene-graph prompt, against 607.2 mm and cos −0.45 for KUDA keypoints.

**The planner** (`plan_pointworld.py`, reimplemented) needed four things at once:
the model must rank candidates rather than the control cost (`rho` +0.03 →
+0.98); rates scale with the distance owed, so a candidate always lands *on* the
goal instead of 80 mm past it; cost read at every step, not only the horizon end;
and execution that tracks (`kp` 400 → 2000 took delivered motion from ~20% of
commanded to 75–97%).

**Collision** is a cost term, not geometry: the model predicts the whole scene, so
disturbing a bystander is predicted motion where none was asked for.
`--avoid-weight 1.0` cut bystander disturbance 15.3 → 8.9 mm *and* increased
progress.

## The load-bearing negative results

**The model predicts "grasped things follow the hand" and nothing finer.** One
finding, reconfirmed three ways:

- **Articulation is not discoverable.** 18 probe directions give 184–199 mm — a
  7% spread, blocked 0.98× free — against a simulator that gives 64.7 mm free and
  0.4 mm blocked. Axes must come from privileged knowledge or perception.
- **PointWorld cannot rank grasps.** Rank correlation **−0.15** against executed
  reality; its top pick closed on air, its worst held. Predictions are identical
  across 0/25/50 mm of lateral offset while reality goes from held to missed. It
  has no representation of **enclosure**. So Contact-GraspNet stays — for this
  measured reason, *not* the retracted "10–15 mm error floor" (the correct
  checkpoint predicts moved points to 3.92 mm, cos 0.999).
- Consequently the drawer success does **not** show the model understands
  drawers: follows-the-hand plus a correct goal is sufficient.

This does not make grasping unplannable — only unplannable *by this cost*. A cost
asking "do the object's points move **differently** from the gripper's" might
separate held from pushed. Untested, and a real option.

## The expensive lesson

**We ran the wrong checkpoint, the wrong domain, and the wrong feature layout**
(`small-droid` + `domains=droid` + 16/31 channels, on simulation data). Fixing it
to `large-droid+behavior` + `behavior` + 17/42 improved error on moved points
**15×** (58.5 → 3.92 mm) and removed the 3.6× magnitude under-prediction.

Days of characterising "model limitations" were describing a misconfiguration.
**"Are we even using the right weights" is cheaper than any measurement built on
top of it, and belongs first.** Everything measured before that fix is retracted;
re-measure before quoting.

Still true and still the dominant error: **60.6 mm of spurious motion on static
points** against a 2.2 mm baseline. The model moves the whole scene.

## Deployment — Jetson Orin / Thor

The venv split accidentally *is* the deployment architecture: a resident
world-model service plus a thin control client. LIBERO is replaced by the real
robot and a RealSense; client, protocol and planner are unchanged.

86% of `small-droid` is frozen DINOv3 (303 M of 354 M), and it runs **once per
observation** (41 ms), while the dynamics predictor runs **per candidate**. That
decomposition is the whole optimisation plan. `large-droid+behavior` is 4.85 GiB
resident, 45.4 ms/candidate and 8.59 GiB at K=32 — it **fits an AGX Orin**; the
13 GB file is mostly optimiser state.

Do not carry K=32 × 3 chunks over: that is 4.36 s/tick on the GB10. Set K and
chunk count from a measured latency budget on the target.

Target repo `~/Documents/GitHub/wardmate_ws`, branch `pointworld-robot-control`;
commit messages there are in Chinese. The robot is a **dual-arm UR7e with
Robotiq grippers**, which is why the bimanual checkpoint is right for the
hardware and not merely for sim. `scripts/setup_pointworld.sh` is arch-locked to
the GB10 (sm_120); Orin is sm_87. Both aarch64, so the expensive spconv/flash-attn
build knowledge transfers.

**Gate:** "an episode running" must mean a **perceived-mask** episode. The oracle
drawer reads `env.points_in_geoms` and cannot run on a robot at all.

## If resumed, in order

1. **Arm-body clearance** via the existing `get_sdf_voxels()` / ReKep machinery.
   The collision term covers object *disturbance*, never the arm's own volume.
2. **Calibrate the grounded displacement** — 66% over (464.7 mm against a true
   279.4 mm) is the largest error left, and `measure_grounding.py` scores it
   without running the loop.
3. **Add an action type to the schema.** "Open the drawer" is not expressible as
   target+destination, so articulated tasks do not ground (cos +0.00).
4. **Persist the target point set** and re-ground on events (grasp, release,
   occlusion, repeated planning failure). Grounding currently re-runs every tick,
   which is free only because it is an oracle.
5. **Find a task where the model's physics is load-bearing** — pushing an
   unrestrained object, or deformables (the deploy target's real application). A
   drawer cannot distinguish follows-the-hand from understanding.

**Not worth doing:** tuning MPPI's temperature/sigma/horizon (all swept; the
search space was the problem, not its parameters); chasing the 60.6 mm spurious
static motion before it costs a task; re-running retracted `small-droid`
measurements for their own sake.

---

# Traps already paid for — do not rediscover

- **A fallback that silently changes experimental conditions is worse than a
  crash.** Three determinism bugs each presented as "the grasp network is flaky".
- **"Flaky model" is a hypothesis of last resort.** Prove reproducibility first.
- **Check the checkpoint before anything else** — wrong weights, domain or layout
  invalidated a session of "model limitations". DINOv3 was inside `model-best.pt`
  the whole time; 1.2 GB was downloaded unnecessarily.
- **When a fix shrinks an error instead of eliminating it, the mechanism is only
  partly understood.** Chase the residual.
- **Distance metrics are isotropic; objects are not.** 22 mm along a handle bar is
  free, 7 mm above it is fatal.
- **A single scalar hides what you care about.** A second camera left mean error
  unchanged and moved directional accuracy 0.73 → 0.94. Score direction and
  location, not just magnitude.
- **`geom_rbound` is the CIRCUMSCRIBED radius.** Cost time three times. Never
  sample or test against it.
- **`EpisodeFinished` is LIBERO reporting SUCCESS**, not an error. Bitten three
  times.
- **Do not wrap long-running jobs in `| tail -N`.** It buffers, so a silent death
  looks identical to progress.
- **PTv3 is not deterministic in `eval()`** (`shuffle_orders` reaches
  `torch.randperm` every forward). Seed it and report the spread.
- **Never `pkill -f <pattern>`** that could match your own command line. Self-kill,
  exit 144, three times. Use a fresh port/socket instead.
- **Two live LIBERO envs corrupt each other's EGL context.** Build any reference
  scene *before* the main env and close it first.
