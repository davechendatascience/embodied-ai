# Start here

Read this, then `NOTES.md`. This file is the current state and the next action.
`NOTES.md` is the accumulated findings and the traps.

---

## Two environments, never merge them

| venv | stack | for |
|---|---|---|
| `.venv` | mujoco 3.1.6, robosuite 1.4.1, numpy 1.26.4, torch 2.13+cu130 | ReKep / LIBERO / Contact-GraspNet |
| `.venv-pw` | torch 2.11+cu130, numpy 2.5, spconv, flash-attn | PointWorld |

They cannot share a process, and **neither may reference the other at all** —
not its interpreter, not its paths, not its environment variables. A client
that can launch its server has the server's configuration baked into it, and
then rebuilding one venv breaks the other for no visible reason.

Two interfaces, and nothing else crosses:

| | for |
|---|---|
| `.npz` on disk | offline work — recording episodes, scoring them |
| UNIX socket | the control loop — `scripts/pointworld_serve.sh` |

**Orchestration goes in shell**, which belongs to neither venv — see
`scripts/run_bridge_test.sh`. Rebuild `.venv-pw` with
`scripts/setup_pointworld.sh` (idempotent).

Any shell that imports spconv needs:
```bash
export CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0
```

---

## What works, verified

**ReKep + Contact-GraspNet — finished.**
```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_drawer_open.py
# 3/3 deterministic, LIBERO _check_success() True, drawer opens 142.5 mm
```

**PointWorld port — finished.** 464/464 checkpoint tensors load; `SubMConv3d`
runs on the GB10; **47 ms** per 10-step chunk (2.13x the paper's 0.1 s):
```bash
CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 .venv-pw/bin/python tests/bench_pointworld.py
```

**Episode recorder — finished.** Exact ground-truth point correspondence, two
cameras, arm removed from the scene, gripper sampled off its real meshes:
```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python scripts/record_pointworld_episode.py --motion drawer
# 130 mm of real articulated motion; 73 of 2789 points move
```

**PointWorld evaluation — finished.** Drives `BaseModel`, DINOv3 loaded from
`model-best.pt` (nothing downloaded):
```bash
CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 .venv-pw/bin/python \
    scripts/extract_dinov3_from_checkpoint.py            # once
CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \
    .venv-pw/bin/python tests/run_pointworld_on_episode.py --sweep-gripper
```

| | all pts | moved pts |
|---|---|---|
| PointWorld | 14.9 mm | **58.5 ± 1.2 mm** |
| predict-no-motion | 2.2 mm | 84.5 mm |

Beats the static baseline where the motion is, loses on the scene as a whole.
Direction `cos = 0.94`, magnitude under-predicted 3.6x, ~10–15 mm error floor
independent of the true motion. `NOTES.md` §4 has the five input defects that
got it here and the evidence that every input is now in-distribution.

**Videos — done.** `videos/drawer_open.mp4` (`tests/test_drawer_open.py
--video`) and `videos/point_flow.mp4` (`scripts/render_point_flow.py`).

**Timing — measured on the recorded episode**
(`tests/bench_pointworld_realtime.py`): perception 41 ms once per observation,
rollout 29 ms per candidate. See "Real-time" below.

---

## The model can rank actions — so the loop is worth building

```bash
CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \
    .venv-pw/bin/python tests/rank_actions_pointworld.py
```
Direction ranked correctly (all three `+Y` candidates beat all sixteen others;
`-Y` worst and worse with speed). **The cost landscape is well shaped.**

> **Re-measured on `large-droid+behavior`** — see the bottom of this file.
> Direction still ranks correctly and the best cost falls to 3.4 mm, but the
> **2.0x magnitude overshoot is gone** (argmin 13 mm/step against a true 13).
> The "open-loop would overshoot 2x" argument for receding-horizon is
> withdrawn; the shape is still right, the reason has to be different.

---

## The bridge — done

```bash
scripts/pointworld_serve.sh              # service, in .venv-pw
scripts/run_bridge_test.sh               # up, test from .venv, down
```
`observe` 133 ms warm, `rollout` 46 ms at K=1, 24 ms/candidate at K=20.
Verified to give the same answer as the in-process path. `NOTES.md` §4.

**The socket is the entire boundary.** Neither venv may reference the other —
not even to launch it. `client.py` imports numpy and stdlib only; lifecycle
lives in shell, which belongs to neither. Do not put a `start_server()` back
into the client.

**A planner must seed or average** — on `small-droid`, where the model scattered
**~9 mm run to run on identical input** (atomics in spconv/scatter, plus
`shuffle_orders`), larger than the gaps between good candidates. On
`large-droid+behavior` that is **1.94 mm apart / 0.46 mm batched**, against
4.2 mm between the top two candidates, so scoring in one batch is sufficient.
The non-determinism is real either way; it is no longer decisive.

---

## Gripper FK — done

`src/rekep_libero/gripper_points.py` — ONE definition, shared by the recorder
and the planner. `bind(env)` freezes each geom's pose relative to the ee;
`at_ee_pose()` / `trajectory()` then place the gripper for candidate poses with
no physics, at 0.8 ms per 11-step candidate (26 ms for K=32, against 29 ms of
model time — negligible).

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_gripper_fk.py
```
The transform is **exact**: 0.000 mm at bind and after re-bind. Residual error
tracks jaw creep 1:1 (0.940 vs 0.940 mm; 1.070 vs 1.062), i.e. it is the
rigidity assumption bending, not the maths. At ~1 mm it is 0.12x the model's
own 9 mm noise — invisible to the planner.

**Re-`bind()` after opening or closing the gripper.** Closing costs 37.6 mm of
FK error; `check_binding()` reports the jaw drift so a stale binding cannot
pass silently.

---

## The closed loop — works, on ORACLE inputs

> **STALE (2026-08-03).** This entry predates the MPPI rewrite. The run below
> was `plan_pointworld.py` when it still used a STRUCTURED candidate set and an
> argmin -- "picked +Y out of 26 directions, annealed 26 -> 16 -> 8 mm" is that
> design, and it is preserved in `tests/plan_drawer_pointworld.py`. The current
> `plan_pointworld.py` is MPPI and does NOT solve the task; measured today it
> closed **-1.7 mm over 60 ticks**. Do not read this section as describing the
> code that runs now. See "THE PLANNER, DIAGNOSED" at the bottom.

```bash
scripts/run_planner.sh --suite libero_goal --task-id 0 --no-cgn
# owes 160.0 -> 19.5 mm, LIBERO _check_success() True, 18 ticks
```
PointWorld chooses every action. It picked `+Y` out of 26 directions on every
tick and annealed the rate 26 → 16 → 8 mm as it closed in — behaviour no
script supplied. `tests/plan_pointworld.py` contains nothing about drawers;
the task enters only through a `TaskSpec`.

**But the target comes from the simulator, not from perception.** This is the
honest limit of the result and it is easy to overstate. Privileged inputs, all
listed at the top of `src/rekep_libero/task_spec.py`:

| input | today | on a robot |
|---|---|---|
| **mask** over the target's points | `env.points_in_geoms` — MuJoCo collision geometry | must segment from cloud + RGB |
| **goal** | `parsed_problem["goal_state"]` — LIBERO's answer key | instruction → VLM |
| **axis** of an articulated part | `model.jnt_axis` | unknown — but possibly discoverable |
| **object pose** | geom poses | must perceive |

`env.robot_geom_mask()` is fine — a robot knows its own kinematics.

**So the claim is "the planner can act given a correct target", not "the
system can do the task".** Do not quote the success rate without this.

One thing makes the gap smaller than it looks: PointWorld's scene encoder
**already computes 128-d DINOv3 features per point** (std 1.55, measured) —
the substrate a segmentation or text-query head would consume, so features are
not the missing part; a query is.

---

## The axis is NOT discoverable — tested, negative, and re-confirmed

```bash
scripts/run_axis_discovery.sh            # ~1 min
```

The table below is `small-droid` and its **inversion is an artefact** — see the
re-measurement at the bottom of this file. On `large-droid+behavior` the
prediction is FLAT instead (184–199 mm in all 18 directions, blocked 0.98x
free), which is the same verdict by a cleaner route: articulation must come
from privileged knowledge or perception, not from rolling out candidates.

| | PointWorld predicts | simulator does |
|---|---|---|
| free `+Y` | 71.5 mm | **64.9 mm** |
| blocked `-Y` | **153.2 mm** | ~0 |
| across `+Z` | 66.0 mm | 13.7 mm |

Rank correlation with reality **-0.50**; argmax **135° from the true axis**.
`NOTES.md` §4.

**This also downgrades the drawer success.** A model believing only "grasped
things follow the hand" is *sufficient* to solve that task given a correct
goal, so opening the drawer does not show PointWorld understands drawers. The
planner works; the model's physics is not what makes it work. The "model
understands the constraint" line in `NOTES.md` §4 is retracted.

---

## Language: PointWorld has none, and the authors did not build the wrapper

Established by reading the paper, not assumed:

- **The trained model is language-free.** No text encoder, no cross-attention
  to embeddings. `BaseModel.forward` takes points, features and a domain tag —
  verified in code. Language never enters training or inference.
- **The one mention is a possibility, not a result:** *"task-relevant points
  can be specified by either human via GUI or by VLMs."* Every reported
  success rate used the human path — Appendix A.6.2: *"we specify tasks
  through a GUI tool that allows users to select object masks using **SAM2**
  and specify target positions in the world frame."*
- The limitations section lists VLM task specification as **future work**.

So PointWorld is purely the dynamics layer. The language layer must be built,
and its interface is already fixed and small: **a mask plus world-frame target
positions** — exactly what `tests/plan_pointworld.py` consumes today from
MuJoCo.

### The blueprint exists: KUDA (`third_party/KUDA`, arXiv 2503.10546)

> **SUPERSEDED as the design basis (2026-08-03).** We build to
> `docs/pointworld_pipeline_implementation_prereqs.md`: scene graph → LLM goal
> formalizer → `pytorch_mppi`, with event-driven re-grounding. KUDA stays as
> evidence for where the language/control split belongs, not as the recipe.
> See the note at the top of `NOTES.md` §6.

Same architecture with a weaker dynamics model — *"assigns keypoints to the
RGB image and queries the VLM to generate target specifications, then converts
these into cost functions, which are optimized using a learned dynamics
model"* — **80% over 60 trials**, free-form language, multi-object,
deformables. Its prompt (`third_party/KUDA/prompts/low_level_prompt.txt`) is
directly reusable:

```
image overlaid with keypoints marked P[i]  ->  VLM returns Python:
    p_i = p_a + [5, 0, 0]     # cm; x right, y up, z out of image
    p_j = p_b + [0, 7, 0]     # or p_i = [dx,dy,dz] from image centre
```
Targets are expressed as OFFSETS FROM OTHER KEYPOINTS, which is what makes a
VLM's ±cm spatial precision sufficient — it never has to name an absolute
coordinate. It also returns `"Done."` when the task is complete, which doubles
as the termination signal.

### What we already have for this, and what is missing

| piece | status |
|---|---|
| lift a 2D mask to 3D point indices | **free** — `scene_featurizer.py` already projects every point into both cameras with visibility + depth checks |
| MPC that consumes mask + targets | **done** — `plan_pointworld.py` |
| mask from language | **missing** — needs a pointing VLM (RoboPoint, arXiv 2406.10721, beats GPT-4o by 21.8% on spatial affordance) + SAM2, matching the authors' own GUI |
| targets from language | **missing** — KUDA's offset-from-reference prompt |

A pointing VLM needs its own venv. That is not a new problem: the socket
bridge pattern in `src/pointworld_bridge/` generalises to a second grounding
service, with shell holding both lifecycles.

---

### Language layer — half built, and built to the wrong blueprint

> **Re-read against the guide before extending it.** `keypoint_grounding.py`
> implements KUDA's keypoint-offset prompt; the guide asks for an object-level
> scene graph consumed by a goal formalizer. The projection and parsing
> machinery is reusable; the prompt and the output schema are not the target.

`src/rekep_libero/keypoint_grounding.py`. KUDA's recipe: farthest-point-sample
keypoints, mark them `P[i]` on the RGB, ask a VLM for the final state as
OFFSETS FROM OTHER KEYPOINTS, parse back to world-frame targets.

Unit-tested and working: keypoint proposal, projection, the parser (against
KUDA's own format, malformed lines, and the `Done.` completion signal), and
the cm→m lift. Because keypoints ARE scene-point indices, `goal_idx` drops
straight into the bridge with no second lookup.

**Not yet done:** the VLM call is untested against a real backend and image,
and nothing is wired into `plan_pointworld.py`, which still takes its target
from `task_spec.py`. Keep that two-stage separation until the grounding is
measured on its own — a grounding failure and a planner failure are
indistinguishable once they are in the same loop.

**One deliberate departure from KUDA:** their camera is top-down so they speak
in image coordinates. Our agentview is oblique, so the prompt is written in the
WORLD frame with axes named. Getting this wrong would silently mean something
else.

---

## MPPI — matched to the paper, and where it stalls

`tests/plan_pointworld.py` now does spline-correlated sampling, 30 steps as 3
chained chunks, control cost, temperature weighting.

**The temperature must be solved for, not chosen.** Both constant-beta failure
modes were measured: `beta = 0.02` gives ESS **1.1/32** (collapsed to argmin,
the model's 9 mm noise deciding); `beta = std(J)` gives ESS **28/32**
(averaging uniformly, plan drifts off-axis). Bisecting for a target ESS of
K/4 pins it at 8.0/32 and took progress 111 → **129.4 mm**.

**It still stalls ~30 mm short**, and that is the model, not the planner:
below ~30 mm of true motion PointWorld loses to predict-no-motion (19.4 vs
17.6 mm, measured). The planner cannot descend a gradient the model cannot
resolve.

**Cost: 96 forwards/tick** (32 samples × 3 chunks) vs 19 before. Most of the
added wall-clock is CPU, not GPU — `build_data_dict` runs 96×/tick and rebuilds
`dist2robot` with a `cKDTree` each time. It is a 2789×500 distance matrix;
move it to the GPU before tuning anything else.

---

## THE NEXT TASK

**Finish the language layer**: test `ground()` against a real VLM backend on a
LIBERO frame, measure its keypoint accuracy ALONE, then wire it into the
planner. Then, still open:

1. **The mask.** Query PointWorld's own per-point DINOv3 features instead of
   MuJoCo geoms — cluster them, or match against a text/image embedding. This
   is the single biggest cheat and the one with the most machinery already in
   place.
2. **The goal, from language.** After 1, so a grounding failure is
   distinguishable from a planner failure.
3. **Articulation from perception**, now that discovery is ruled out. Depth
   over time gives relative motion between parts; that is where a joint axis
   could come from honestly.

Also worth doing, given the axis result: find a task where the model's
physics is actually load-bearing — pushing an unrestrained object, where
"follows the hand" is wrong and contact dynamics decide the outcome. The
drawer cannot distinguish those.

Older, still open:

1. **The planner.** Sample ee trajectories, batch the rollouts (they batch on
   one GPU — do not loop), score against the target spec, execute one step,
   re-observe. `rank_actions_pointworld.py` already contains the cost function.
2. **The target spec, and only here does an LLM enter.** PointWorld needs a
   MASK over scene points plus 3D TARGET POSITIONS for them. Nothing else in
   the pipeline consumes language. Build and test the loop with a geometric
   spec first — the ranking test uses one — so a planner bug is never confused
   with a grounding failure. Our VLM grounding has failed once already, picking
   a keypoint 153 mm from any drawer handle (`NOTES.md` §4).

**On LIBERO and the LLM, measured rather than assumed.** `libero_goal` and
`libero_spatial` are each ONE scene with ten instructions — task ids differ
only in the sentence, and `libero_spatial`'s are all "pick up the black bowl
⟨disambiguator⟩". So ReKep success rates over those suites are pure language
grounding and genuinely need the VLM. PointWorld never sees the instruction, so
sweeping task ids there would have produced ten copies of one scene. For
dynamics, enumerate INTERACTIONS instead: one `libero_goal` scene already
offers eight targets, and `libero_object` / `libero_10` vary real geometry per
task id.

---

## Real-time

Measured, `tests/bench_pointworld_realtime.py`, GB10, 2789 scene points.
**The 29 ms is `small-droid`; on `large-droid+behavior` it is 98 ms unbatched
and 45 ms batched at K=32** — full table at the bottom of this file.

| stage | cost | how often |
|---|---|---|
| DINOv3 + projection | **41 ms** | once per camera observation |
| PTv3 trunk + heads | **29 ms** | once per candidate trajectory |

`BaseModel.forward(..., encoded_scene_feat0=...)` exists so perception is
reused; re-running it per candidate costs 70 ms instead of 29 and is the first
mistake to avoid. One candidate per tick is 14 Hz. K candidates unbatched is
`41 + 29K` ms — 6.4 Hz at K=4 — and they batch on one GPU, so that is an upper
bound.

Fast enough for a closed loop at single-digit Hz. What does **not** exist yet
is the loop itself: no MPPI/CEM sampler, no target-point specification, no
controller. Only the forward model is ported and measured.

---

## Traps already paid for — do not rediscover

- **A fallback that silently changes experimental conditions is worse than a
  crash.** Three separate determinism bugs each presented as "the grasp network
  is flaky". `NOTES.md` §1.
- **"Flaky model" is a hypothesis of last resort.** Prove reproducibility before
  interpreting anything.
- **When a fix shrinks an error instead of eliminating it, the mechanism is only
  partly understood.** Chase the residual.
- **Distance metrics are isotropic; objects are not.** 22 mm along a handle bar
  is free, 7 mm above it is fatal. `NOTES.md` §3.
- **Check the checkpoint's own keys before downloading anything.** DINOv3 was
  inside `model-best.pt` the whole time; I downloaded 1.2 GB unnecessarily.
- **`EpisodeFinished` is LIBERO reporting SUCCESS**, not an error. It has bitten
  three times.
- **Do not wrap long-running jobs in `| tail -N`.** It buffers, so a silent
  death looks identical to progress. Cost real time twice.
- **PTv3 is not deterministic in `eval()`.** `shuffle_orders=True` reaches
  `torch.randperm` on every forward. Seed it and report the spread; two runs
  differed by 5 mm. `NOTES.md` §4.
- **`geom_rbound` is the CIRCUMSCRIBED radius.** Third time this has cost
  time: it swallowed the table into object clouds, and it inflated a
  63x93x206 mm gripper into a 234 mm blob. Never sample or test against it.
- **A single scalar hides the thing you care about.** Adding the second camera
  left the mean error unchanged and moved directional accuracy from 0.73 to
  0.94. Score direction and location, not just magnitude.

---

## Open, lower priority

- `--motion push` produces `0.0 mm` of scene motion — a dynamics model scores
  perfectly on it while demonstrating nothing. Use `--motion drawer`.
- Images are 256x256; the release contract asserts 180x320, and LIBERO's
  agentview FOV is not DROID's. Untested as a factor.
- Full LIBERO suite has never been run; only single tasks. Two different jobs
  hide behind that sentence: (a) ReKep+CGN task success rates over the suite,
  which needs the VLM grounding that failed once already (`NOTES.md` §4), and
  (b) PointWorld episodes across many tasks, which needs only scripted
  interactions. (b) is the cheaper and more informative one.
- **No LLM or VLM is in the PointWorld path at all** — depth to points, DINOv3
  (a ViT), PTv3, dynamics head, and a scripted grasp. `vlm_backends.py` is
  reachable only from the ReKep runner.

---

## REDESIGN (2026-08-03) — read before touching the planner

### We were running the wrong checkpoint, the wrong domain, and the wrong feature layout

The guide's own simulation recipe:

```bash
MODEL_PATH=pretrained_checkpoints/large-droid+behavior/model-best.pt
  --domains=behavior  --norm_stats_path=stats/droid_behavior
```

We ran `small-droid` + `--domains=droid` + `stats/droid` on LIBERO, which is
simulation. Three separate errors:

| | was | should be |
|---|---|---|
| checkpoint | `small-droid` (weakest, real-robot only) | `large-droid+behavior` |
| domain tag | `droid` | `behavior` (sim) |
| norm stats | `stats/droid` | `stats/droid_behavior` |
| robot feature channels | **16** | **17** |
| scene feature channels | **31** | **42** |

The channel counts are the trap. `stats/droid_behavior` is BIMANUAL: gripper
state occupies 2 slots, not 1, so robot goes 3+3+3+**2**+3+3 = 17 and scene
goes 3+3+3+**22**+11 = 42. `build_data_dict` hardcodes the single-arm layout
via `has_bimanual_robot=False`. It must switch, with a zeroed `left_gripper_open`.

**CORRECTION — the domain switch is NOT the fix for magnitude under-prediction.**
I suggested it might be. It cannot be: `behavior`'s step-10 output std is
`[0.043, 0.043, 0.016]` against `droid`'s `[0.074, 0.074, 0.043]`, so sim stats
un-normalize to SMALLER displacements. Switching alone would predict less
motion, not more. It is required for correctness; it is not the cure.

**Every number in this file predates these fixes** — 58.5 mm, the 10-15 mm
floor, the axis-discovery failure, "follows the hand". Re-measure before
quoting any of it.

The authors independently confirm the non-determinism finding: *"Eval outputs
are not deterministic on GPU; small run-to-run variation is expected even with
fixed seeds."*

### The planning layer is NOT released

`deploy/robots.py` is referenced from `robot_sampler.py` and `arguments.py` but
is absent from the release. There is no MPC code to port — it has to be built.

### KUDA is FEW-SHOT, so its 80% does not transfer

`planner/prompt_retriever.py` uses CLIP to retrieve the top-5 examples from a
curated library (`prompts/examples`: coffee_beans, cubes, pushing_T, rope, plus
13 flat examples) into every prompt. Zero-shot means dropping the retriever, so
take their PROMPT STRUCTURE only and expect worse grounding than they report.

### Architecture

```
RGB-D x2 ─► PointWorld encoder (DINOv3) ─► per-point features ──┐
   └─► cloud ─► FPS keypoints ─► P[i] image ─┐                  │
instruction ─────────────────────────────────┤                  │
                              zero-shot VLM (no examples)        │
                                p_i = p_a + [dx,dy,dz]           │
                                             │                   │
                              goal_idx, goal_pos                 │
                                             ▼                   ▼
                          pytorch_mppi  ──────────────►  PointWorld rollout
                            dynamics    = ee integration (exact, ours)
                            running_cost= path length + reachability
                            terminal    = ONE batched socket call, all K
                                             │
                              first action ─► execute ─► re-observe
```

`pytorch-mppi` (0.9.1, installed) owns sampling, temperature and warm start.
Its `terminal_state_cost(states, actions)` receives the whole K x T rollout,
which is exactly the seam for one batched PointWorld call; `sample_null_action`
gives the "do nothing" candidate for free. We keep only the two cost functions.
This also deletes the hand-rolled temperature bisection.

### Assembly optimisation — done, 6x, verified equal

`src/pointworld_bridge/fast_features.py`. The reference pipeline runs ONCE per
`observe` for the invariant channels; only the four action-dependent groups
(robot flows, velocity, acceleration, `dist2robot`) are recomputed per
candidate, on the GPU.

```bash
CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \
    .venv-pw/bin/python tests/test_fast_features.py
```

| | per candidate |
|---|---|
| reference (`gather_features`, CPU, cKDTree) | 10.2 ms |
| GPU assembly | **1.81 ms** |

Agreement is exact to float32 noise (2.3e-6, `cdist` vs `cKDTree`). In the
live service `rollout` assembly dropped 10.4 -> **2.0 ms**, and the bridge
control still passes. At 96 candidates/tick that is ~980 -> ~174 ms, about a
third of the whole tick.

**Two traps worth keeping.** The flat `cdist` over `(K*T, Ns, Nr)` allocates
491 MB at K=8 and was measured **3.5x SLOWER than the CPU it replaced**;
chunking over time (45 MB resident) is what actually made it fast. And the
first measurement had no warmup, which is the same mistake that made `observe`
look 7x more expensive than it is.

`fast_features.py` is a shortcut around upstream code, so it is worth exactly
its agreement with the reference. Re-run the control after touching it.

---

## DEPLOYMENT TARGET: Jetson Orin / Thor — design constraint

Everything below is chosen with edge deployment in mind, not just the GB10.

### The architecture already fits, by accident of the venv split

The socket bridge built to keep `.venv` and `.venv-pw` apart **is** the
deployment architecture: a resident world-model service plus a thin control
client. On the robot, LIBERO is simply replaced by the real robot and the
RealSense; the client, the protocol and the planner are unchanged. Nothing
needs re-architecting for the move.

### 86% of the model is DINOv3, and it runs once per OBSERVATION

Measured on `small-droid`:

| | value |
|---|---|
| total params | 354 M |
| of which DINOv3 (frozen) | **303 M — 86%** |
| dynamics predictor (per candidate) | ~51 M |
| fp32 weights / fp16 | 1.32 GiB / **0.66 GiB** |
| peak VRAM, one forward | 1.32 GiB |

That decomposition is the whole optimisation plan, because the two halves have
different duty cycles:

* **DINOv3** — once per observation, 133 ms, frozen, a stock ViT-L. TensorRT +
  fp16 is well-trodden here and cannot affect anything else, since the encoder
  is `_freeze_encoder`'d. Biggest single latency win.
* **PTv3 + heads** — 96x per planning tick but only 51 M params. The levers
  here are batch size, fp16, and reducing K or the chunk count, NOT model
  surgery.

At 1.32 GiB `small-droid` fits an Orin comfortably. `large-droid+behavior` is
12+ GB **on disk**, but that file includes optimiser state; the weight-only
footprint has to be measured before assuming it is infeasible.

### The checkpoint choice is now three-way

Accuracy x latency x **memory**. The bench must report VRAM and parameter count
alongside error and ms/candidate, because the sim-domain checkpoint winning on
accuracy does not make it deployable.

### `scripts/setup_pointworld.sh` is arch-locked and must be parameterised

It hardcodes `CUMM_CUDA_VERSION=13.0` / `CUMM_CUDA_ARCH_LIST=12.0` for the
GB10. Orin is **sm_87** (JetPack 6, CUDA 12.6); Thor is Blackwell-class, closer
to the GB10 we already build for. Both are **aarch64**, so the hard-won
spconv/cumm/flash-attn build knowledge transfers directly — that was the
expensive part and it is already paid for. Take the arch as an argument.

### Consequences for the planner

96 forwards/tick is a GB10 budget, not an Orin one. Expect the Ampere GPU in
Orin to be several times slower, so the MPPI configuration (K=32, 3 chunks)
should be treated as a knob to be set from a measured latency budget on the
target, not a constant.

### Port target: `wardmate_ws/src/pointworld_robot_control`

Repo `~/Documents/GitHub/wardmate_ws`, branch **`pointworld-robot-control`**
(created 2026-08-03 off `wam-robot-control`). Commit messages there are in
Chinese — match the convention.

**The robot is a DUAL-ARM UR7e with Robotiq grippers**
(`ur_arm_description/urdf/ur7e_dual_arm_real.urdf.xacro`,
`ros2_robotiq_gripper`). Two consequences that settle open questions:

1. **The bimanual checkpoint is right for the ROBOT, not just for sim.**
   `large-droid+behavior` uses 17 robot / 42 scene channels because gripper
   state occupies two slots. A single-arm checkpoint would represent half the
   hardware. This is now the leading choice on grounds independent of accuracy.
2. **The Robotiq gripper matches DROID's exactly** — PointWorld's own URDF is
   `franka_panda_robotiq_2f85.urdf`. The arm differs, which is precisely the
   cross-embodiment claim the point-flow action space exists to support.

`vision_perception/vision/bedsheet_segmentation_node.py` already exists. So the
real application is **deformable** manipulation — PointWorld's claimed
strength, the thing ReKep structurally cannot do (`NOTES.md` §4), and exactly
the regime the drawer task could not tell us anything about.

**What ports unchanged** (no simulator dependency):
`src/pointworld_bridge/*` (protocol, client, server, model, episode,
fast_features), `scripts/pointworld_serve.sh`, `keypoint_grounding.py`, and the
MPPI core once it is on `pytorch_mppi`.

**What needs a robot equivalent** — same output contract, different source:
`pw_observation.py` (RealSense + calibration + URDF/TF self-filter instead of
`env.get_cam_obs` / `robot_geom_mask`) and `gripper_points.py` (URDF meshes via
TF instead of MuJoCo geoms — upstream's `robot_sampler.py` already does this
with urdfpy+trimesh).

**What does not port:** `task_spec.py` (oracle), `environment_libero.py`, the
recorder, the LIBERO tests. Contact-GraspNet is already shared with
`llm_robot_control` (`NOTES.md` §3).

**Gate.** "An episode running successfully" should mean a **perceived-mask**
episode. The oracle drawer already passes, but it takes its mask from
`env.points_in_geoms` and therefore cannot run on a robot at all — porting it
would move something structurally undeployable.

### All three checkpoints downloaded and measured (2026-08-03)

| | small-droid | large-droid | large-droid+behavior |
|---|---|---|---|
| ptv3 / predictor_dim | small / 128 | large / 256 | large / 256 |
| domains | `droid` | `droid` | **`droid`, `behavior`** |
| norm stats | `stats/droid` | `stats/droid` | **`stats/droid_behavior`** |
| robot / scene channels | 16 / 31 | 16 / 31 | **17 / 42** |
| params (DINOv3 of that) | 354 M (303 M) | 1289 M (303 M) | 1289 M (303 M) |
| **weights fp32 / fp16** | 1.32 / 0.66 GiB | **4.80 / 2.40 GiB** | **4.80 / 2.40 GiB** |

**The 13 GB file is mostly optimiser state — weights are 4.8 GiB fp32, 2.4 GiB
fp16.** So `large-droid+behavior` IS deployable on an AGX Orin, and the
memory-vs-accuracy tension flagged earlier does not exist. Use it.

The real cost is latency, not memory: DINOv3 is 303 M in all three, so the
DYNAMICS predictor grows 51 M -> 986 M, a 19x jump in the part that runs once
per candidate. Measure ms/candidate before fixing K.

**To actually use it, two changes remain.** `data_info_from_checkpoint` already
reads 17/42 from the checkpoint, and `FastFeatures` takes the layout as a
parameter, so both adapt automatically. Missing:

1. `build_data_dict` hardcodes `has_bimanual_robot=False` and supplies no
   `left_gripper_open`. It must pass `True` and a zeroed left gripper for a
   single-arm scene — that is what makes 17/42 instead of 16/31.
2. `__domain__` is hardcoded `"droid"`. LIBERO is simulation, so it must be
   `"behavior"`, which also selects row 1 of the `(2, 11, 3)` per-step norm
   stats. Getting this wrong silently un-normalises through the wrong domain.

---

## CHECKPOINT SWAPPED (2026-08-03) — most earlier findings are now INVALID

`POINTWORLD_CKPT` selects the checkpoint; default is now
**`large-droid+behavior`**, `__domain__` resolves to `behavior` for simulation,
and `build_data_dict` assembles the bimanual 17/42 layout
(`args._bimanual`, set from the checkpoint's own projection widths).

```bash
CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \
    .venv-pw/bin/python tests/run_pointworld_on_episode.py
```

| | small-droid (all prior numbers) | large-droid+behavior |
|---|---|---|
| moved pts | 58.5 ± 1.2 mm | **3.92 ± 0.45 mm** |
| direction `cos` | 0.940 | **0.999** |
| predicted motion, moved | 36.4 mm vs 130.3 true | **126.4 mm vs 130.3 true** |
| all pts | 14.9 mm | 38.0 mm |
| spurious motion, static pts | 14.6 mm | **60.6 mm** |

**A 15x reduction on the points that move, and the 3.6x magnitude
under-prediction is gone.** The model was never as weak as this file said. The
error was ours: wrong checkpoint (smallest, real-robot-only), wrong domain
(`droid` on simulation data), wrong feature layout (16/31 single-arm).

**Lesson, and it is the expensive one from this session:** "are we even using
the right weights" is a cheaper question than any of the careful measurements
built on top of it, and it should be asked FIRST. Several days of characterising
"model limitations" described a misconfiguration.

### RETRACTED — do not quote, re-measure before relying on any of it

Everything below was measured on `small-droid` + `domains=droid` + 16/31:

- **58.5 mm on moved points**, the 3.6x magnitude under-prediction, and the
  `cos = 0.94` direction figure. Superseded by the table above.
- **The ~10-15 mm error floor**, and with it the conclusion that the model
  cannot resolve millimetre-scale features. Re-run the 3.2 mm/step episode.
- **"The axis is NOT discoverable"** (`NOTES.md` §4) -- predicted motion
  anti-correlated with reality at -0.50, argmax 135 deg off.
  **RE-RUN, and the CONCLUSION HOLDS**: on the correct checkpoint the model is
  flat rather than inverted (blocked 0.98x free), which is the same negative
  answer. Only the anti-correlation was an artefact.
- **The action-ranking result** and its "2.0x overshoot", which motivated
  receding-horizon execution. **RE-RUN: direction still ranks correctly, the
  overshoot is GONE (1.0x).** The receding-horizon justification is withdrawn.
- **The DINOv3 ablation** (14.6 vs 131 mm spurious motion).
- Downgrading the drawer success to "follows the hand is sufficient" -- that
  argument was built on the axis result.

### What survives, because it is independent of the checkpoint

`RobotPoints` + joint-space FK, the GPU feature assembly (verified equal to
upstream), the bridge and its control, `get_ee_pose()` reading live sim, the
joint controller, the KUDA/zero-shot findings, and the deployment analysis.

### Still true, and now the dominant error

**60.6 mm of spurious motion on STATIC points**, worse than small-droid's
14.6 mm, against a 2.2 mm static baseline. The model moves the whole scene.
This is a real property of this checkpoint on our data.

---

## RE-MEASURED ON `large-droid+behavior` (2026-08-03, later) — items 1 and 2 done

### 1. The axis is STILL not discoverable — and now for a cleaner reason

`scripts/run_axis_discovery.sh`, 18 directions, 10-step probe at 20 mm/step
(200 mm of hand travel):

| | small-droid (retracted) | large-droid+behavior |
|---|---|---|
| free `+Y` | 71.5 mm | 195.7 mm |
| blocked `-Y` | 153.2 mm (**2.14x free**) | 192.3 mm (**0.98x free**) |
| across `+Z` | 66.0 mm | 192.9 mm |
| spread over all 18 | — | 184.4 – 199.3 mm (**7%**) |
| rank corr. vs simulator | **-0.50** | **+0.50** |
| argmax vs true axis | 135 deg | 90 deg |

Simulator, same grasp, executed and rewound: free 64.7 mm, blocked 0.4 mm,
across 17.2 mm (over 4 steps / 80 mm of hand travel).

**The verdict survives the checkpoint fix; only the mechanism was an
artefact.** small-droid's ordering was INVERTED — it predicted most motion into
the blocked direction. large-droid+behavior's ordering is FLAT: 184–199 mm
whichever way you push, against 200 mm of hand travel, i.e. the target follows
the hand at 0.92–1.00x in all 18 directions. That is "grasped things follow the
hand" stated as precisely as this test can state it.

So `NOTES.md` §4's conclusion is **reinstated**, with the inversion removed:
articulation must come from privileged knowledge or from perception, not from
rolling out candidates. And the downgrade of the drawer success stands — a
follow-the-hand model plus a correct goal is sufficient to open that drawer, so
solving it does not show PointWorld understands drawers.

**One flaw in the test, worth fixing before quoting the two columns
side by side.** The model is asked over `T_LEN = 11` (200 mm of hand travel)
while the simulator is probed over `PROBE_STEPS = 4` (80 mm). The
discrimination result is unaffected — it compares directions against each other
at one horizon — but "says 195.7 / moved 64.7" is not like for like.

### 2. Ranking — better, and it retires the receding-horizon rationale

`tests/rank_actions_pointworld.py`:

| | small-droid (retracted) | large-droid+behavior |
|---|---|---|
| top 3 candidates | all `+Y` | all `+Y` |
| worst | `-Y @ 26` | `-Y @ 26` |
| best cost | ~62 mm | **3.4 ± 0.5 mm** |
| argmin rate vs true 13 mm/step | 26 — **2.0x overshoot** | **13 — 1.0x, no bias** |

**The 2.0x magnitude overshoot is gone.** It was the sole evidence for
"open-loop execution would overshoot 2x", which was the stated reason the loop
had to be receding-horizon. Receding-horizon is still the right shape — for
disturbance, contact and re-grounding — but that argument now has to be made on
other grounds, and this test must not be cited for it.
`rank_actions_pointworld.py` printed the overshoot conclusion unconditionally;
it now derives the sign of the bias and says so.

**Run-to-run noise fell with it**, `tests/test_bridge_roundtrip.py`:

| | small-droid | large-droid+behavior |
|---|---|---|
| 8 identical candidates, run apart | 8.9 mm | **1.94 mm** |
| 8 identical candidates, one batch | 3.9 mm | **0.46 mm** |

The gap between the best and second candidate is 4.2 mm, which is 9x the
batched noise. **"A planner must seed or average" is no longer load-bearing**
if candidates are scored in one batch — which they are.

### 3. Latency and VRAM — the 19x parameter jump costs 3.4x, not 19x

Batched rollout with scene features reused, 2789 scene points, horizon 10:

| K (one batch) | small-droid ms/cand | large ms/cand | small VRAM | large VRAM |
|---|---|---|---|---|
| 1 | 27.4 | 92.0 | 1.39 GiB | 4.98 GiB |
| 8 | 11.6 | 49.3 | 1.82 GiB | 5.80 GiB |
| 32 | **10.5** | **45.4** | 3.27 GiB | **8.59 GiB** |

Weights 1.32 vs **4.85 GiB** resident; params 354 M (303 M DINOv3 / 51 M
dynamics) vs 1288 M (303 M / **985 M**). Marginal cost is ~0.12 GiB per
candidate, so K is bounded by latency long before memory.

Perception is unchanged at **41 ms** — DINOv3 is the same 303 M in both, as
predicted. Through the bridge: `observe` 133 -> **206 ms** warm, `rollout` K=1
46 -> **104 ms**, K=20 24 -> **55.5 ms/candidate**. Batching still buys only
1.9x.

**So `large-droid+behavior` costs 4.3x the latency and 2.6x the VRAM for 15x
the accuracy.** It is the right checkpoint on this trade, and it fits an AGX
Orin at K=32. The three-way choice flagged earlier resolves in its favour.

**But the MPPI configuration does not survive.** K=32 x 3 chunks = 96 forwards
is 96 x 45.4 = **4.36 s per tick** on the GB10 (1.01 s on small-droid). Both
are unusable, and the Orin is several times slower again, so this is not fixed
by picking the smaller model. At a 500 ms tick the budget is ~11 forwards:
**K=8, one 10-step chunk, 394 ms, 2.5 Hz** is what the measurement supports
here. Set K and the chunk count from a measured budget on the target, as the
deployment section already says — do not carry K=32 x 3 over.

### 4. The planner still stalls — and the model is now exonerated

`scripts/run_planner.sh --suite libero_goal --task-id 0 --no-cgn`, 25 ticks:

| | small-droid + MPPI | large-droid+behavior + MPPI |
|---|---|---|
| progress closed | 129.4 mm | **106.4 mm** |
| owed at the end | ~30 mm | **53.6 mm** |
| `_check_success()` | False | **False** |

**A 15x more accurate model made the closed loop slightly WORSE.** That kills
the standing explanation for the stall — "below ~30 mm of true motion
PointWorld loses to predict-no-motion, and the planner cannot descend a
gradient the model cannot resolve" — twice over: it was measured on
`small-droid`, and the stall survives a checkpoint that predicts moved points
to 3.92 mm. Neither of the other two suspects survives either: magnitude bias
is 1.0x and batched noise is 0.46 mm.

**So the stall is in the planner or the execution, not the model.** The
per-tick trace points at execution: commanded point motion vs achieved runs
`38.9 -> 13.3`, `36.5 -> 38.6`, `15.2 -> 17.5`, `12.9 -> 4.1` mm — the tracking
error is the same size as the command. The plan is made in joint space and
executed through the simulator's OSC *pose* controller
(`tests/plan_pointworld.py`, "FAITHFUL EXECUTION"), and `owes` moves backwards
on 9 of 25 ticks. Instrument the joint-space command against what the joints
actually did before touching the cost function.

### 5. Grounding runs EVERY tick, unconditionally — not once, and not event-driven

`tests/plan_pointworld.py:263-270`, inside the tick loop:

```python
mask = env.points_in_geoms(points0, spec.geoms(env), margin=0.01)
goal_idx = np.flatnonzero(mask)
owed = spec.offset(env)
state["goal_pos"] = points0[goal_idx] + owed
```

Every tick re-derives the mask from a fresh cloud and re-anchors the goal. That
is free and exact today only because it is an ORACLE — MuJoCo geoms, not
perception — so the cost of re-grounding is zero and the answer never drifts.
Both properties disappear with a real grounder, and the guide names this
exactly: its occluded-grasped-object test fails a design that re-runs full
grounding every control step to rediscover the object, and asks instead that
task-relevant points stay ATTACHED until release, with re-grounding only on
grasp, release, occlusion or repeated planning failure.

Two consequences, and the second may bear on the stall above:

1. **There is no event layer and no persistent target-point set to attach.**
   `goal_idx` is a fresh index array each tick; nothing tracks identity across
   ticks. That has to exist before any real grounder is wired in, or the
   grounder is queried at 2 Hz.
2. **The target point SET changes tick to tick** as visibility changes, so the
   MPPI cost is evaluated over different points each time. The goal is
   self-correcting (`current + owed`), but the set is not stable, and an
   unstable cost domain is a candidate cause of `owes` moving backwards.

---

## THE PLANNER, DIAGNOSED (2026-08-03) — the world model was not in the loop

Iterated on `small-droid` deliberately: it is 4.3x faster per candidate, and
the checkpoint swap moved the loop the WRONG way, so the model is not the
variable. Every number below is `POINTWORLD_CKPT=small-droid`.

### The diagnostic that found it

`tests/plan_pointworld.py` now reports, per tick: `cos best/far` (the direction
of the candidate PointWorld ranked first, at step 1 and over the whole
horizon), `cos cmd -> got` (what was commanded, and what the arm achieved),
`track` (joint tracking error), and **`rho`** — the rank correlation between
the total cost MPPI weights by and the task cost PointWorld returned.

**`rho` was +0.03 over 15 ticks.** The world model contributed nothing to the
ranking. Meanwhile `cos far` was above +0.8 on most ticks, so the task cost
itself was well shaped — the ranking test's result does hold inside the loop.
Three terms were drowning it, each found only after the previous was fixed:

| # | term | size vs task cost | fix |
|---|---|---|---|
| 1 | joint-limit penalty | up to **1500x** | clamp limits in `dynamics` |
| 2 | perturbation cost via the FINGER joints | ~**1000x** | plan arm joints only |
| 3 | perturbation cost via the nominal `U` | ~**10x** | `--warm-start 0` |

1. **`W_LIMIT` charged `over` at every one of 30 steps.** 31 of 32 candidates
   breached a limit on one tick. Clamping in `dynamics` makes candidates
   feasible by construction; `viol` now reads 0 and `W_LIMIT` is a tripwire.
2. **The fingers had the largest weight because they had the smallest noise.**
   MPPI's perturbation cost is `lambda * sum(U * noise / sigma^2)`, so
   `SIGMA_FINGER = 0.001` gives an inverse-variance weight of **1e6**. The two
   joints deliberately excluded from the search dominated the cost. The action
   space is now the 7 arm joints; the fingers are held and reinserted before FK.
3. **After the `exp(-cost/lambda)` weighting the lambda CANCELS out of the
   perturbation term**, so no temperature can out-weigh it. Its size is
   `~|U|/sigma * sqrt(T*A)`, and the warm start random-walked `U` out to the
   `3*sigma` bound, giving `3*sqrt(210) ~ 43` against a task exponent of ~4.

With all three fixed, **`rho` is +0.93 to +0.98** and progress is near
monotonic — one backwards tick in 15, against 9 of 25 before.

### And the temperature had to come back

`beta` was hardcoded `lambda_=0.02` in the `pytorch_mppi` migration, which
deleted the bisection this file had already established was necessary. It sat
at ESS 14-26/32, the "averaging uniformly" failure mode. `solve_temperature()`
restores the bisection for ESS = K/4; `beta` now solves to 0.005-0.018 per tick
and ESS pins at 8.0/32.

### Where it stands, and the uncomfortable comparison

| planner | progress | success |
|---|---|---|
| straight-line candidates + argmin (pre-MPPI) | 160 -> **19.5 mm** | **True**, 18 ticks |
| hand-rolled MPPI + bisected temperature | 129.4 mm closed | False, stalls ~30 mm short |
| `pytorch_mppi`, as found today | 41.5 mm closed / 15 ticks | False |
| `pytorch_mppi` + the three fixes above | **45.2 mm closed / 15 ticks** | False |

**The simple planner that was replaced is still the only one that solved the
task.** The fixes above recover most of what the migration lost, but they do
not close the gap to the version that worked, and that is the honest state.

A plausible reason, and it is measured rather than guessed: the model ranks a
SMALL STRUCTURED SET extremely well — `rank_actions_pointworld.py` separates 20
straight-line candidates cleanly, and the old planner picked `+Y` out of 26
directions on every tick. MPPI is instead searching **30 steps x 7 joints =
210 dimensions with 32 samples**, and `sqrt(210)` is exactly what makes term 3
above unbeatable. The search space, not the model, is the difficulty.

Shortening the horizon does NOT help — it shrinks the task cost's own spread
faster than it shrinks the noise:

| config | progress / 15 ticks |
|---|---|
| horizon 30, warm 0.0 | **45.2 mm** |
| horizon 10, warm 0.5 | 16.5 mm |
| horizon 10, warm 0.0 | 3.4 mm |

### CONTROLLED COMPARISON, same service / checkpoint / oracle goal

No LLM or VLM is in any of this -- verified by grep, every mention in the
planner path is a comment. The goal is handed over perfectly by `task_spec.py`.

| planner | checkpoint | opened | stalls at | success |
|---|---|---|---|---|
| MPPI, 15 ticks | small-droid | +45.2 mm | — | False |
| MPPI, **60 ticks** | small-droid | **-1.7 mm** | never converges | False |
| structured + argmin, 25 ticks | small-droid | **140.2 mm** | 21.1 mm owed | False |
| structured + argmin, 30 ticks | large-droid+behavior | **121.5 mm** | 42.8 mm owed | False |

**MPPI does not converge.** The +45.2 mm at 15 ticks was drift; at 60 ticks it
is -1.7 mm and `owes` pins at 161.7 (the drawer's closed stop). The three fixes
above made `rho` correct (+0.93 to +0.99, sustained over all 60 ticks) and did
NOT make the planner work. Correct cost ranking was necessary, not sufficient.

### The endgame stall is the CANDIDATE SET, not the model

Both structured runs stall by choosing `still` forever. That was previously
attributed to the model's error floor. It is not:

* At the stall the margins are **3.0-3.2 mm** on `large-droid+behavior`, against
  a batched noise floor of **0.46 mm**. The planner is confidently choosing
  `still`; it is not a coin flip.
* `RATES_MM = (8, 16, 26)` mm/step over a 10-step horizon means the slowest
  moving candidate commands **80 mm of travel**. There is NOTHING between 0 and
  80 mm. With 42.8 mm owed, `+Y @ 8` overshoots by ~37 mm (cost 37.3) and
  `still` undershoots by 39.7 (cost 36.4). `still` wins, nothing moves, and the
  state never changes again -- a fixed point.

So the candidate set has no resolution in the endgame. This also explains why
the ACCURATE checkpoint stalls EARLIER (42.8 vs 21.1 mm owed): a model that
predicts the overshoot correctly rejects the only moving candidate sooner. The
better model exposes the quantisation bug rather than causing it.

**Fix is a rate that shrinks with the remaining distance**, or a terminal cost
evaluated at the best point along the trajectory rather than only at its end.

## SOLVED (2026-08-03) — the planner is reimplemented and the task passes

```bash
POINTWORLD_CKPT=large-droid+behavior \
    scripts/run_planner.sh --suite libero_goal --task-id 0 --no-cgn --ticks 40
# 159.9 -> 19.9 mm, LIBERO _check_success() True, 21 ticks
```

`+Y` chosen on **every** tick at `cos 1.00`, monotonic, no freeze. Progression
of the whole session, all on the same task, service and oracle goal:

| planner | checkpoint | closed | success |
|---|---|---|---|
| `pytorch_mppi`, 60 ticks | small-droid | -1.7 mm | False |
| structured + argmin (old file) | small-droid | 140.2 mm | False |
| structured + argmin (old file) | large-droid+behavior | 121.5 mm | False |
| **reimplemented, 21 ticks** | large-droid+behavior | **140.0 mm** | **True** |

### The four things that had to be true at once

1. **The model must rank the candidates, not the control cost.** `rho` +0.03 ->
   +0.98. Joint limits clamped in the dynamics rather than penalised; fingers
   out of the action space; no MPPI perturbation cost.
2. **Rates scale with the distance owed** (`owed / horizon` is always in the
   set). This is what removed the endgame freeze -- see the quantisation
   analysis above. The rate anneals itself 24.0 -> 3.2 mm/step with nothing
   scripted.
3. **Cost read at every step** (`cost_steps`, added to the server), so a
   candidate is judged where it ARRIVES. Argmin is over both candidate and step.
4. **Execution must track.** `kp` 400 -> 2000 took delivered motion from ~20%
   of commanded to ~75-97%. At kp=400 the arm settles where `kp * err` balances
   the load, so the step budget was irrelevant -- a fixed fractional undershoot
   that is indistinguishable from a world model that under-predicts.

Directions become joint deltas by damped least squares on the **robot-point**
Jacobian (`point_jacobian`), so no end effector is ever named and the method
carries to the dual-arm UR7e unchanged.

### Known, and deliberately left

**Convergence is geometric, not linear** — each tick closes a fixed FRACTION of
what remains, because the rate is derived from the remaining distance. Late
ticks move 1.5 mm where early ticks move 20. A floor under the rate would fix
it; it was not needed to pass, and a floor risks reintroducing the overshoot
that the scaling exists to prevent. Measure before adding one.

`--ticks 40` was the budget; it finished in 21.

### Where the tick time actually goes — measured, not assumed

Per-tick, 73 candidates, one 10-step chunk (the planner now prints `fk` and
`model` on every line):

| | large-droid+behavior | small-droid |
|---|---|---|
| candidate FK (CPU, MuJoCo) | 158-216 ms | 158-216 ms |
| PointWorld rollout (GPU) | ~4100 ms | ~1050 ms |
| **FK as a share of the tick** | **4%** | **15%** |

**The GPU is already doing 95% of the work.** `at_configs()` batches the
save/restore that `at_config` was doing per configuration -- 1606
`sim.forward()` calls became 804 -- which took FK from 252-308 ms to 158-216,
verified equal to the reference at **0.00006 mm** (float32 rounding).

**Raising the batch does NOT help**, which is worth writing down because it is
the obvious thing to try: `--max-batch` 8 / 16 / 32 measured ~4100 / ~3700-4700
/ ~3900-5100 ms. The model is compute-bound at ~55 ms/candidate, not
launch-bound. `MAX_BATCH` is now an env var on `run_planner.sh`; 8 is fine.

**So the only real lever on tick time is FEWER CANDIDATES**, 73 x 55 ms being
the whole budget. Coarse-to-fine would do it: score 18 directions at one rate,
then a few rates along the winner -- 22 forwards instead of 73, a 3.3x cut.
Not done, because it changes the search and must be re-validated for SUCCESS
rather than for speed.

Porting the FK to the GPU is worth ~200 ms of a 4.3 s tick and is not the
place to spend effort. On the Orin, where the model is several times slower
again, that share only shrinks.

## REAL-TIME (2026-08-03) — 1.3 s -> ~0.18 s per tick, still solves

Three optimisations, each measured, none changing the verdict. Steady-state
tick on `small-droid`:

| stage | before | after |
|---|---|---|
| `observe` (round trip) | 206 ms | **59 ms** |
| candidate FK (CPU) | 216 ms | **9 ms** |
| PointWorld rollout | 1078 ms | **108 ms** |
| **total compute** | ~1.5 s | **~176 ms (5.7 Hz)** |

Solved in **14 ticks**, 159.9 -> 19.8 mm, `_check_success()` True.

1. **`observe` ran a full model forward on every observation** to set
   `_current_domain_indices` -- a whole PTv3 trunk and dynamics head plus a
   SECOND DINOv3 encode, on a call whose only job is to encode the scene once.
   `encode_scene_features` derives those indices itself when the attribute is
   absent (`base.py:422`), so the forward is a one-time graph warmup. Encode
   186 -> 40.7 ms, which is exactly the 41 ms the bench always reported for
   perception -- the gap was never real.
   **Trap:** the indices are cached, and a preceding `rollout` leaves them
   sized to its K candidates, so `observe` at B=1 asserts. They must be dropped
   before the cheap path. That is why the full forward looked necessary.
2. **The candidate set is warm-started.** The direction is highly persistent --
   `+Y` won on every tick of every successful run -- so a full 18-direction
   sweep re-derives an unchanged answer at 73 forwards. Between sweeps only the
   incumbent (4 rates) and its 3 nearest neighbours (1 rate each) plus `still`
   are scored: **K=8**. A full sweep runs on tick 0, every `--resweep` ticks
   (8), and whenever the incumbent goes stale -- `still` wins, or the best cost
   worsens by >1.5x.
3. **`at_configs()` batches the FK save/restore** -- 1606 `sim.forward()` calls
   became 804, and K=73 -> 8 did the rest. Verified equal to the per-config
   reference at **0.00006 mm**.

**Raising `--max-batch` does NOT help** (8/16/32 -> ~4100/~3700-4700/~3900-5100
ms at K=73): the model is compute-bound at ~55 ms/candidate, not launch-bound.
`MAX_BATCH` is an env var on `run_planner.sh`; 8 is right.

**The rollout is per-CANDIDATE, not per-step** -- one forward predicts the whole
10-step chunk, and perception is reused across candidates via
`encoded_scene_feat0`. So tick cost is K x 11 ms, and K is the only lever.

Resweep ticks still cost ~1.2 s. If that jitter matters, drop `--resweep` to 0
(never re-sweep, relying on the neighbour cone) or cut K further -- incumbent
plus `still` alone is K=5.

`videos/planner_drawer.mp4` -- the solved episode. `execute_joint_positions`
now takes `capture_every`, because the planner drives joints directly and
bypasses `env.step()`, which is the only place frames were ever recorded; a
planning run rendered an EMPTY video before this.

### STALE OBSERVATIONS — found by watching the video, not by any metric

`_refresh_obs()` called robosuite's `_get_observations()` WITHOUT
`force_update=True`. That call only reads each observable's CACHED value; the
cameras are re-rendered by `_update_observables`, which `env.step()` invokes on
its own schedule and nothing else does. `execute_joint_positions` writes
torques and steps `sim` directly, so every planner tick after the grasp read
**the images from before the motion**.

| | px changed, start of pull vs end |
|---|---|
| planner video, before | **19** |
| planner video, after | **15111** |
| `test_drawer_open.py` (drives `env.step`) | 13770 |

**This was not a rendering artefact.** `get_cam_obs()` reads `self._last_obs`,
and `live_observation` builds BOTH the RGB and the scene point cloud from it,
so PointWorld was scoring candidates against a scene frozen at grasp time while
the robot points were live. Fixed in `_refresh_obs`.

**Why it still solved the task, and why that is the uncomfortable part:** the
goal comes from `task_spec.py`, i.e. the simulator, not from the frozen images.
`owed` shrank correctly every tick regardless of what the cameras showed, so the
direction stayed right and the drawer opened. **The oracle masked a perception
bug completely.** Results before and after the fix differ only in the third
significant figure (cost 23.6/19.3/16.3/14.6 -> 23.1/18.9/16.3/14.7, both 14
ticks to success).

On a real grounder this would have been fatal rather than invisible: the mask
and the goal would both be anchored to a scene that never updates. It is direct
evidence for the survey's point that grounding must be built only once the
planner is trusted -- and a reminder that "the task passed" is not evidence the
perception path works.

**Rule, and it is the session's cheapest lesson:** a metric that is fed by the
oracle cannot audit the path that bypasses the oracle. Watch the video.

### 86% OF THE SIM TIME WAS THE ARM STANDING STILL

Watching the video raised "why is it not a smooth motion", and the profile
answered it. Motion within each 1500-step tick, binned by step:

```
  steps     0-  100: ##########################################   <- all of it
  steps   100-  200: ################
  steps   200-  300: ######
  steps   300-  400: #
  steps   400- 1500: (nothing)
```

`execute_joint_positions` broke on `|err| < tol` with `tol=1e-3`. **A PD holding
a load settles at a NONZERO steady-state error**, so that condition is
unreachable while the arm pulls a damped joint, and every tick burned its whole
1500-step budget to grind at a residual it could not close. The motion was over
by step ~300.

The exit test is now whether the error is still IMPROVING -- sampled every 50
steps, break if it came down by less than 2% and the joints have nearly
stopped. Settling, not an absolute threshold.

| | before | after |
|---|---|---|
| video length | 2139 frames / 107 s | **282 frames / 14 s** |
| frames with visible motion | 14% | **74%** |
| ticks to success | 14 | **13** |
| closed | 140.1 mm | **142.1 mm** |

It got slightly BETTER, not just faster: the wasted steps were letting the
grasp creep. `videos/planner_drawer.mp4` now shows a continuous pull.

**This is the same lesson as the stale observations, twice in one session:
neither defect moved any number the planner printed.** `track` read 0.001-0.003
rad throughout and looked fine; success was already True. Both were only
visible by watching the episode.

## PERCEPTION ABLATION (2026-08-03) — the drawer task cannot validate a grounder

`--freeze-scene` pins every SCENE channel (`scene_flows`, `scene_colors`,
`scene_normals`, `rgb`, `depth`) to tick 0 for the whole episode while the
robot points stay live. It reproduces the stale-observation bug deliberately.

```bash
scripts/run_planner.sh --suite libero_goal --task-id 0 --no-cgn --freeze-scene
```

| checkpoint | fresh scene | FROZEN scene | difference |
|---|---|---|---|
| small-droid | 142.1 mm, 13 ticks, **True** | 141.1 mm, 13 ticks, **True** | 1.0 mm |
| large-droid+behavior | 140.8 mm, **True** | 140.5 mm, **True** | **0.3 mm** |

**Blinding the planner for an entire episode costs 0.3 mm on a 140 mm task.**
Perception is not load-bearing here, on either checkpoint.

### Why, and the mechanism is worth understanding before designing around it

`goal_pos = points0[goal_idx] + owed`. Freeze `points0` and the goal is stale by
exactly however far the drawer has travelled -- but the model's PREDICTED points
are computed from the same frozen cloud, so both sides of the cost sit in the
same stale frame and **the error cancels**. The loop is really doing relative
control: "move these points by `owed`", where `owed` comes from the simulator
every tick. Absolute perception is not required for that, so it contributes
almost nothing.

This is also exactly what the axis result predicts: a model that predicts
184-199 mm of motion in ALL 18 directions is dominated by the robot points, not
by scene content.

### Consequence for the language layer

**Do not build the grounder against this task.** Its whole job is to supply the
mask and target FROM THE SCENE, and here a completely frozen scene still
succeeds -- so a grounding failure would be invisible, which is the usual trap
inverted and worse. Survey item 2 is now a hard prerequisite for item 3, not a
parallel nicety: get a task where the scene is load-bearing (an unrestrained
push, where "follows the hand" is WRONG and contact decides), re-run this
ablation there, and only build grounding once freezing the scene visibly breaks
the task.

That ablation is now a permanent gate, and it is one flag.

### PUSH TASK — attempted, PARKED (2026-08-03)

`libero_goal/5` "push the plate to the front of the stove" is a real LIBERO
push with its own `_check_success()`, and `task_spec.push_to()` +
`region_centre()` derive the 259 mm goal from LIBERO's own BDDL region. Those
are kept and are correct.

**The staging is not.** `tests/test_push_control.py` is the control -- scripted
straight-line push, no model, no planner -- and it reports:

```
staged  : ee [0.062 -0.078 1.067]  (wanted [0.114 -0.173 0.901], off by 197.7 mm)
step  0..19: plate moved 0.0 mm     (ee drifts 1 mm across 20 commanded 20 mm steps)
verdict : pushing is NOT achievable with this staging -- a SETUP failure
```

So the planner's 0.0 mm push and flat cost landscape were **not** a model
finding. Without this control they would have read as "PointWorld cannot
predict contact", which is exactly the confident-and-wrong verdict `NOTES.md`
section 4 warns about. The control is the finding.

**The bug, for whoever retries:** `env.finger_offset()` measures the fingertip
offset along the CURRENT approach axis using the CURRENT orientation. Calling
it before commanding the push orientation returns -4 mm instead of the finger
length, so the grip SITE was placed at plate height and the fingers were driven
through the table. Compute it after the orientation is set, or derive it
geometrically.

**Parked anyway**, and not only for that bug: a 20 mm-thick plate pushed side-on
by a Panda gripper needs the fingertips within a couple of centimetres of the
table, which is a fragile setup to build a research claim on. If a load-bearing
task is wanted later, prefer a TALL free object (the wine bottle, a soup can in
`libero_object`) where contact is unambiguous.

### GROUNDING CAN START WITHOUT IT — the gate was too strict

The ablation forbids ONE thing: reading end-to-end drawer success as evidence
that a grounder works. It does not forbid MEASURING THE GROUNDING DIRECTLY,
which is a stronger test and needs no load-bearing dynamics task:

| grounder output | oracle to score it against |
|---|---|
| target object / mask | `env.points_in_geoms(spec.geoms(env))` -- IoU |
| goal position | `spec.offset(env)` -- error in mm |

Both are available on every LIBERO task today. Measure grounding alone, in mm
and IoU, and the end-to-end question does not arise until it passes.

## GROUNDING, STARTED (2026-08-03) — features exposed, and a negative result

### The bridge now serves per-point DINOv3 features

New `features` op: `pw.features()` returns the (Ns, C) tensor `observe`
already computed, so it costs a memcpy rather than a second encoder, and it is
the SAME points in the SAME order and frame the planner masks over -- which is
the only reason a mask derived from it can be handed straight back as
`goal_idx`.

**Correction: the features are 256-d, not 128.** `NOTES.md` section 6 says 128;
measured (4052, 256), std 3.687, no zero rows.

### A global feature-similarity mask will NOT work — measured ceiling

For each object: take the ORACLE mask, average its features into a prototype,
and find the threshold on cosine similarity that maximises IoU. That is the
best a similarity query could possibly do, because the query is derived from
the answer.

| target | pts | on-target | off-target | **best possible IoU** |
|---|---|---|---|---|
| `wooden_cabinet_1_cabinet_middle` | 102 | +0.915 | +0.767 | **0.287** |
| `wooden_cabinet_1_cabinet_top` | 99 | +0.966 | +0.768 | 0.500 |
| `akita_black_bowl_1` | 89 | +0.938 | +0.812 | 0.345 |
| `plate_1` | 104 | +0.965 | +0.837 | **0.674** |
| `wine_bottle_1` | 64 | +0.918 | +0.796 | 0.458 |
| `flat_stove_1_button` | 13 | +0.976 | +0.807 | 0.429 |

On/off separation is a real but small ~0.15 of cosine, and the ceiling is
0.29-0.67 IoU. **A real text or image query would do WORSE than every number in
that column.** The drawer is worst, which makes sense: three drawer faces on
one cabinet are appearance-identical, and telling them apart is a spatial
question that appearance features cannot answer in principle.

This refines `NOTES.md` section 6's "features are not the missing part; a query
is". Half right: the features exist and carry signal, but a GLOBAL query over
them tops out well below a usable mask.

### So the mask should come from projection, not from feature search

`scene_featurizer.py` already projects every scene point into both cameras with
visibility and depth checks -- exactly and for free. So the cheap path is:

    VLM/pointing model -> 2D box or point
      -> project scene points into it            (exact, already built)
      -> features only to clean up the edges     (local, not a global threshold)

which is also why the authors' own GUI uses SAM2: it is SPATIALLY SEEDED. The
next measurement is the ceiling of that path -- IoU of a projection-only mask
given an ORACLE 2D box. That bounds the whole approach and needs no VLM.

### Backend: Qwen3-VL-2B-Instruct, local, no API key

Cached at `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct`;
`QwenLocalBackend` loads it in-process via transformers 5.14.1 (present in both
venvs). No Anthropic/OpenAI credentials are configured in this environment and
none are needed for this path.

## GRASP PERSISTENCE (2026-08-03) — and what it unblocked

`holding(env, spec)` uses MuJoCo's contact list (`env._contacting_objects()`),
not finger width, because width cannot say WHICH object is held. On loss the
planner re-grasps, up to `--max-regrasp`.

It returns True when the answer is unknown -- a drawer is a fixture and is not
in `_object_geom_ids`, so re-grasping it every tick because the contact list
cannot see it would be far worse than not checking.

**`libero_goal/1`, "put the bowl on the stove":**

| | before | after |
|---|---|---|
| owed, while held | stuck at **309.1 mm** from tick 9 | **16.0 mm** by tick 44 |
| closed | 76.8 mm | **275.5 mm** |
| re-grasps | n/a | 2 (both on tick 0-1) |

Before, the bowl slipped at tick 8 and the loop spent 21 ticks confidently
commanding motion with an empty gripper -- `owes` frozen at exactly 309.1 while
`cos far` drifted to -0.77. **Every number it printed looked like a planner
working.** The only tell was that `owes` did not move while `pts got` did, and
nothing was checking that.

### Two spec bugs it exposed, both still open

1. **The release drops the bowl.** It reaches 16.0 mm owed while held, then
   `release_at_end` opens the jaw and `owes` jumps to **110.4 mm** -- `move_to`
   aims 30 mm ABOVE the destination's top face, so the bowl is dropped from
   30 mm and rolls. Either lower before releasing, or make the clearance a
   function of what is being placed.
2. **The registry targets the wrong region.** `("libero_goal", 1)` is
   `move_to("akita_black_bowl_1", "flat_stove_1_button")`, but LIBERO's own
   predicate is `('on', 'akita_black_bowl_1', 'flat_stove_1_cook_region')`. The
   button is not the cook surface, so `_check_success()` could not pass even
   with a perfect placement.

**CORRECTED after measuring** -- my first diagnosis, "wire `for_task` to fall
back to `from_bddl` and this disappears", was WRONG. `from_bddl` fails too:

```
from_bddl ERR KeyError "'flat_stove_1' is not in this scene.
  Available: [... 'flat_stove_1_button' ...]"
```

The predicate names `flat_stove_1_cook_region`; `_resolve_region` reduces that
to `flat_stove_1`, which is not a body with geoms. So the hand-written override
was a WORKAROUND for a resolution failure, not a careless typo -- and the only
resolvable stove part was the control knob.

**Root cause:** `_cache_robot_geoms` registers movable objects plus fixture
links that have a JOINT (`_articulated_links`). The stove's `button` has a
hinge, so it is registered; `burner` and `burner_plate` are static, so they are
invisible to every naming path. **Any "place X on a static surface" task is
unnameable**, which is most of them.

**Fixed** -- `object_geoms` falls back to a body-name lookup, and
`placement_surface()` resolves a BDDL region to the body to place on.
`for_task(suite, task_id, env)` now falls back to `from_bddl`.

**And the selection rule had to be measured, not guessed.** My first version
took the highest top-z, which picks the knob:

```
flat_stove_1_burner        top 0.930   361 cm2
flat_stove_1_burner_plate  top 0.930   338 cm2
flat_stove_1_button        top 0.965   107 cm2   <- tallest, and wrong
```

(An earlier probe reported burner_plate at 0.992 because it ignored geom
ROTATION; with rotation it is 0.930. That mistake is what made top-z look
right.) Area is the discriminator: you place things on broad surfaces and turn
narrow ones. Verified:

| | before | after |
|---|---|---|
| surface | `flat_stove_1_button` | **`flat_stove_1_burner`** |
| bowl target | [-0.409, 0.202, 1.024] | **[-0.254, 0.202, 0.990]** |
| error vs hob | **155 mm** | 0 mm in xy |
| owed at start | 387.5 mm | 266.7 mm |

**RE-RUN, and it SOLVES:**

```
spec    : move akita_black_bowl_1 to flat_stove_1_burner   (owes 266.7 mm)
released: owes 51.7 mm
progress: 265.4 -> 51.7 mm (+213.7 mm closed)
success : True   (LIBERO's own _check_success)   regrasps: 2
```

`videos/planner_bowl.mp4`. Second LIBERO task solved by planning through
PointWorld.

**And it was hiding behind an exception.** The first corrected run crashed with
`EpisodeFinished` inside `env.open_gripper()` and printed no verdict at all --
but `EpisodeFinished` is **LIBERO reporting SUCCESS**, and releasing is exactly
when a place task completes, so the terminating exception is the MOST likely
outcome of that line. The trap list in `NOTES.md` says this has now bitten four
times. It is caught at the release site now.

## GROUNDING BASELINE (2026-08-03) — Qwen3-VL-2B, zero-shot: UNUSABLE

`tests/measure_grounding.py` scores the grounder ALONE, in millimetres against
the oracle, without running the loop. Two things are scored separately because
they fail independently and need different fixes: WHICH points were named
(on-target %) and WHERE they should go (error in mm, and cos against the true
direction).

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/measure_grounding.py \
    --suite libero_goal --task-ids 0 1
```

| task | on-target | goal error | cos | task owes |
|---|---|---|---|---|
| `/0` open the drawer | -- | replied **"Done"** | -- | 160.0 mm |
| `/1` bowl on the stove | **0.0%** | **607.2 mm** | **-0.45** | 266.7 mm |

**Worse than doing nothing.** On the drawer it declares the task already
complete and returns no targets at all. On the bowl it names keypoints of which
NONE are on the bowl, and asks for a displacement 2.3x larger than the task
needs, pointing the wrong way (`cos -0.45`).

This is the second time VLM grounding has failed on exactly this (`NOTES.md`
section 4: a keypoint 153 mm from any drawer handle). It is a baseline, not a
verdict on the approach -- three things are all suspect at once and the
measurement cannot yet separate them:

1. **The model is 2B.** Qwen3-VL-2B-Instruct is what is cached locally; spatial
   keypoint grounding is exactly where a small VLM should be weakest.
2. **The prompt is KUDA's**, asking for offsets between marked keypoints. The
   design basis is `docs/pointworld_pipeline_implementation_prereqs.md`, whose
   layer is a SCENE GRAPH plus a goal formalizer -- object-level, not
   keypoint-level. That prompt has not been written.
3. **Keypoints are farthest-point samples**, so nothing guarantees one lands on
   the target at all -- and on the bowl, none did. That is a proposal-coverage
   failure upstream of the VLM, and it is measurable separately.

Fix (3) first: it is free, it is ours, and while it holds no VLM can succeed.

Also fixed here: `QwenLocalBackend` defined `query()` but not `__call__`, while
`ground()` and ReKep both call the backend directly -- so it raised
"'QwenLocalBackend' object is not callable" at the point of use rather than at
construction. Every other backend inherits `__call__` from `RetryingBackend`.

## PERCEPTION WORKS (2026-08-03) — a LIBERO task solved with NO oracle target

```bash
POINTWORLD_CKPT=small-droid scripts/run_planner.sh \
    --suite libero_goal --task-id 1 --ground --ticks 45
# success : True   (LIBERO's own _check_success)
```

`videos/planner_bowl_grounded.mp4`. The goal comes from the point cloud and one
VLM call; `task_spec` is used only to PRINT progress and is never an input.

### What made it work: object proposals, not keypoints or feature search

`src/rekep_libero/scene_graph.py`. Cluster the cloud above the table by
connectivity, describe each proposal in the guide's schema, number them on the
image, and ask the VLM for `{target, destination, offset_cm}`.

Proposal quality against the oracle, and it beats the DINOv3 ceiling on every
free-standing object:

| object | cluster IoU | recall | global-DINOv3 CEILING |
|---|---|---|---|
| `plate_1` | **0.889** | 0.923 | 0.674 |
| `wine_bottle_1` | **0.852** | 0.963 | 0.458 |
| `akita_black_bowl_1` | **0.716** | 0.951 | 0.345 |

Connectivity does what appearance could not. The cabinet's three drawers fuse
into ONE cluster (recall 1.0, IoU 0.13) -- correct, because the cabinet is one
connected body; separating drawers is articulation, not segmentation.

### Grounding measured alone, in millimetres

`tests/measure_grounding.py`, Qwen3-VL-2B, zero-shot:

| prompt | task | on-target | goal error | cos | owed |
|---|---|---|---|---|---|
| KUDA keypoints | `/1` bowl | 0.0% | 607.2 mm | -0.45 | 266.7 |
| **guide scene-graph** | `/1` bowl | **74.3%** | **24.9 mm** | **+1.00** | 266.7 |
| guide scene-graph | `/0` drawer | 14.1% | 160.0 mm | +0.00 | 160.0 |

**24x better on the goal vector**, and the direction is exact. The keypoint
prompt failed for a reason upstream of the VLM: farthest-point samples put NONE
of 24 keypoints on the bowl, so no model could have named it.

### Honest limits, and they matter

* **The grounded displacement was 464.7 mm against a true 279.4 mm** -- a 66%
  overestimate. It still succeeded because the direction was adequate, placing
  has tolerance, and LIBERO terminated the episode on success at tick 12. Do
  not read this as a calibrated goal.
* **The `cos` columns in the tick trace look poor (+0.05)** because they are
  measured against the ORACLE's owed vector, which is no longer what the
  planner is chasing. They are the wrong diagnostic in `--ground` mode.
* **The drawer does not ground.** `cos +0.00` and error exactly equal to owed
  means it asked for ~zero displacement: "open the drawer" is not expressible
  as target+destination, and its cluster is the whole cabinet. This needs an
  ACTION TYPE in the schema (`slide` with a direction), and the direction is
  the axis -- which the axis-discovery result says the model cannot supply.
* **The grasp is still placed by Contact-GraspNet**, not planned.

## COLLISION (2026-08-03) — there was no cost term at all, now there is one

"The robot kept bumping into other objects" -- checked, and it is **not an
artifact**. The cost is `||predicted[goal_idx] - goal_pos||` over the TARGET
points only, so a candidate that sweeps a bottle off the table scores exactly
as well as one that does not. `grep` for collision/sdf/obstacle in the planner
returns nothing. The guide names this as a subsystem that must be added
externally.

**It does not have to be geometric.** The model already predicts the WHOLE
scene, so disturbing a bystander is just predicted motion where none was asked
for. `--avoid-weight` adds `w * mean||pred[avoid] - pred[avoid, 0]||`.

Referenced against `pred[:, 0]` rather than the observed cloud on purpose: the
model leaks spurious motion onto static points (14.6 mm small, 60.6 mm large),
and differencing WITHIN the prediction cancels the part of that bias common to
every candidate, leaving what the action did.

Measured on `libero_goal/1`, with a ground-truth `disturb` diagnostic that
tracks every bystander's centre from start to end:

| | bystander disturbance | progress | success |
|---|---|---|---|
| no term | **15.3 mm** (plate 15) | 209.2 mm | True |
| `--avoid-weight 1.0` | **8.9 mm** (plate 9) | 216.5 mm | True |

**42% less disturbance and slightly MORE progress** -- it is not a trade here,
because a candidate that ploughs through the plate was also wasting motion.

### What this does NOT fix

The diagnostic measures OBJECT DISPLACEMENT, and 15 mm on one plate is modest.
If the arm visibly sweeps over or past objects without moving them, that is
**arm-body clearance**, which no term covers: the cost only ever sees scene
points, never the arm's own volume. The env already has ReKep's machinery for
it -- `get_sdf_voxels()` and `get_collision_points()` -- and that is the right
tool for clearance, as opposed to disturbance. Untouched so far.

## MULTI-STAGE (2026-08-03) — machinery works, the GRASP does not

`--stages` decomposes LIBERO's multi-predicate goals and runs them in order.
`stages_from_bddl` also fixes an ordering trap: `libero_10/3` lists

    [['close', 'white_cabinet_1_bottom_region'],
     ['in', 'akita_black_bowl_1', 'white_cabinet_1_bottom_region']]

so closing is listed FIRST, and doing it in that order makes the second
predicate unreachable. Rule: articulated `open` first, `close` LAST, everything
else in listed order. Verified on three tasks; `close_articulated` added as the
mirror of `open_articulated` (same axis arithmetic, opposite stop).

**`libero_10/0` "put both the alphabet soup and the tomato sauce in the
basket": both stages decomposed and sequenced correctly, and both failed at the
GRASP** -- 4 re-grasps each, 474.0 -> 474.0 mm and 307.7 -> 307.7 mm. Contact-
GraspNet was available and used (checked; not a silent fallback). It simply did
not hold on either can.

## WHY IS CONTACT-GRASPNET HERE AT ALL? The justification is RETRACTED

`grasp_target`'s docstring says:

> whether [PointWorld] could also discover grasps is unmeasured, and the prior
> is that it cannot here -- the margin deciding grasp success is ~7 mm and
> **the model's error floor is 10-15 mm**.

**That error floor is a retracted finding.** It was measured on `small-droid`
with `domains=droid` and the 16/31 layout; on `large-droid+behavior` the error
on moved points is **3.92 mm** with `cos 0.999`. So the stated reason for not
planning grasps through PointWorld rests on a number that no longer holds, and
7 mm against 3.92 mm is a very different proposition from 7 against 10-15.

Three things now point the same way:

1. **The prior has no evidence behind it any more.**
2. **A grasp is exactly what this model should score well.** Its action space
   IS robot point flow, and the one thing every measurement agrees it predicts
   confidently is "the grasped thing follows the hand" -- which is precisely
   the question "did this trajectory acquire the object".
3. **CGN is not earning its place.** It failed 4/4 on both `libero_10/0`
   objects. This is not a case of replacing something that works.

The counter-argument is real and should be stated: CGN placed the grasps that
solved `libero_goal/0` and `/1`, and a scripted grasp is one fewer thing to
debug. But it is inherited from ReKep's AnyGrasp design, not from anything
measured about PointWorld.

## TESTED: PointWorld CANNOT rank grasps — Contact-GraspNet stays, for a
## MEASURED reason now

`tests/test_pointworld_grasp.py` / `scripts/run_grasp_ranking.sh`. 12 candidate
grasps around the bowl (4 approaches x lateral offsets 0/25/50 mm), each scored
by whether the model predicts the object FOLLOWS a subsequent lift, and each
also EXECUTED in the simulator from the same saved state.

| candidate | model says | sim does |
|---|---|---|
| `[0,-1,0] off 25mm` | **115.2 mm** (top pick) | **0.0 mm** |
| `[0,-1,0] off 50mm` | 114.8 mm | 0.0 mm |
| `[0,0,-1] off 0mm` | 114.4 mm | 2.6 mm |
| `[-1,0,0] off 0mm` | 96.4 mm (7th of 12) | **106.1 mm** (best real) |
| `[-1,0,0] off 50mm` | 1.7 mm (worst) | 35.0 mm |

**Rank correlation -0.15.** The model's top pick closed on air; its worst pick
held. 5 of 12 candidates really held, and the ordering finds none of them.

### WHY, and it is the same finding as everywhere else

Look at the lateral sweep. For `-Y` and top-down approaches the prediction is
112.3 / 115.2 / 114.8 and 114.4 / 114.1 / 114.8 -- **identical across 0, 25 and
50 mm of offset**, while reality goes from held to closed-on-air. The model
varies by ~2 mm on the variable that decides a grasp, and reality varies by
100 mm.

That is follows-the-hand again: it predicts the object moves with the gripper
whether or not the jaw is actually around it. It has no representation of
enclosure. The same property explains the flat axis probe, the 60.6 mm of
spurious static motion, and why an IoU-0.00 mask still solved the drawer.

There IS coarse signal -- the `+Y` approach scores 27-45 mm and really does
fail (0.0 mm), so gross reachability registers. What is missing is exactly the
~7 mm margin that decides whether jaws enclose an object.

### So the docstring's conclusion was right and its reasoning was wrong

`grasp_target` justified CGN with "the model's error floor is 10-15 mm", which
is retracted. The real reason is not resolution at all -- the correct
checkpoint predicts moved points to 3.92 mm -- it is that the model does not
represent enclosure, so no amount of accuracy would help. Updated in place.

**This does not make grasping unplannable**, it makes it unplannable BY THIS
COST. A cost that asked "do the object's points move DIFFERENTLY from the
gripper's" might separate held from pushed. Untested, and a real option.

### THE NEXT TASK

0. **Arm-body clearance** via the existing SDF, if the visible bumping persists
   once bystander disturbance is down.
1. **Calibrate the grounded displacement** -- 66% over is the largest error
   left, and `measure_grounding.py` scores it without running the loop.
2. **Add an action type to the schema** so articulated tasks can be expressed.
3. **Replace the `cos` diagnostics in `--ground` mode** with the grounded goal,
   not the oracle's.

### THE OLD NEXT TASK — see "SURVEY" at the bottom of this file. Item 1 is done.
2. **Persist the target point set** across ticks and re-ground on events only,
   per the guide. Cheap now, and it is a prerequisite for a real grounder
   rather than a refactor after one.
3. **Re-cost the MPPI config** against the latency table above: K=8 x 1 chunk,
   not K=32 x 3.
4. **The mask**, still the biggest cheat (`task_spec.py` reads MuJoCo geoms).

**Design basis is `docs/pointworld_pipeline_implementation_prereqs.md`, not
KUDA.** The guide's layering is: DINOv3 coarse grounding -> object scene graph
(`object_id`, `label`, `centroid_xyz`, `bbox_extent_xyz`, `relations`) -> an LLM
**goal formalizer** emitting structured 3D goals, never low-level actions ->
MPPI over PointWorld, starting from `pytorch_mppi` -> event-driven re-grounding
after grasp, release, occlusion or repeated planning failure. KUDA's
keypoint-offset prompt is not the target; `keypoint_grounding.py` should be
re-read against the guide's scene-graph schema before more is built on it.

---

# SURVEY: what actually needs doing (2026-08-03)

Ordered by what blocks what. Each item says why it exists and how we know.

## The one-paragraph state

The world model works: 3.92 mm on moved points, direction `cos 0.999`, and it
ranks candidate actions correctly. The perception, the bridge, the feature
assembly and the FK are all measured and sound. **The only thing standing
between here and a solved task is the action optimiser**, and the current one
(MPPI) is measurably worse than the simpler one it replaced. Nothing about
language or grounding is on the critical path yet, because the goal is
currently handed over perfectly and the planner still cannot use it.

## Is an optimiser necessary at all, given grounding?

Yes, and the two are not substitutes. Grounding answers *which points and where
they should end up*. Something still has to decide *what the robot does* to get
them there, and PointWorld only answers "if you move like this, here is what
happens" -- it is a simulator, not a policy. Searching over candidate actions
and scoring them is the only way to invert it.

But "an optimiser" does not have to mean MPPI as currently written. The
structured search IS a sampling optimiser, and it beats the Gaussian one here
by 140 mm to -2 mm. The guide asks for an MPPI/MPC layer; it does not ask for
isotropic joint-space sampling.

## 1. BLOCKER — reimplement the action optimiser

Keep from the current file: joint-space FK (removes the gripper re-binding
problem), `TaskSpec` (task-agnostic), the batched single-call rollout, and the
`rho`/`cos best`/`cos far` diagnostics that found all of this.

Keep from the old file: **the structured, low-dimensional, straight-line
candidate set**, and an argmin over it.

Then fix the three things neither had:

1. **Rate resolution in the endgame.** Scale the candidate rates by the
   remaining distance so there is always a candidate that lands ON the goal
   rather than 80 mm past it. This alone should convert 121-140 mm of travel
   into a success; both structured runs stalled on it.
2. **Score the best point along the trajectory, not only its end.** A terminal
   cost at a fixed horizon cannot express "arrive and stop".
3. **Coarse-to-fine, not random.** If sampling is wanted, sample AROUND the
   structured winner with a shrinking radius. That is the paper's MPPI in
   spirit and keeps the search in the space the model was validated on.

Guard rails, all learned the expensive way today and cheap to keep:
`rho > 0.9` (the model, not the control cost, is ranking), `viol == 0`
(limits clamped, not penalised), the fingers out of the action space, and the
temperature solved for ESS = K/4 rather than chosen.

## 2. Then — re-validate the model claims on the real loop

Only after 1, because a planner that does not converge cannot measure a model.

* Re-run the drawer end to end on `large-droid+behavior` and report
  `_check_success()`, not millimetres of travel.
* **Find a task where the model's physics is load-bearing.** The axis result
  says PointWorld predicts "grasped things follow the hand" in all 18
  directions. A drawer cannot distinguish that from understanding. Pushing an
  unrestrained object can, and the deploy target's real application
  (deformables) needs exactly that.

## 3. Then — the language layer, to the guide's design

`docs/pointworld_pipeline_implementation_prereqs.md`, NOT KUDA. The order
matters: build it only once the planner converges on oracle goals, or a
grounding failure and a planner failure are indistinguishable.

1. **Coarse grounding**: DINOv3 patch features -> cluster -> back-project with
   depth -> open-vocabulary labels -> an object-level scene graph
   (`object_id`, `label`, `centroid_xyz`, `bbox_extent_xyz`, `relations`).
   Much of this is free: `scene_featurizer.py` already projects every scene
   point into both cameras with visibility and depth checks, and the encoder
   already computes 128-d DINOv3 features per point.
2. **LLM goal formalizer**: instruction + scene graph -> structured 3D goals,
   never low-level actions. Replaces `task_spec.py`, which is the single
   biggest cheat left (it reads MuJoCo collision geometry).
3. **Event-driven re-grounding.** Today grounding re-runs EVERY tick because
   the oracle is free and exact; a real grounder is neither. Target points must
   persist and be tracked, re-grounding only on grasp, release, occlusion or
   repeated planning failure. This needs a persistent target-point set, which
   does not exist yet and should be built before any grounder is wired in.
4. `keypoint_grounding.py` is built to KUDA's keypoint-offset schema. Re-read
   it against the scene-graph schema before extending it.

## 4. In parallel — deployment, which is not blocked by any of the above

* `scripts/setup_pointworld.sh` hardcodes the GB10 arch. Take it as an
  argument; Orin is sm_87, Thor is Blackwell-class. Both aarch64, so the
  expensive spconv/flash-attn knowledge transfers.
* Port `src/pointworld_bridge/*` unchanged; write robot equivalents of
  `pw_observation.py` (RealSense + TF self-filter) and `gripper_points.py`
  (URDF meshes).
* Budget from the measured table: batched, `large-droid+behavior` is 45.4
  ms/candidate and 8.59 GiB at K=32; small-droid is 10.5 ms and 3.27 GiB.
  Set K from the target's latency, not from a constant.
* **Gate**: "an episode running" must mean a PERCEIVED-mask episode. The oracle
  drawer cannot run on a robot at all.

## What is NOT worth doing

* Tuning MPPI's temperature, sigma or horizon further. All three were swept
  today; the search space is the problem, not its parameters.
* Chasing the model's spurious static motion (60.6 mm) before the planner
  converges. It has not yet cost us a task.
* Re-running the retracted `small-droid` measurements for their own sake. Only
  re-run what a decision depends on.
