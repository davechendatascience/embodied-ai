# A URDF-conditioned action head

Companion to [`design_VLA_goal.md`](design_VLA_goal.md). Two halves, as the goal
doc asks for: how SOTA VLAs are built and trained today, and what *Modern
Robotics* offers the action head that they are not using.

Written 2026-09-07; Parts III–V revised 2026-09-16 to the design as built.
Claims about `third_party/` are cited to file:line and were read, not recalled.
Claims about the outside literature are from memory and are marked; verify before
building on any of them.

---

## Part I — How SOTA VLAs are built

### The shared skeleton

Almost every 2024–2026 manipulation VLA is the same three pieces:

1. **A pretrained VLM backbone** takes images + a language instruction. Frozen
   or lightly tuned. This is where all the semantic generalization comes from.
2. **An action expert** — a smaller transformer or DiT, cross-attending to the
   backbone — that produces a *chunk* of future actions rather than one step.
3. **A continuous-action decoder head**: flow matching or diffusion, replacing
   the discretized-token decoders of the RT-2 generation.

Action chunking is near-universal: predict H steps (H ≈ 16–50), execute some
prefix, re-plan. It buys temporal consistency and amortizes the backbone's
latency over many control steps — the only reason a 2B-parameter VLM can sit in
a 50 Hz loop at all.

### π0 / π0.5 — `third_party/openpi`

- PaliGemma-class VLM + a separate small "action expert" stream, joint attention
  between them (`src/openpi/models/pi0.py:66`).
- Flow matching: the expert regresses a velocity field `v_t` over a noised
  action chunk, integrated at inference (`pi0.py:212`, `pi0.py:269`).
- **Fixed 32-dim action vector** (`pi0_config.py:25`), horizon set by config.
- A single `action_in_proj` / `action_out_proj` linear pair maps that 32-vector
  to and from the expert width (`pi0.py:92`, `pi0.py:100`).
- π0-FAST is the autoregressive variant: actions are compressed (DCT + BPE-style
  coding, `models/utils/fsq_tokenizer.py`) into discrete tokens so the VLM can
  emit them directly. Slower per step, easier to train.

### GR00T N1.6 — `third_party/Isaac-GR00T`

The checkout was removed on 2026-09-25 to save disk. The paths below are at
github.com/NVIDIA/Isaac-GR00T commit 5dc80c4; the robocasa checkout it was read beside was at
github.com/robocasa/robocasa commit 921c9a5.

- Eagle VLM backbone (`gr00t/model/modules/eagle_backbone.py`) + DiT flow-matching
  action head (`modules/dit.py`, `modules/flowmatching_modules.py`).
- `max_state_dim = 29`, `max_action_dim = 29`, `max_num_embodiments = 32`
  (`gr00t/configs/model/gr00t_n1d6.py:57-58,104`).
- The state encoder, action encoder and action decoder are **all
  `CategorySpecificMLP`, indexed by an integer `embodiment_id`**
  (`gr00t/model/gr00t_n1d6/gr00t_n1d6.py:47,56,58`). Each embodiment gets its own
  weight slab: `self.W = nn.Parameter(0.02 * torch.randn(num_categories,
  input_dim, hidden_dim))`, gathered per batch element
  (`modules/embodiment_conditioned_mlp.py`).
- Embodiments are a hand-maintained enum — `ROBOCASA_PANDA_OMRON`, `GR1`,
  `UNITREE_G1`, `LIBERO_PANDA`, `OXE_GOOGLE`, `OXE_WIDOWX`
  (`gr00t/data/embodiment_tags.py:14`).

### The rest of the field (from memory — verify)

- **RT-1 / RT-2**: actions discretized into 256 bins per dimension and emitted as
  text tokens. Established the "VLM as policy" framing; the discretization is
  what later work replaced.
- **OpenVLA**: 7B Llama-2 with DINOv2 + SigLIP vision, 256-bin discrete actions,
  trained on Open X-Embodiment. The open baseline everyone reports against.
- **Octo**: transformer with a diffusion action head, OXE-pretrained, designed for
  cheap fine-tuning to new observation/action specs.
- **RDT-1B, CogACT, SpatialVLA** and others: variations on backbone choice,
  diffusion head design, and 3D/spatial conditioning.

### How cross-embodiment is handled today — and why it is the gap

This is the part the goal doc is right to attack. Every system above resolves
"different robots have different action spaces" the same two ways:

**Pad to a maximum dimension.** π0 pads a 7-dim LIBERO action into a 32-vector
and slices `[..., :7]` back out at inference
(`src/openpi/policies/libero_policy.py:100`). GR00T pads to 29. The unused slots
are zeros. Nothing in the model knows which slots mean what.

**Give each robot its own adapter.** GR00T's per-embodiment weight slabs, above.
Open X-Embodiment / RT-X, CrossFormer, and HPT (heterogeneous pre-training with
per-embodiment "stems") are the same idea at dataset scale — from memory, but
the pattern is consistent.

Both are *learned* embodiment encodings. Consequences that matter here:

- A new robot needs a **new slot and new data**. There is no zero-shot path. The
  helper `CategorySpecificLinear.expand_action_dimension` handles a *bigger*
  action dim by literally tiling the old weight tensor
  (`embodiment_conditioned_mlp.py`) — a shape fix, not a kinematics fix.
- Slot count is capped (32 in GR00T) and the enum is hand-edited.
- **No model reads the robot's kinematics.** The URDF exists, is parsed by the
  simulator, and never reaches the policy. The one artifact that fully specifies
  the mapping from joint motion to tool motion is discarded, and the policy is
  asked to re-learn it from demonstrations, separately, per robot.

The closest prior art in the direction the goal doc proposes is **MetaMorph**
(Gupta et al., ICLR 2022 — from memory): a transformer over a kinematic-tree
token sequence derived from the morphology description, trained across 100+
locomotion morphologies, generalizing to unseen ones. It is locomotion, not
manipulation, and not language-conditioned. The manipulation + VLM version
appears not to exist yet. That is the opening.

---

## Part II — What *Modern Robotics* gives the head

All citations to `docs/Modern_Robotics_Complete.pdf`, via the `textbook-index`
MCP server.

### The URDF already contains the screw axes (Ch. 4 §4.5, p. 100)

The conversion is a five-step procedure at zero configuration, walking the tree
from the base:

1. Chain the joint `origin` transforms → pose `T_i` of each joint frame in base
   coordinates.
2. Axis in base coordinates: `ω_i = R_i · axis`, where `axis` is the joint's
   `<axis xyz=...>` entry.
3. A point on the axis: `q_i` = that joint frame's origin.
4. Revolute: `v_i = −ω_i × q_i`. Prismatic: `ω_i = 0`, `v_i` = slide direction.
5. Continue through the last joint and the tool offset to get `M`.

> "That is the entire conversion. The URDF's per-link frames are used once, to
> locate axes at the home posture." (p. 100)

The embodiment therefore reduces to **`(M, {S_1 … S_n})`** — one home pose plus a
variable-length list of 6-vectors, one per joint. A token sequence. Length-agnostic
by construction, which is exactly what a fixed 29- or 32-dim padded vector is not.

Forward kinematics is then `T(θ) = e^{[S_1]θ_1} ··· e^{[S_n]θ_n} M` (p. 112).

**Do not use D-H.** App. C (p. 378) is explicit: the common normal is ambiguous
when axes intersect, arbitrary when they are parallel, and "a milliradian
machining error makes the common normal jump by a kilometre." PoE stores 6
numbers per joint of which only 4 are independent (p. 101) — no more information
than D-H, but continuous in the physical parameters. Continuity is what makes it
safe to differentiate through.

### The Jacobian is closed-form in those tokens (Ch. 5 §4.2–4.3, pp. 115, 121)

> "The manipulator Jacobian is not a matrix of tedious partial derivatives…  It
> is the collection of the physical joint screw axes, moved to where they are at
> the current posture: column *i* is joint *i*'s screw axis, re-expressed by the
> Adjoint of everything upstream of it." (p. 115)

So `J(θ)` is a differentiable function of the URDF tokens and nothing else. This
is the actual lever, and it is stronger than the goal doc's framing: it means the
policy can emit an **end-effector twist in SE(3) — embodiment-free — and a fixed,
non-learned, differentiable layer parameterized by the URDF decodes it to joint
rates.** The learned part never sees a joint count.

Use the **body form** (`T = M e^{[B_1]θ_1} ··· e^{[B_n]θ_n}`, p. 112): axes are
expressed in the tool frame, so an embodiment swap changes only `M` and `{B_i}`
and leaves the policy's output frame untouched.

Statics comes free by transpose: `τ = Jᵀ F` (p. 115). One matrix, both directions.

### IK as a differentiable layer (Ch. 6)

Newton on SE(3): pose error by matrix logarithm, step by body Jacobian, iterate
(p. 140). The embeddable version is damped least squares (§4.4, p. 151), derived
from the cost

```
minimize  ‖J Δθ − e‖²  +  λ² ‖Δθ‖²      →      Δθ = (JᵀJ + λ²I)⁻¹ Jᵀ e
```

Two properties make this the right layer:

- It is Levenberg–Marquardt — a linear solve, differentiable, no branching.
- The step is **bounded**: `‖Δθ‖ ≤ ‖e‖/(2λ)` (p. 161). A bounded step is what
  makes unrolling IK inside a network safe near singularities, where the raw
  pseudo-inverse blows up like `1/σ_min` (p. 139).

Cost at inference is a 6×6 solve per iteration — trivial next to the backbone,
which matters for the Orin/Thor target.

### Redundancy is where "baked-in robotics" actually lives (pp. 290, 292, 346)

For a redundant arm the secondary term lies in `null(J)` and does not disturb the
tool (p. 346). Joint-limit distance, singularity distance, and obstacle repulsion
all ride there.

One trap: the ordinary pseudo-inverse does **not** guarantee this. Khatib's
inertia-weighted ("dynamically consistent") inverse leaves the hand at machine
zero under a secondary torque, while `J⁺` moves it — "a tool wandering off the
weld seam every time the posture optimiser adjusts an elbow" (p. 292).

Manipulability gives the scalars to push on: Yoshikawa's `m = √det(JJᵀ)`, zero at
a singularity, and the condition number `σ_max/σ_min` as an alarm (p. 126). Also
worth knowing for contact tasks: the force ellipsoid has the same axes with
lengths `1/σ_i`, so "where the arm is fast it is weak" (p. 126).

### Obstacle avoidance — the goal doc needs a correction here

The doc lists obstacle avoidance as bakeable. Ch. 10 says what is and is not:

> "Potential fields are reactive controllers, not planners… a U-shaped obstacle,
> or any obstacle in line with the goal, stops the robot dead." (p. 260)

And grid planning is out for arms: 10 cells/axis at 6 DoF is 10⁶ cells "before a
single collision check" (p. 259).

So a **local** repulsive term can go in the null space. **Global planning cannot
be baked into an action head** — it needs a graph or a sampler (PRM/RRT/RRT*),
and Ch. 10 §4.5 reduces all of them to one `is_free(q)` oracle plus edge
checking, noting that "testing collision only at waypoints" is the classic error
(p. 254). If global avoidance is required, it belongs beside the head, not inside
it.

### Grasping factors the same way (Ch. 12)

Grasp map `F_b = G f_c`, with the velocity dual `v_c = Gᵀ V_b` (p. 307) — the
same transpose duality as manipulator statics. Form closure needs ≥ 4 contacts
planar / 7 spatial with the origin strictly inside the convex hull of contact
wrenches; force closure is form closure applied to the friction-cone edge
wrenches, and two rubber fingertips suffice (p. 306).

These are geometric predicates on contact normals and friction cones. They do not
know what arm is behind the wrist. So the gripper action factors exactly as the
arm action does: emit a closure-satisfying contact configuration, let the
embodiment decode it to finger joints. Directly relevant to the `gripper-transfer`
branch.

*As built (Part III):* the policy currently emits the simplest grasp intent — a
target jaw aperture in metres — and a fixed servo realizes it. The grasp-map and
force-closure predicates are implemented and verified (`screwhead/analysis/grasp.py`), but
not yet in the policy's output.

---

## Part III — The head as built

```
  agent-view + wrist images (128 px)        instruction
         │                                       │
  [ frozen DINOv2-base ]                  [ frozen CLIP text ]
   8×8 patch tokens / camera                512-d
         │                                       │
         └──────────────┬────────────────────────┘
                        ▼
  [ TokenHead: 4 learned queries + 4-layer transformer ] ◄── tool pose (10-d)
                        │                              ◄── spec tokens, one per joint
                        ▼                                   (PoE body axes, from the URDF)
   body twist V_b (6)  +  gripper target aperture (1)   ← LEARNED, embodiment-free, one step
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
  [ TwistServo ]                [ GripperServo ]                ← FIXED, per robot
   SE(3)-integrated pose ref.    measured aperture + rate,
   → DLS-IK (+ null space)       lag-compensated close/hold/open
   → absolute joint target
          │                            │
          ▼                            ▼
     joint position controller    gripper action
```

Commitments, and what changed from the 09-07 proposal:

1. **Embodiment encoder** — as proposed: one token per joint (`screwhead/student/spec.py`:
   body-form screw axis with its linear part divided by the arm's reach, joint type,
   limits, index), masked rather than zero-padded. (A mask-polarity bug made the first
   token heads ignore these tokens; fixed, and older checkpoints load in legacy mode.)
2. **Output space** — a body twist plus a gripper **target aperture** in metres,
   `g = 1 − 2a / 0.08 m`. One control step, not a chunk: the teacher labels every
   visited state and closed-loop visual feedback is what the policy needs.
3. **Decoder** — DLS-IK as proposed, but at *execution*, not inside the training graph.
   Measured: LIBERO's joint-position controller achieves 82% of each commanded step, so
   per-step deltas compound (a demonstration's own labels replayed 0/50). `TwistServo`
   (`screwhead/sim/servo.py`) integrates the twist on SE(3) into a pose reference, solves IK
   from a joint reference rather than the lagging measurement, and commands an absolute
   target (replay 48/50).
4. **Gripper decoder** — `GripperServo` (`screwhead/sim/gripper_servo.py`). A lag-compensated
   close/hold/open law is a controller, not an intent; learned as a command it matched
   the teacher on 20–25% of pre-shape frames. The policy names the aperture, the servo
   reaches it, and the student's output is snapped to the apertures the programs use
   (closed, 26 mm pre-shape, open), with the snap levels stored in the checkpoint.
5. **Perception** — frozen DINOv2 patch tokens, chosen by a probe of grasp localization
   in the states the VLA itself drives into: CLIP pooled 16.8 mm → DINOv2 10.6–13.0 mm
   median (`tools/feature_bakeoff.py`).
6. **Secondary objectives** — null-space posture in `TwistServo` (used by the stove
   task's program); collision-aware terms remain future work.

## Part IV — Making the policy look before testing transfer

The 09-07 plan flagged one confound (LIBERO's Cartesian controller absorbs the arm).
Measurement found a deeper one: **LIBERO-spatial does not require vision.** A
demonstration-trained head with its images zeroed matched the sighted head; the
bundled init states vary the target by ~12 mm, inside the basin of one mean
trajectory. A transfer test on such a policy would test a trajectory prior. So the
thesis is now gated on grounding:

| Piece | What it does | Where |
|---|---|---|
| Randomization | object layouts (8 cm) that keep each instruction's spatial relation true; robot start pose ±10 cm / ±5 cm / ±30° / ±10° / null space | `screwhead/scripted/layouts.py`, `teacher_env.py` |
| Teacher | per-task demonstration programs on privileged state; Markov feedback laws, so they label any visited state | `screwhead/scripted/scripted_teacher.py` |
| Distillation | success-filtered teacher data, then DAgger (student drives with β = 0.5, teacher labels) | `tools/distill.py`, `tools/token_data.py` |
| One execution path | demonstrations, DAgger and evaluation execute through the same servos and the same stored decode | `scripts/token_vla.sh` |
| Controls | an image-zeroed copy trained on the same data must fail; per-episode trials go to the ledger | `belief.yaml` gate 2 |

Results so far (seed 555, randomized, VLA driving alone):

| Policy | Success | Blind copy |
|---|---|---|
| Scripted teacher | 96.5% | — |
| CLIP-feature VLA after DAgger | 33% | 1% |
| DINOv2 token VLA, 2 DAgger rounds (200 episodes) | **82%** (90.6% outside the drawer task) | **4.5%** |

## Part V — What to measure, in order

`belief.yaml` holds the declared form of this table; it is the source of truth.

| Gate | Question | Status |
|---|---|---|
| 1 | Do URDF→PoE, the Jacobian, DLS-IK and the null space reproduce the simulator exactly? | supported on panda, ur5e, iiwa, kinova3, jaco |
| 1 | Is the gripper geometry right? | supported for Panda and Rethink; Robotiq85 leaves its declared limits (excluded from transfer cells) |
| 2 | Does the randomization keep instructions true? | supported (200/200) |
| 2 | Does a blind policy fail and the sighted one succeed? | supported (4.5% / 82%) |
| 2 | Does the teacher solve every task under randomization? | pending (per-task test) |
| 3 | Does a joint-space padded-vector baseline, trained the same way, fail an arm swap? | blocked: needs arm swapping in `PrivilegedEnv` |
| 3 | Does the twist head keep its rate on an unseen arm, and do spec tokens matter? | blocked, same |

The teacher programs emit twists, so they should drive any arm through that arm's own
`TwistServo`; the gripper servo's aperture range should come from each gripper's finger
kinematics (`screwhead/analysis/gripper.py`) rather than the Panda's 0.08 m. Those two are the
remaining engineering between gate 2 and gate 3.

The two silent URDF failures noted on 09-07 still apply to any new arm: `rpy` is
fixed-axis `Rot(ẑ,γ)·Rot(ŷ,β)·Rot(x̂,α)`, and a joint's zero is wherever the file says.
The conversion test (gate 1) runs on every arm before any policy result is read.

## Part VI — Scaling past these ten tasks

Gate 2 is scored on ten libero_spatial tasks driven by ten hand-written demonstration
programs. Tuning further fits those ten. What is task-specific today, and what replaces it:

| Hand-written today | Replaced by |
|---|---|
| One `ProgramConfig` per task (rim sector, pre-shape, posture, approach heights) | A skill library parameterized by measured geometry, sequenced from the task's BDDL goal predicates (`CMP-skill-library`) |
| Gripper apertures 0 / 26 / 80 mm in the decode | Closed and open from the gripper's own finger FK (`screwhead/analysis/gripper.py`); an intermediate hold from the scene's measured clearance |
| `layouts.py` relations switched on task index | Relations read from the instruction's goal predicates, which name the same objects |
| 5 cm/s approach floor, 4 mm grasp tolerance | Kept as servo constants, but checked per gripper and object rather than assumed |

Two contracts, declared before either is built (`belief.yaml`, gate 2b):

- **CTR-held-out-task** — train on eight tasks, evaluate on the two never trained on, same
  suite and randomization. Bar 0.3 over at least 60 episodes: a blind-level 0.05 is refuted,
  a true 0.5 is supported. The trained-task rate is 0.83-0.88, so this is "it transfers
  something", not a performance claim. It fails loudly if the head has memorized ten tasks.
- **CTR-skill-teacher-solves-unseen** — a program generated from a task's goal specification
  solves that task at the bar the hand-written programs meet (0.75 per task, 50 episodes).

Order of work: hold out tasks first (cheap, and it tells us whether the current head
generalizes at all) -> skill library -> the other LIBERO suites (object, then goal and long,
which need new skills) -> arms, where `PrivilegedEnv` must take a robot and gripper and the
three transfer contracts stop being blocked.

Evaluation budget, which drives these choices: 200 episodes x 10 tasks is about an hour.
Forty tasks on three arms is a day per iteration, so iterate at 20 episodes per task and
spend 200 only on milestones. The ledger slices per task and per policy revision either way.
