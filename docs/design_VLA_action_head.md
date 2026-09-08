# A URDF-conditioned action head

Companion to [`design_VLA_goal.md`](design_VLA_goal.md). Two halves, as the goal
doc asks for: how SOTA VLAs are built and trained today, and what *Modern
Robotics* offers the action head that they are not using.

Written 2026-09-07. Claims about `third_party/` are cited to file:line and were
read, not recalled. Claims about the outside literature are from memory and are
marked; verify before building on any of them.

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

---

## Part III — The proposed head

```
  images + language
         │
    [ VLM backbone ]                    URDF ──► PoE parse (Ch.4 §4.5)
         │                                        │
         │                              (M, {B_1…B_n}, limits, types)
         │                                        │
         ▼                                        ▼
  [ action expert ]  ◄──── cross-attend ──── [ embodiment tokens ]
         │                                    (n tokens × 6 + type)
         ▼
   body twist V_b ∈ se(3)  +  grasp intent      ← LEARNED, embodiment-free
         │
         ▼
  [ DLS-IK layer: Δθ = (JᵀJ + λ²I)⁻¹ Jᵀ e ]     ← FIXED, differentiable,
         │            + null-space secondary       parameterized by URDF
         ▼
    joint commands θ̇ (n-dim, any n)
```

Four commitments:

1. **Embodiment encoder**: PoE tokens, one per joint, from the URDF. Variable
   length. Replaces GR00T's `embodiment_id` integer and π0's zero padding.
2. **Output space**: a body twist in the tool frame, plus a grasp intent. Fixed
   6+k dimensions regardless of `n`. This is the invariance the whole design buys.
3. **Decoder**: DLS-IK, differentiable, non-learned. The robot-specific knowledge
   lives in `J(θ)`, which is computed from the tokens, not learned.
4. **Secondary objectives** (joint limits, manipulability, local repulsion) in the
   null space, with the inertia-weighted inverse if dynamics are available.

Practical footing (from memory — check before committing): `pytorch_kinematics`
already does URDF → differentiable FK and Jacobians in torch; `curobo` (NVIDIA)
does GPU collision-aware IK and is the natural fit for the Orin/Thor target if
the null-space term needs real collision distances rather than sphere
approximations.

---

## Part IV — The proposed test has a confound

The goal doc's plan is: "train on libero and then we switch the urdf to another to
see if it automatically generalize." That test does not currently measure what it
is meant to measure.

**In both in-tree stacks, LIBERO's action space is already Cartesian.**

- π0: LIBERO actions are 7-dim and sliced back as 7
  (`openpi/src/openpi/policies/libero_policy.py:100`) — 6-DoF end-effector delta
  plus gripper, executed through robosuite's operational-space controller.
- GR00T: the `libero_panda` action modality keys are literally
  `x, y, z, roll, pitch, yaw, gripper`
  (`Isaac-GR00T/gr00t/configs/data/embodiment_configs.py:91`).

So the existing baseline **already** abstracts the embodiment away — via the
controller, which runs its own IK. Swap the URDF under that setup and a large part
of "it generalized" is attributable to the OSC controller, not to anything the
action head learned. The result would be real but would not be evidence for the
thesis.

To make the swap informative, one of these has to hold:

- **Train a joint-space baseline.** Have the LIBERO policy emit joint positions or
  velocities directly, so the embodiment is genuinely inside the learned mapping.
  Then the URDF-conditioned head has something to beat, and the padded-vector
  baseline should fail the swap.
- **Or swap to an embodiment the controller cannot absorb**: different DoF count,
  different topology, a redundant 7-DoF arm where null-space choice matters. A
  Panda→Panda-variant swap tests nothing; a Panda→7-DoF-redundant or
  Panda→different-wrist swap does.

Two more things that will silently break a URDF swap, both from p. 101's reality
check — failures that "produce a correct-looking wrong robot":

- `rpy` in URDF is **fixed-axis** roll-pitch-yaw, i.e. `Rot(ẑ,γ)·Rot(ŷ,β)·Rot(x̂,α)`.
  Reverse the order and `M` comes out tilted but plausible.
- A continuous joint's zero is wherever the file says, not where the encoder index
  sits. If they differ, every `θ_i` is offset and `M` is wrong.

Either one presents downstream as "the policy failed to generalize" rather than
"the conversion was wrong."

**Therefore, before any policy conclusion: a conversion test.** Assert PoE forward
kinematics against the simulator's own FK over randomly sampled configurations, per
URDF, to machine precision. It is cheap, it is a declared test with a scalar metric,
and it is the gate that stops a parsing bug from being read as a scientific result.

### What to measure, in order

| # | Question | Test |
|---|---|---|
| 1 | Does the URDF→PoE conversion reproduce the sim's kinematics? | max ‖FK_PoE(θ) − FK_sim(θ)‖ over random θ, per URDF |
| 2 | Does the DLS-IK layer converge on reachable targets? | success rate + iteration count vs. λ, incl. near-singular starts |
| 3 | Does a joint-space padded-vector baseline actually fail the swap? | LIBERO joint-space policy, source → target URDF |
| 4 | Does the URDF-conditioned head beat it? | same swap, same seeds, paired |
| 5 | Does it hold on an unseen DoF count/topology? | held-out embodiment, never trained on |

Question 3 is the one that decides whether the project has a premise. If the padded
baseline transfers fine, there is nothing to fix.
