# The optimization teacher

The general teacher is found by optimizing action sequences against the task loss, treated as
a field over the interaction space, and a policy learns the mapping from randomized scenes to
those sequences. It replaces the RL teacher (RL v7, BRN-rl-teacher-equilibrium-settle), which
is abandoned (2026-09-22). Status: design, before any measurement of the optimizer itself.

## 1. Why RL was abandoned

The PPO pilot on libero_goal 8 (`runs/rl/goal8_pilot`, commit 75422ff) ran 1.03M steps on the
ten performance cores:

| measured | value |
|---|---|
| successes | 0 in 1,663 logged episodes |
| reach term (tool to bowl) at episode end | 427-515 mm in every 50-iteration block, against 274 mm at the start |
| task loss at episode end | above its start in every block |
| critic value loss (normalized by return variance), median | 1.36 (iters 100-149) rising to 6.41 (350-399) |
| critic explained variance, median | 0.00 from iteration 200 |

The reach term is a dense, smooth signal (every centimetre the tool moved toward the bowl paid
one step of time cost), and the learner did not use it: the failure was the learner, not the
field. With
gamma = 1, truncation bootstrapped and no success ever reached, the current policy's value is
unbounded, and the critic chased it (a hypothesis consistent with the rising value loss at zero
explained variance, not separately tested). The deeper reason for leaving RL is structural:
RL learns a value function because it cannot rewind the world. This teacher is privileged: it
has the exact simulator and can clone and restore any state, so it can evaluate candidate
futures directly instead of estimating them.

## 2. Formulation

**State.** $s$ is the full execution state: MuJoCo's integration state plus everything the
execution layer carries between control periods -- the joint controller's cache and ramp, the
gripper's finger target (`current_action`), the twist servo's reference. With the lean control
period (`SimArm._advance`, integration state bit-identical to `env.step`,
BRN-lean-step-keeps-controller-cache for the physics half) the transition is a deterministic
function $s_{t+1} = F(s_t, u_t)$ (AXM-reproducible-measurement).

**Action.** $u_t = (v_t, g_t)$: a normalized body twist $v_t \in [-1,1]^6$ and one of the
student's gripper snap levels $g_t \in \{0, 0.026, 0.08\}$ m, executed through the one
execution path (AXM-one-execution-path). The linear part is scaled by
$v_{\max} = 0.05$ m per 50 ms step (robosuite `osc_pose.json`, `geometry/interface.py`).

**Randomization.** $\xi \sim P$: the LIBERO init state, the layout perturbation and the start
noise; $s_0 = R(\xi)$.

**Target set.** $S^\star$ = settled success: LIBERO's predicate holds, no robot geom touches the
moved object, each moved free object is in static equilibrium inside its contacts' friction
cones, its kinetic energy is below the tipping barrier and the slide bound, and each goal
joint's predicted rest satisfies its predicate (`teacher/settle.py`).

**Violation set.** $V$, checked at every 2 ms substep: release not gentle (DEF-gentle-placement),
a movable body the goal does not name changes its support body or resting face, the moved object
below the arena surface.

**The field.** $\Phi(s) = L(s) + \rho(s)$: the task loss $L$ (BRN-task-loss-core -- zero exactly
where LIBERO scores the goal satisfied, with $d/2 \le \max(a, b) \le D^\star$ against the true
distance) and the reach term $\rho$, the tool's distance to what must move (zero once held or
satisfied).

**Per-instance problem.** For a start $s_0$:

$$\min_{T,\,u_{0:T-1}} T \quad \text{s.t.}\quad s_{t+1}=F(s_t,u_t),\;\; s_t \notin V\;\forall t,\;\; s_T \in S^\star,\;\; T \le H = 600 .$$

It is solved by receding-horizon search: at each step, plans $U$ over a horizon of $N$ steps
are ranked lexicographically by the key

$$\big(\;\mathbb{1}[\text{the rollout records a violation}],\;\; N_{\text{reach}}(U),\;\; \rho_g(s_N) + d(s_N)/2\;\big),$$

where, for a plan that violates, the middle entry is the negated period of its violation (a
later violation ranks higher), and otherwise $N_{\text{reach}}$, the period at which $S^\star$ is
entered ($N+1$ if not); a rollout ends at its first decided violation or first settled period, so
nothing after settling is judged. The last entry orders unfinished plans (section 3.4, 0 once
LIBERO accepts). No weight trades one entry against another; plans with equal keys share the mean
of their ranks' weights. The verdicts are `teacher/verdicts.py` (BRN-teacher-verdicts): the
release watch is carried state -- one continuous watch judges the executed trajectory, every
rollout runs a fork of it, and a contact loss still open at a rollout's end is undecided and enters
no entry; the disturbance check is against both the episode start's and the search start's
references, so undoing a disturbance is not one. The first
action of the best plan is executed and the search repeats from the new state. The search is
closed-loop, so it acts on any randomized instance.

**The abstraction across randomization.** A policy $\pi_\theta(s)$ holds what the searches
find, so that nearby scenes get nearby actions and any state can be labelled cheaply
(AXM-dagger-needs-markov-labels):

$$\min_\theta\; \mathbb{E}_{\xi\sim P}\big[\,T(\pi_\theta; R(\xi))\,\big] .$$

**One network per task.** $\pi_\theta$ is trained separately for each task, on the full
privileged state (every free object's pose, box and velocities, fixture joints, the execution
state, the loss's gap vector, and each movable body's resting body and face at the episode
start -- `RLTaskEnv.observe`, 171 dimensions on libero_goal 8). The start's supports are part of
the state because the disturbance violation is judged against them; without them the label would
depend on the episode's history (AXM-dagger-needs-markov-labels). It is the
student's architecture without its two hardest inputs: images are replaced by the state they
depict, and the language instruction is gone because the task is fixed per network, so there is
no task-spec ambiguity to resolve. The abstraction each network must learn is only over the
randomization of its own task. The VLA student later distills across the per-task networks,
from images and language.

## 3. Optimization theory in the design

### 3.1 The field is non-smooth: optimize its smoothed version
Contact makes the cost of a plan discontinuous in the plan (a finger touches or it does not).
Gradients through contact are biased and high-variance (Suh et al., 2022, "Do differentiable
simulators give better policy gradients?"). The Gaussian-smoothed objective
$J_\sigma(U) = \mathbb{E}_\epsilon[J(U + \sigma\epsilon)]$ is differentiable even when $J$ is not,
with $\nabla J_\sigma(U) = \mathbb{E}[J(U+\sigma\epsilon)\,\epsilon]/\sigma$ (Nesterov & Spokoiny,
2017). Sampling-based MPC (MPPI, CEM, predictive sampling) estimates exactly this: it optimizes
the field blurred by its own noise, and the blur is what lets it see across a contact event.

### 3.2 Local minima: continuation, not descent
*Modern Robotics* Ch. 10 §4.4 shows gradient descent on a bowl-plus-bump field stalling short of
the goal, and names the remedies: random walks and fields built to have a single minimum. Our
field has such traps (the bowl pushed against the plate's rim: close in xy, not on top). The
design uses continuation (graduated optimization): start with a large $\sigma$, whose smoothed
field has few minima, and shrink $\sigma$ as the search converges. $\sigma_0$ comes from the scene
(the displacement it produces over the horizon matches the object-to-goal distance,
AXM-object-geometry-known); it shrinks when the best cost stops improving, so there is no
schedule to tune.

### 3.3 The search around the policy
With a scalar cost $J$, the optimal sampling distribution under a KL penalty to a prior $p$ is
$q^\star(U) \propto p(U)\,\exp(-J(U)/\lambda)$ (the path-integral / MPPI view; MPC as online
mirror descent, Wagener et al., 2019). Our plans are ranked lexicographically (3.5), which gives
an order, not a scalar, so there is no $J$ for a temperature to act on. The update instead weights
plans by rank only (as CMA-ES does) and bounds the KL divergence of the updated distribution from
the start distribution, which is centred on $\pi_\theta$'s plan. That keeps the search-as-E-step,
fit-as-M-step structure: the search improves on the policy's plan, fitting $\pi_\theta$ to the
searched plans absorbs the improvement, and the KL bound to the $\pi_\theta$-centred start is the
proximal term the consistency argument (3.10) needs. Every control step's search starts afresh
from $\pi_\theta(s)$ with a spread set from $s$ alone, and a categorical over snap levels equal to
$\pi_\theta$'s level probabilities floored so every level keeps positive probability (otherwise
the KL to the start is infinite and the level can never change). Its generator is seeded with one
fixed constant, recorded with every demonstration (a seed built from the execution state would
depend on its step counter), and the step counter is not an input, so, with $\pi_\theta$ frozen
before labelling, the label is a function of the teacher's state (AXM-dagger-needs-markov-labels).
That state includes the carried release watch -- which objects the robot touches, and each open
contact loss's age, gap and vertical speed -- because the release verdict depends on it.

### 3.4 Ordering unfinished plans by the loss
Plans that neither violate nor settle within the horizon are ordered by
$\rho_g(s_N) + d(s_N)/2$ at the horizon, taken as 0 once LIBERO accepts (otherwise an accepted but
unsettled plan would be ordered by $\rho_g$ alone, pulling the tool toward the object that
settling needs it to leave): $\rho_g$ is the tool point's distance to the nearest
point of the object's contact boxes (the point-box kernel in `geometry/box_distance.py`; the
loss's current `reach` measures to the box centre and is not this), $d$ the loss's distance
without its constant $M$. Only the order is used, so no speed scale enters -- which matters,
because the commanded speed bound is per axis (`sim_arm.py`), and the default path moved the
tool 0.0974 m in one step against a nominal 0.05 m.

The form is justified by LMA-loss-lower-bounds-steps: under its conditions -- the object's
orientation unchanged at acceptance, a static target, the object moving only while the tool
point is within it and no faster -- both terms are tool travel, and
$(\rho_g + d/2)/v_{\max}$ lower-bounds the steps to acceptance (admissible in A*'s sense,
*Modern Robotics* Ch. 10 §4.3; on the lean path saturated commands moved the tool point at most
0.0425 m per period). Those conditions held for 10.9% of the states of 245 successful
skill-teacher episodes and for none on libero_spatial or libero_goal: releases and drops,
squeezes that push the object faster than the tool point, and pivoting rim pinches fall outside.
So outside that scope the order is a heuristic, not a bound, and it is weakest exactly after a
release, where the object falls toward its goal without the tool. Receding-horizon performance
bounds (Grüne & Rantzer, 2008) depend on how closely the terminal term approximates the true
cost-to-go; Q4 measures that against the steps the search takes.

### 3.5 Constraints by ranking, not by penalty
A penalty weight on violations is exact only above the constraint's Lagrange multiplier, which
is unknown here. Sampling allows exact handling instead: every infeasible plan ranks below every
feasible one, and infeasible plans are ranked by how early they violate. No weight enters.

### 3.6 Grasping has a derived slope
Between touching and holding, $\rho$ is already near zero and $L$ has not moved, so the field is
flat exactly where the hard decision is. *Modern Robotics* Ch. 12 §4.9 gives a continuous,
object-generic measure: the grasp quality $\epsilon(s)$, the radius of the largest origin-centred
ball of wrenches the contacts can resist inside their friction cones ($\epsilon > 0$ is force
closure). The wrench to resist is the object's weight and its inertial load, from the MuJoCo
model's mass and the servo's acceleration bound. A candidate field term
$\max(0, \epsilon_{\text{req}} - \epsilon(s))$ gives the grasp a slope with nothing hand-set. It
is a candidate only: it enters the field if the stall map (3.7) shows the search stalling there.
It reuses `settle.py`'s friction-cone wrench columns.

### 3.7 Where the field needs work is measured, not guessed
$\Phi$ is used like a Lyapunov certificate: a good plan decreases it. At every step the search
logs whether any sampled plan within the horizon decreases $\Phi$. Steps where none does are the
field's flat regions and traps; their locations (approach, grasp, lift, rim) are the stall map.
Any field refinement is justified by a stall-map entry, and nowhere else.

### 3.8 Smoothness and dimension
The action sequence is parameterized by segments (section 3.11: one body twist and snap level
each), instead of free per-step actions. That cuts the search dimension -- the variance of zeroth-order estimates grows with
dimension (Nesterov & Spokoiny, 2017) -- and a knot spacing near the servo's bandwidth leaves
little for its rate limiter (2 m/s^2) to smooth. The spacing is a compute parameter (section 6).
Smoothness is not left to it: a demonstration is kept only if, over the episode, its tool
acceleration read at every executed substep through `_advance`'s substep hook has a 95th
percentile of at most 2 m/s^2 with no servo re-anchor (DEF-smooth-motion), and its continuous
watch records no violation. (TST-teacher-motion
wraps `sim.step` and sees no substeps under the lean period, so it cannot be the instrument.)

### 3.9 The gripper is discrete
The gripper level is categorical, so the plan is mixed-integer. Sampling handles it directly: the
knots carry a level, sampled from a categorical distribution updated like the continuous part
(CEM on categorical variables).

### 3.10 Consistency across randomization
Fitting one $\pi_\theta$ to searches on many instances is a consensus problem: minimize
$\sum_i J_i(U_i)$ subject to $U_i = \pi_\theta(s_i)$. Solving it by alternating a per-instance
search (with the KL term to $\pi_\theta$ of 3.3 as its proximal term) and a regression of
$\pi_\theta$ is guided policy search (Levine & Koltun, 2013; with Bregman ADMM, Levine et al.,
2016). The proximal term is what keeps labels consistent: nearby scenes are searched around the
same policy, so they do not flip between strategies. Whether the searched solutions vary smoothly
with $\xi$ at all -- or switch between grasp modes, in which case $\pi_\theta$ needs a discrete mode
output -- is measured before $\pi_\theta$'s form is chosen (question Q3).

### 3.11 Screw mechanics in the search (BRN-screw-search)
The teacher already acts through screwhead's interface: a body twist that TwistServo integrates on
SE(3) and decodes through PoE kinematics and damped least squares. The search adopts the same
mechanics, where the field is hardest:

- **Screw segments.** Each plan segment is one normalized body twist and one snap level. Because
  the servo's pose reference integrates the twist exactly ($T_{\text{ref}} \leftarrow
  T_{\text{ref}}\exp([V]\,dt)$), a segment moves it along exactly one screw once the rate limiter
  has reached $V$. Rollouts simulate the servo's tracking exactly, so the parameterization only
  decides where plans are sampled.
- **Isotropic in se(3).** Spreads are isotropic in the normalized twist, i.e. in se(3) with the
  length scale $L = v_{\max}/\omega_{\max} = 0.1$ m that robosuite's `osc_pose` declares. No
  per-axis weight is chosen.
- **First-order where smooth, sampling across contact.** Before sampling, the mean of the leading
  segments moves down the gradient of the terminal order's smooth part: $\rho_g$ through the tool
  point's velocity under a body twist (the body Jacobian) for a free tool; $d$ through the held
  object's velocity as a rigid attachment of the tool frame (the adjoint map) for a held one. It
  never differentiates contact dynamics (section 3.1); the samples decide.
- **Grasp quality from contact wrenches.** Section 3.6's term, the largest wrench ball the contacts
  resist (*Modern Robotics* Ch. 12 §4.9), comes from the same wrench columns as `settle.py`. It
  enters the field only where the stall map shows plans stalling between touching and holding.
- **Embodiment-free plans.** Plans, $\pi_\theta$'s outputs and the verdicts contain no joint-space
  quantity. $\pi_\theta$'s input does (the execution state), so transferring a teacher to another
  arm needs that input split into tool-space state plus screwhead's robot spec tokens (Q8).

It must stay a coordinate choice, not rules: no approach direction, pre-grasp pose or grasp bonus
is written in. Whether (a)-(c) reduce the samples a search needs is Q7.

## 4. The algorithm

1. **Search probe (iteration 0).** No policy: the prior is a zero-mean plan. On $K$ randomized
   instances, receding-horizon search with knots, continuation on $\sigma$, lexicographic ranking,
   terminal cost $h$; rollouts from cloned states through the lean control period, parallel over
   the ten performance cores. Log success, steps, wall time, $h(s_0)$, the stall map.
2. **Search + learn.** Fit $\pi_\theta$ by regression on the searched (state, action) pairs; search
   again around $\pi_\theta$ under the KL budget; repeat. The searches get cheaper as $\pi_\theta$
   improves.
3. **Teacher.** $\pi_\theta$, optionally refined by a short search seeded from the state, labels
   DAgger states for the student. The teacher acts through
   `Execution(lean=True, anchor=True, scale_lead=True, gripper_mode="target")`, snap levels fixed
   at 0, 0.026 and 0.08 m, horizon 600, no execution noise. The student's collection, DAgger and
   evaluation must adopt the same execution -- today they differ in more than the three options
   (`PrivilegedEnv` defaults to gripper command mode, snap levels are optional in `distill.py`, its
   horizon defaults to 300, DART noise reaches the gripper channel unsnapped) -- which is a
   separate obligation following BRN-policies-read-one-forwarded-state.

Reused: `teacher/task_loss.py`, `geometry/box_distance.py`, `teacher/settle.py`, the violation
checks (to move out of `teacher/rl_env.py`), the execution options (lean period, anchor, uniform
servo lead). Retired: `tools/rl_train.py` and the RL reward in `rl_env.py`. Built and measured: `sim/exec_state.py` (BRN-execution-state-restore: the integration
state, the joint ramp, the controller cache with its memory layout, the servo's ref, T_ref and
rate-limiter memory, the finger target, the step counter; then `mj_forward`, a cleared snapshot
cache and a fresh observation; across instances also the body poses LIBERO re-samples -- 609/609
same-instance replays bit-identical on six configurations) and `teacher/verdicts.py`
(BRN-teacher-verdicts).

## 5. Questions measured before any claim

| | question | measurement |
|---|---|---|
| Q1 | Does search alone reach settled success? | success, steps and wall time on $K$ randomized goal-8 instances |
| Q2 | Where does it stall? | the stall map (3.7) by stage |
| Q3 | Do nearby instances get the same solution? | grasp mode and plan distance on pairs $(\xi, \xi+\delta)$ |
| Q4 | How does the terminal order compare to the true cost-to-go? | $(\rho_g + d/2)$ at each state against the steps the search then takes |
| Q5 | What does it cost? | milliseconds per control step on half the CPU |
| Q6 | Does the settle residual tolerance separate? | measured: no residual between 1e-12 and 1e-6 in 1653 samples (TRL-0230) |
| Q7 | Do screw segments, se(3)-isotropic spreads and the first-order proposal reduce the samples a search needs? | samples to settled success against knot plans with per-axis spreads and no proposal |
| Q8 | Does a teacher transfer to another arm? | the same verdicts and search through another arm's execution path; $\pi_\theta$ with a tool-space input |

Only measured numbers go into the branch claims (one mechanism per branch).

## 6. Parameters and their sources (AXM-parameters-derived)

The problem -- objective, constraints, target set -- and every verdict take their numbers from
declared sources:

| parameter | source |
|---|---|
| $H$ = 600 | LIBERO's evaluation horizon |
| acceptance tolerances | LIBERO's predicates (BRN-task-loss-core) |
| gentle release: 5 mm, 0.05 m/s | DEF-gentle-placement |
| settled: friction, masses, damping | the MuJoCo model |
| release window: 16 substeps | the free-fall time of the 5 mm gap (DEF-gentle-placement, gravity, timestep) |
| friction cone | an 8-edge pyramid with faces at 0.854 mu: the inscribed one (0.924 mu) accepted incline states MuJoCo then slid |
| settle residual counted as zero (1e-6 of the weight), coincident-point guard (1e-9 m) | **numerical, not yet measured** against the verdicts they could change (Q6) |
| $\epsilon_{\text{req}}$ (if 3.6 is needed) | object mass (MuJoCo model), gravity, the servo's acceleration bound |

The search's own settings are compute parameters: they appear in neither the problem nor any
verdict, they change which plan a finite search finds, and every demonstration records them --
horizon $N$, knot spacing and count, iterations per step, samples $M$, rank weights, the spread's
initial value (from the object-to-goal distance) and shrink rule, the categorical update, the KL
budget $\varepsilon$, instances $K$. The execution layer's constants (acceleration bounds, joint
step) belong to the execution path the student shares (BRN-execution-bounds-acceleration), not to
the teacher.
