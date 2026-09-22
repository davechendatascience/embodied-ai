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

It is solved by receding-horizon search: at each step, over a horizon of $N$ steps,

$$U^\star = \arg\min_{U}\; \Big[\,N_{\text{reach}}(U) \;+\; h(s_N)\Big]\quad\text{ranked lexicographically after feasibility,}$$

where $N_{\text{reach}}$ is the step at which $S^\star$ is entered ($N$ if not) and $h$ is the
terminal cost-to-go (section 3.4); the first action is executed and the search repeats from the
new state. The search is closed-loop, so it acts correctly on any randomized instance.

**The abstraction across randomization.** A policy $\pi_\theta(s)$ holds what the searches
find, so that nearby scenes get nearby actions and any state can be labelled cheaply
(AXM-dagger-needs-markov-labels):

$$\min_\theta\; \mathbb{E}_{\xi\sim P}\big[\,T(\pi_\theta; R(\xi))\,\big] .$$

**One network per task.** $\pi_\theta$ is trained separately for each task, on the full
privileged state (every free object's pose, box and velocities, fixture joints, the execution
state, the loss's gap vector -- `RLTaskEnv.observe`, 171 dimensions on libero_goal 8). It is the
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

### 3.3 The search as inference, the policy as its prior
The optimal sampling distribution under a KL penalty to a prior $p$ is
$q^\star(U) \propto p(U)\,\exp(-J(U)/\lambda)$ (the path-integral / MPPI view; MPC as online
mirror descent, Wagener et al., 2019). Setting $p = \pi_\theta$ makes one design out of two
pieces: the search is the E-step (improve on the policy's plan), fitting $\pi_\theta$ to the
searched plans is the M-step. $\lambda$ is not tuned: it is the dual variable of a KL budget
$\mathrm{KL}(q\,\|\,p) \le \varepsilon$ (as in relative-entropy policy search), and
$\varepsilon$ is a free parameter that changes how far one iteration moves, not where the
iterations converge.

### 3.4 A cost-to-go derived from the loss
The tool moves at most $v_{\max}$ per step, and the object moves only with the tool while it is
quasi-static. So from $s$, reaching $S^\star$ needs at least $\rho_{\min}(s)$ of tool travel to
touch the object and $D^\star \ge d(s)/2$ of travel with it (LMA-loss-lower-bounds-steps):

$$h(s) = \frac{\rho_{\min}(s) + d(s)/2}{v_{\max}} \quad\text{steps}$$

Here $\rho_{\min}$ is the tool point's distance to the *nearest point* of the object's contact
boxes (the point-box kernel in `geometry/box_distance.py`), not the loss's current `reach`,
which measures to the box centre and exceeds $\rho_{\min}$ by up to the box's half-diagonal --
with it, $h$ would not be a lower bound. $d$ is the loss's distance without its constant $M$.

is a lower bound on the steps to go (admissible, in A*'s sense, *Modern Robotics* Ch. 10 §4.3),
with no weight chosen by hand: $1/v_{\max}$ = 20 steps per metre. The bound is not claimed when the
object is reoriented, slides after a push, falls, or is pushed by another body, nor if the
executed tool overshoots the commanded 0.05 m per step; those are where it is tested.
Receding-horizon performance bounds (Grüne & Rantzer, 2008) depend on the horizon and on how
closely the terminal cost approximates the true cost-to-go; measuring $h(s_0)$ against the steps
the search actually takes gives that gap.

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

### 3.8 Smoothness and dimension from the servo
The action sequence is parameterized by knots, linearly interpolated, instead of free per-step
actions. Knot spacing is the servo's bandwidth: DEF-smooth-motion bounds tool acceleration at
2 m/s^2, so a twist change of $\Delta v$ takes at least $\Delta v/(2\,\text{m/s}^2)$, and knots
closer than that cannot be tracked. This gives smooth demonstrations without a smoothness weight,
and it cuts the search dimension -- the variance of zeroth-order estimates grows with dimension
(Nesterov & Spokoiny, 2017).

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

## 4. The algorithm

1. **Search probe (iteration 0).** No policy: the prior is a zero-mean plan. On $K$ randomized
   instances, receding-horizon search with knots, continuation on $\sigma$, lexicographic ranking,
   terminal cost $h$; rollouts from cloned states through the lean control period, parallel over
   the ten performance cores. Log success, steps, wall time, $h(s_0)$, the stall map.
2. **Search + learn.** Fit $\pi_\theta$ by regression on the searched (state, action) pairs; search
   again around $\pi_\theta$ under the KL budget; repeat. The searches get cheaper as $\pi_\theta$
   improves.
3. **Teacher.** $\pi_\theta$, optionally refined by a short search, labels DAgger states for the
   student through the one execution path.

Reused: `teacher/task_loss.py`, `geometry/box_distance.py`, `teacher/settle.py`, the violation
checks (to move out of `teacher/rl_env.py`), the execution options (lean period, anchor, uniform
servo lead). Retired: `tools/rl_train.py` and the RL reward in `rl_env.py`. New: saving and
restoring the whole execution state (the field list is the one the lean-step review verified).

## 5. Questions measured before any claim

| | question | measurement |
|---|---|---|
| Q1 | Does search alone reach settled success? | success, steps and wall time on $K$ randomized goal-8 instances |
| Q2 | Where does it stall? | the stall map (3.7) by stage |
| Q3 | Do nearby instances get the same solution? | grasp mode and plan distance on pairs $(\xi, \xi+\delta)$ |
| Q4 | Is $h$ a lower bound, and how loose? | $h(s_0)$ against steps taken; cases where it fails |
| Q5 | What does it cost? | milliseconds per control step on half the CPU |

Only measured numbers go into the branch claims (one mechanism per branch).

## 6. Parameters and their sources (AXM-parameters-derived)

| parameter | source |
|---|---|
| $v_{\max}$ = 0.05 m/step, so $1/v_{\max}$ = 20 steps/m | robosuite `osc_pose.json` via `geometry/interface.py` |
| $H$ = 600 | LIBERO's evaluation horizon |
| knot spacing | DEF-smooth-motion's 2 m/s^2 and the twist range |
| $\sigma_0$ | object-to-goal distance over the horizon (AXM-object-geometry-known) |
| $\sigma$ schedule | shrink on stalled improvement (no schedule) |
| $\lambda$ | dual of the KL budget $\varepsilon$ |
| $\varepsilon$, samples per step $M$, instances $K$ | free: they change convergence speed and cost, not the optimum |
| horizon $N$ | at least the gripper's measured closing time plus the lift to clear the support; to be measured |
| $\epsilon_{\text{req}}$ (if 3.6 is needed) | object mass (MuJoCo model), gravity, servo acceleration bound |
