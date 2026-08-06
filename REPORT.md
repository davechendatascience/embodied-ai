# Cross-Gripper Transfer of a Frozen VLA

**Status report. Written for an outside reader; all numbers are measured, and dead ends are
included deliberately because they constrain the solution space.**

---

## 1. The problem

We have a frozen vision-language-action policy trained exclusively on a Franka Panda arm with
a PandaGripper. We want it to drive a **different gripper** with no retraining, no fine-tuning,
and no demonstrations on the target.

**Swapping the arm already works.** Panda → UR5e, keeping the PandaGripper, transfers at 11/12
across four LIBERO suites. **Swapping the gripper does not.** UR5e + Robotiq85Gripper and
UR5e + RethinkGripper both failed 0/3 on every task at the start of this work.

The question is why the arm is free and the gripper is not, and what the minimum intervention
is that recovers the gripper.

### Setup

| | |
|---|---|
| Policy | VLA-JEPA: Qwen3-VL-2B + V-JEPA2 encoder + DiT flow-matching action head, frozen |
| Inputs | agentview RGB, wrist ("eye-in-hand") RGB, language, 8-D state (eef pos 3, axis-angle 3, gripper qpos 2) |
| Output | 7-D `[world_vector(3), rotation_delta(3), open_gripper(1)]`, Cartesian deltas into an OSC_POSE controller |
| Sim | robosuite 1.4.1 / MuJoCo 3.1.6, LIBERO benchmark |
| Arms | Franka Panda (7-DoF), Universal Robots UR5e (6-DoF) |
| Grippers | PandaGripper (source), Robotiq85Gripper, RethinkGripper (targets) |

---

## 2. Why the arm transfers and the gripper does not

The policy's only view of its own body is the wrist camera, and that camera mounts on the **arm's
wrist link**, not on the gripper. So:

```
camera → TCP,  Panda + PandaGripper     [0, -50,  -97] mm
camera → TCP,  UR5e  + PandaGripper     [0, -50,  -97] mm   ← bit-identical
camera → TCP,  UR5e  + Robotiq85        [0, -50, -145] mm   ← broken
```

The policy learned to drive the **camera**; the TCP followed because camera→TCP was a rigid
constant in training. Swapping the arm preserves that constant — the arm is *upstream* of the
invariant. Swapping the gripper destroys it.

The DoF count never enters, because the policy commands Cartesian deltas and the controller
resolves 6 or 7 joints. The 7th Panda joint only buys null-space redundancy, which the policy
neither controls nor observes.

---

## 3. Measurements that constrain any solution

### 3.1 Every non-visual channel into the policy is closed

| channel | measurement | verdict |
|---|---|---|
| language | a deliberately **wrong** instruction changes the action by cos 0.80 | too weak to steer |
| proprioception | swapping the entire state vector between arms changes output by **cos 1.000** | ignored entirely |
| temporal context | `horizon=0`; the image-history deque is empty | no in-episode adaptation |
| action distribution | `do_sample=True`, 24 draws: spread **±5 mm**, 0/24 reach the needed offset | 3–10× too narrow |

The action head is a diffusion sampler with sampling *disabled by default*; enabling it barely
widens the distribution beyond inference noise. The policy is also queried **once per 7-step
action chunk** and replays that chunk open-loop, so it is not a tight visual servo.

**Consequence: vision is the only channel with bandwidth, and the policy cannot be told anything.**

### 3.2 The gripper dominates the wrist image

Near-black fraction of the wrist view, in 8 horizontal bands top→bottom:

```
Panda      mean  3.4%   [1.7, 2.7, 3.4, 1.8, 3.6, 2.2, 12.0,  0.0]
Robotiq85  mean 25.8%   [0.5, 0.4, 0.5, 0.3, 2.3, 15.2, 89.2, 97.6]
```

The Robotiq blacks out the **bottom quarter** — exactly where the fingers and the
object-between-them appear when the policy decides to close. It makes its closing decision blind.
This is a projection effect of the 48 mm: the hand is nearer the lens.

### 3.3 Gripper geometry (free pad-face gap, not centroid distance)

```
                  closed → open      pinch offset from TCP    closing axis (EE frame)
PandaGripper       0.00 → 79.00 mm        [0, 0, -3.6] mm         [1, 0, 0]
Robotiq85         11.65 → 99.39 mm        [0, 0, -9.2] mm         [1, 0, 0]
RethinkGripper    13.43 → 60.10 mm        [0, 0, -6.6] mm         [1, 0, 0]
```

The closing axis is `[1,0,0]` in the end-effector frame for all three, with **zero spread across
arm poses** — a genuine pose-independent constant. The pinch offset's lateral component is
**exactly zero** for all three: for any two-jaw hand mirror-symmetric about a plane containing the
tool axis, the pinch midpoint lies on that axis at every closure. This is structural, not
coincidental.

---

## 4. The rule that survived: aperture bracketing

A two-sided grasp on a feature of local width `W` requires

```
w_closed  ≤  W  ≤  w_open
```

Nothing is fitted. It predicts every scripted outcome we measured:

| hand | feature | test | predicted | measured |
|---|---|---|---|---|
| Panda | bowl rim 2.56 mm | 0 ≤ 2.56 ≤ 79 | grasps | grasps (baseline) |
| Robotiq85 | bowl rim 2.56 mm | 11.65 > 2.56 ✗ | **zero contacts** | 0/0/0 contacts, 3 inits |
| Robotiq85 | bowl body 90.9 mm | 11.65 ≤ 90.9 ≤ 99.39 | envelope, 4.2 mm clearance | holds, but knife-edge |
| Robotiq85 | soup can ~62 mm | bracketed | **no offset needed** | 27/30 cells at δ=0 |
| Rethink | bowl rim 2.56 mm | 13.43 > 2.56 ✗ | infeasible | 0/37 |
| Rethink | bowl body 90.9 mm | 90.9 > 60.10 ✗ | infeasible | 0/37 |

So the gripper contributes a **threshold**; the object decides whether it is met. When the source
and target bracket the *same* feature, the correction is zero. When they must bracket *different*
features, the lateral correction is the radial gap between them.

**This is the deliverable we would put in front of a real robot**: from two numbers off a
gripper's CAD plus one measurement of the target, you can say *before running anything* whether a
given hand can take over a given demonstration, and why not if not.

---

## 5. Nine approaches that failed, and why

These matter because they rule out whole families.

1. **Constant additive action bias** — 0/3 at *every* scale. The policy is a feedback controller;
   an additive bias just moves the closed-loop fixed point, settling where `policy(image) = −bias`.
2. **Learned corrector** (5k-param MLP, matched-pose action pairs) — offline action agreement
   44.4% → 97.7%, closed-loop **0/3**. Offline agreement failed to predict closed-loop success
   four separate times in this project.
3. **Fingertip friction equalisation** — the Panda has a uniquely grippy 2.0 pad; equalising gives
   0/3, *worse*.
4. **Gripper-channel pass-through / sign fixes** — no effect; all grippers close on +1.
5. **Metric aperture commanding** — a **no-op** in this simulator: robosuite's `format_action` is
   `current + speed*np.sign(action)`, so command magnitude is discarded entirely.
6. **Moving the wrist camera in the model** to restore camera→TCP — gave 1/3, but it is a scene
   edit with no hardware analogue and was discarded as a solution.
7. **A bidirectional shim** (proprioception rescale + aperture translation + camera↔finger
   blending) — 0/3 in all modes, though blending doubled peak object lift.
8. **Cross-painting** — remove the real gripper from the wrist render, composite a rendered
   PandaGripper in its place. Reached **1/3** with one genuine task completion. Fails because
   pinning the proxy to the real pinch point puts it 44 mm further from the lens than natural,
   inflating its footprint to 29.7% against a true 19.0%, which broke navigation in 2 of 3 inits.
   Placing it at natural depth fixes the size and loses the correction. The image-versus-fingers
   conflict is not removable, only relocatable.
9. **Pinch-point alignment as the rule** — predicted a 1.2 mm correction where 35 mm was needed.
   Refuted outright.

---

## 6. The architecture that works

Invert the problem. Instead of deceiving the policy into driving the target, let it drive the
gripper it knows, and have the target **follow**.

```
   VLA ──▶ LEADER  (source gripper, real observations, fully in distribution)
                │  TCP + rotation + gripper command
                ▼
        correction layer  ◀── live pad geometry of BOTH hands
                │
                ▼
           FOLLOWER  (real target: no camera, no policy, Cartesian goals only)
```

The target's wrist camera is **never used**, so the occlusion, the 48 mm overshoot, and the
proxy-sizing conflict are all out of scope by construction.

### The correction

```
goal_follower(t) = tcp_leader(t) + R_ee(t) · offset_ee
```

Expressed in the **end-effector frame** so it rotates with the wrist. In world coordinates it
would be valid only at the one approach angle it was measured at, and would silently break on the
next task.

| term | source | value here |
|---|---|---|
| axial | `pinch_ee(source) − pinch_ee(target)`, recomputed each step from live pad geometry | ~6 mm |
| lateral | radial gap between the feature the source brackets and the one the target can, along the target's own closing axis | 45 mm (bowl), 0 (can) |
| sign | **not derivable** | one bit |

Everything gripper-side is read off the mounted hardware at runtime; swap the hand and the numbers
change themselves. The only object-dependent input is the feature width.

### Hard implementation constraint

**One MuJoCo environment per process.** Constructing a second EGL context leaves the first's
renders as uninitialised memory — the wrist view goes from 27.6% to 95.0% near-black at an
identical pose. It is not repaired by `make_current()`, by robosuite's observation path, by
reversed build order, or by disabling the second env's cameras: the damage happens at
*construction*. Three harnesses were built and three sets of results discarded before this was
respected. The follower now runs in its own process and never renders.

---

## 7. Results

Frozen Panda-only VLA, UR5e arm, target gripper following:

| | soup can | two-object (long-horizon) | bowl | drawer |
|---|---|---|---|---|
| **Robotiq85Gripper** | **3/3** | **3/3** | 0/3 | 0/3 |
| **RethinkGripper** | **2/3** | 1/3 | 0/37 (scripted) | — |

- `libero_object/0` — "pick up the alphabet soup and place it in the basket", lifts 179–215 mm
- `libero_10/0` — "put **both** the alphabet soup and the tomato sauce in the basket", two
  sequential grasps, 230–250 steps, correction stays valid across the whole sequence
- Tracking error 2.8–3.5 mm; identity control (Panda following Panda, zero offset) is 3/3

**The split was predicted before it was measured**, from the aperture rule. The two tasks that work
are the two where the target hand has real margin.

---

## 8. Open problems

1. **The bowl (0/3).** The rim is *below* the Robotiq's closed gap, so it cannot be pinched at all.
   The body *can* be enveloped, but with **4.2 mm of clearance per side** (99.4 mm span, 90.9 mm
   body) — the hand clips it on descent regardless of control effort. More sub-steps made tracking
   *more* variable, not less, because the spikes are collisions, not lag. This may be an object at
   the edge of feasibility rather than a method failure, but we have not proven that.

2. **The drawer (0/3), and it is a different regime.** Object-path straightness separates them:
   the drawer's leader path is **straightness 1.00** (a slide joint), the can's is 0.73 (a free
   arc). Once the hand engages a drawer it is kinematically coupled to a 1-DoF joint and has no
   freedom to absorb an offset — the error becomes a force fight and the follower jams (tracking
   median 137 mm, peak 288 mm). **Constrained-contact tasks need compliance along the constrained
   axis, not a position offset.** The aperture predicate is also silent about *hooking*, which an
   underactuated hand can do and a parallel jaw cannot. The leader baseline there is only 1/3 and
   needs repair before any transfer claim is meaningful.

3. **The sign of the lateral correction is not derivable.** Magnitude and axis are; the sign is one
   bit. Attempt-and-verify is affordable — failed attempts displace the object ~12 mm (max 17.7,
   0/18 over 20 mm on the bowl) — but the attempts are **not independent**, because 12 mm is about
   the width of the holding basin itself.

4. **Open loop.** Nothing flows back from the follower, so a dropped object is never retried. We
   believe the fix is to feed the follower's *object state* into the leader's scene (not the
   policy), so the policy sees the object still there and retries with no notion of what a Robotiq
   is. An attempt at this destabilised the leader by calling `sim.set_state()` every step, which
   perturbs the state the OSC controller integrates against.

5. **Scope.** The policy enacts the task on a **simulated source robot** while the target executes.
   This is trajectory transfer with an embodiment correction, not a frozen policy driving the
   target directly. On hardware it needs a source-embodiment twin fed by perceived scene state.

6. **Collateral contact.** Even on successful runs the target disturbs neighbouring objects the
   source clears — its fingers occupy more space and the approach is not planned around neighbours.

---

## 9. Methodological notes

Recorded because they cost real time and would cost a reader the same.

- **Offline agreement does not predict closed-loop success.** Four occurrences. A corrector reached
  97.7% action agreement and scored 0/3.
- **Diagnose the trajectory before interpreting the score.** A 0/3 was read as evidence against a
  prior result when the arm had never come within 57 mm of the object — a broken control loop, not
  a finding. Another 0/3 was a premature loop break hiding two successes.
- **Watch the video.** The single largest fix this project — the follower's scene was never pinned
  to the leader's, so the two robots tracked each other perfectly *in different scene layouts* —
  was invisible in every log column (tracking read 3.6 mm) and obvious in one frame.
- **Check what the metric is measuring.** Contacts and lift on the bowl task were being counted
  against `flat_stove_1_burner_plate`, because "plate" appears twice in the instruction. Caught only
  because a *successful* leader showed 0.0 mm of object motion.
- **Measure geometry, not parameterisation.** A width constant of 35.3 mm turned out to be the
  distance between jaw *centroids* rather than the free gap between pad faces. It failed its own
  control — demanding a 16.5 mm offset for the Panda on a rim the Panda grasps at zero — and only
  worked because it landed inside a valid band by luck.

---

## 10. What we would most like suggestions on

1. **The constrained-contact regime.** Is there a principled way to transfer a demonstration
   through a kinematic constraint (drawer, door, lever) between grippers, short of full force
   control on the target?
2. **Resolving the sign** without perception, or a better search than attempt-and-verify given the
   attempts perturb the scene by about the tolerance width.
3. **Closing the loop** without destabilising the leader — how to let the source-embodiment twin
   observe the target's world safely.
4. **Whether the aperture predicate generalises** beyond two-jaw hands: it is derived for
   mirror-symmetric two-finger grippers and says nothing about hooking, suction, or multi-finger
   hands.
5. **The 4.2 mm clearance case.** Is a knife-edge envelope grasp worth pursuing at all, or is the
   correct answer to declare it infeasible and pick a different grasp?
