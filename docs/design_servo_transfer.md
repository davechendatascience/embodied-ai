# Servo transfer: moving the teacher to another arm

Companion to [`design_VLA_action_head.md`](design_VLA_action_head.md). The project's goal is to swap the
arm and keep the policy; this note plans the research on the layer that has to change when the arm does --
the servo that turns a body twist into joint motion -- and the screens that decide what the arm may try.

Written 2026-09-25, after the first cross-arm run of the skill teacher. Numbers are measured unless marked.
Claims about the outside literature cite their source and are summarized, not reproduced; verify before
building on a number from them.

---

## 1. What transferred on the first try

`Execution` now names the embodiment (`robot`, `gripper`; the Panda by default, so every Panda run is
unchanged), and the arm's command width is read from its controller instead of assumed to be 7. Nothing in the
teacher was changed.

UR5e (6 joints) with the Panda gripper, every task of libero_spatial, libero_object, libero_goal and libero_10
at initial state 0, LIBERO's horizons:

| suite | UR5e + PandaGripper | Panda (v48, same episodes) |
|---|---|---|
| libero_spatial | 10 / 10 | 10 / 10 |
| libero_object | 10 / 10 | 10 / 10 |
| libero_goal | 9 / 10 | 10 / 10 |
| libero_10 | 9 / 10 | 10 / 10 |

Episodes ran 10-20% longer (libero_spatial 0: 112 steps against 97). The geometry-driven teacher -- grasp
tiers, the reach screen, roofed entries, drawers, the stove knob -- carried over as written.

## 2. What broke, and why (measured)

Both failures are drawer hooks (libero_goal 0, libero_10 3): the tool reaches within 1-2 cm of the hook's
staging pose and stays there for the rest of the episode.

- **Not reach.** Every joint is at least 1.6 rad from its limit; no contact with the scene.
- **Not the gains.** robosuite's joint controller is inertia-weighted -- tau = M(q) (kp e - kd qdot) + bias
  compensation -- so the same kp gives the same closed-loop response on either arm. The Panda tracks its
  servo reference to 0.2 mrad in the same phase.
- **Self-collision.** The hook's pitched tool frame needs a wrist roll that folds the Panda hand into the
  UR5e's forearm: `robot0_forearm_link` against `gripper0_right_gripper`, 1.7 mm deep. The last two wrist joints
  sit 0.046 and 0.0998 rad behind the servo's reference (the second at the 0.1 rad lead cap), the controller
  asks 368 Nm and gets the 28 Nm limit, and the contact holds.
- **Why the teacher did not see it.** The reach screen grades contacts between the robot and the scene only;
  robot-against-robot contacts were never a case on the Panda-with-Panda-gripper combination.
- **Why the servo cannot escape.** A 6-joint arm has no redundancy for a 6-D pose: the pose has a finite set of
  joint solutions (up to 8 for a UR arm), and a damped least-squares step from the current joints stays on the
  current one. The collision-free solution, if there is one, is a different branch (a wrist flip).

The lesson, in one line: *the teacher's geometry transferred; the execution layer's unstated assumptions --
no self-collision, a redundant joint, the Panda's reach of wrist roll -- did not.*

## 3. What the literature offers

- **Velocity dampers in a QP.** Faverjon and Tournassoud (1987) turned collision avoidance into linear
  inequality constraints on joint velocity -- for each close pair of bodies, the approach speed along their
  separating normal is bounded by a multiple of the remaining clearance -- solved together with the task in a
  quadratic program; Kanoun et al. (2011) generalized it to prioritized tasks. Self-collision, obstacles and
  joint limits all take the same form. ([RSS 2008, local collision avoidance](https://www.roboticsproceedings.org/rss04/p20.pdf);
  [set-based tasks in multi-priority IK](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00016/full))
- **EmbodiSteer (2026).** A Cartesian (end-effector) diffusion policy made safe on a new arm by lifting its
  sampling into that arm's joint space and solving a CBF-style QP that keeps whole-body clearance above a
  margin. Reported: across 9 simulated arms, success 35.7% to 64.2% and collisions 57.6% to 11.5%; on real
  UR5 and Panda, 25/60 to 47/60 with obstacles. Its stated limits are ours too: local and reactive, and no
  IK-branch selection. ([arXiv 2606.12965](https://arxiv.org/html/2606.12965))
- **IK branches.** A UR arm reaches a full pose with up to 8 joint solutions; enumerating them and choosing the
  nearest feasible one avoids the elbow and wrist flips a local solver cannot see coming.
  ([UR5 IK](https://alexanderelias.com/ur5-ik/); [closed-form vs DLS on the UR5e](https://github.com/dumitrubogdan03/ur5e-motion-planning))
- **Action-space conventions.** Octo standardized on end-effector deltas, with each robot's own controller
  hiding the embodiment ([Octo](https://arxiv.org/html/2405.12213v2)); the mismatch then surfaces exactly where
  we found it -- constraints and tracking, not the policy. Our body twist with a URDF-parameterized decode is
  the explicit version of that choice.
- **The visual gap** (the student seeing a new gripper) is a separate problem, addressed for example by
  masking the end-effector from the VLA's view ([Cloak](https://arxiv.org/pdf/2606.22836)). Out of scope
  here.

## 4. The plan

Each step is a design branch first (proposed, verified, declared), then code, then a measurement.

1. **Instrument the servo.** Per step, cheap and always on in experiments: the joint gap between the servo's
   reference and the measured joints, the fraction of joints at their torque limit, the robot's self-contacts,
   and the IK branch. A failure then names its mechanism instead of reading "stalled".
2. **The reach screen grades self-contact.** A candidate pose where two robot bodies that are not neighbours in
   the kinematic tree interpenetrate is graded as a fixed collision. On the Panda this should change nothing
   (to be checked episode for episode); on the UR5e it rejects the folded hook pose so the teacher looks
   elsewhere.
3. **The collision-aware servo.** The prototype of 2026-09-25 (a QP after the servo's step: stay close to the
   commanded step, tool motion weighted, subject to velocity dampers on every robot-scene pair within 3 cm)
   gains self-pairs and the exemptions a default needs: the object being grasped or held, the handle or
   drawer being driven, the inside of the container being placed into, and the support under a deep pinch.
   Measured on the prototype: +0-6% time per step; it holds the hand 5 mm off an obstacle instead of pressing
   it, and leaves a succeeding episode unchanged in outcome. Adopted as the default only if a full sweep loses
   no task and the cost stays within 10%.
4. **Branch-aware IK for 6-joint arms.** The reach screen solves each candidate on every IK branch (analytic for
   the UR5e; multi-seed numeric for general arms, as `SimArm` already does for start poses) and keeps a branch
   on which the whole approach -- column, approach probes, release -- is feasible; the servo follows the chosen
   branch. This is what gets the UR5e's hook out of its fold.
5. **The gripper.** The Robotiq 85: aperture from its six hinge joints instead of the Panda's two slides, its
   pad geoms by name, its hand's shape scanned rather than assumed (the planner's `GripperScan`), and the
   flange-to-tool offset (0.1450 m against the Panda's 0.0970 m, already measured by `gripper_geom`).
6. **The cross-embodiment matrix.** The teacher on {Panda, UR5e, iiwa, Kinova3, Jaco} x {Panda gripper,
   Robotiq 85} over the 40 benchmark tasks at initial state 0 (later all 50 initial states): success, steps,
   and the step-1 diagnostics per cell. This is the research dataset; it says which of steps 2-5 matter most
   and for which arm.

### The first matrix (2026-09-26)

The teacher unchanged, the Panda gripper on every arm, the 40 benchmark tasks at initial state 0, servo
diagnostics on (`runs/matrix_<arm>`; commit cb033ff):

| arm | spatial | object | goal | 10 | total | steps at a torque limit | steps in self-contact |
|---|---|---|---|---|---|---|---|
| Panda | 10/10 | 10/10 | 10/10 | 10/10 | 40/40 | 0.9% | 0% |
| UR5e | 10/10 | 10/10 | 9/10 | 9/10 | 38/40 | 13.4% | 6.3% |
| iiwa | 10/10 | 10/10 | 7/10 | 9/10 | 36/40 | 5.2% | 0% |
| Jaco | 9/10 | 10/10 | 9/10 | 8/10 | 36/40 | 35.5% | 5.3% |
| Kinova3 | 9/10 | 10/10 | 7/10 | 6/10 | 32/40 | 30.2% | 0% |

142 of 160 on the four new arms. The 18 failures by the diagnostics' first-cut mechanism: **tracking under
torque saturation 8**, **pinned at a joint limit 5**, **self-collision 2**, teacher-level 3 (the servo tracked;
the plan or grasp does not suit the arm).

What the matrix taught beyond the counts:

- **The start pose is a design choice.** The first run copied LIBERO's recorded Panda joint angles onto every
  7-joint arm (the init-state remap recognized the Panda by joint count): the Kinova3 started with a joint at its
  limit and the Panda hand folded into its upper arm, 11 of 40. With its own ready pose, 32 of 40. The iiwa went
  the other way, 38 of 40 in the Panda's angles and 36 in its own: a redundant arm keeps the elbow family it starts
  in for the whole episode, since the local IK never leaves it, and the Panda's family happened to suit these tasks
  better. Branch and family awareness matters on 7-joint arms too.
- **Weak actuators are the commonest break.** The servo's lead (0.1 rad per period) and the controller's
  inertia-weighted gains were sized on the Panda; the Kinova3's and the Jaco's actuators saturate on a third of
  their steps and the arm lags the reference by 0.06-0.1 rad.

### Understanding the 18 failures (traced step by step, 2026-09-26)

The diagnostics' first-cut labels were read against per-phase traces of every failure (saturated joints, tracking
error, joint speed, the robot's contacts with the scene):

| what happens | episodes | where |
|---|---|---|
| contact stall: the arm or hand pushes on the scene or the target, frozen at the servo's lead cap with a joint at its torque limit for hundreds of steps | 4 | UR5e on the wine rack (libero_10 3); Kinova3 on the cabinet top (goal 3); Kinova3 and Jaco hand on the moka pot during the grasp descent (libero_10 2) |
| frozen at a joint limit: the reference cannot advance, about 590 steps at 1 mrad of error and no motion | 3 | iiwa goal 2, 7, 9 |
| grasp lost during loaded motion: the object drops mid-lift or mid-carry while the arm saturates | 5 | Kinova3 goal 4, 9, libero_10 4; iiwa and Jaco libero_10 9 |
| the squeeze never holds: the jaws close beside the object and the teacher waits to the horizon | 2 | Kinova3 libero_10 6, Jaco goal 2 |
| out of the envelope: reaching into the top drawer, the arm cycles up and over while saturated, touching nothing | 2 | Kinova3 and Jaco spatial 4 |
| self-fold | 1 | UR5e goal 0 |
| regrasp loop | 1 | Kinova3 libero_10 9 |

Three concepts the Panda teacher never needed are missing:

- **Blocked progress** (9 episodes). The servo commands its reference whatever resists it, and the teacher waits for
  a phase to finish. Blocked by a contact, a joint limit or an empty squeeze, the episode stalls to the horizon. The
  episode log already detects stalls ("tool still for 30 steps"); nothing acts on them.
- **Grip under the arm's dynamics** (6). The acceleration bounds that keep a grip (AXM-acceleration-loads-the-grip)
  were measured on the Panda; on arms whose joints saturate on a third of their steps the motion is no longer the
  commanded one, and objects slip.
- **The arm's envelope and families** (6, overlapping). What is reachable, and without passing a limit, depends on the
  arm's joint limits and the elbow family it starts in.

Also measured: every arm's joint controller (kp 4000, inertia-weighted) saturates at a tracking error of 1-24 mrad,
while the servo allows a lead of 100 mrad; the joint inertias are nearly equal across arms (the model's rotor inertia
dominates), so the arms differ mainly in torque capacity (Panda 80/12 Nm, UR5e 150/28, iiwa 176/110/40, Kinova3
32/13 with half of the shoulder's taken by gravity, Jaco 30.5/6.8).

**Revised order**, by episodes each fix can reach: (a) a servo whose lead and acceleration are sized to each
arm's torque capacity -- measured from its model (8 failures); (b) branch and family awareness, including the
choice of start pose (5); (c) the self-contact screen and dampers (2). The Robotiq port (step 5) is independent
and follows. The non-Panda trials are not ingested into `CTR-teacher-reliable`, whose slices are keyed by task and
revision and would mix embodiments; a cross-embodiment contract is to be declared first.

### The first matrix measured disturbed scenes (found 2026-09-26)

Tracing the grasp losses further refuted two explanations before finding a confound under all of them:

- **Not acceleration.** Peak tool accelerations while holding were 1.5-5.3 m/s^2 on the other arms and 3.6-4.0 on
  the Panda. What differed was the grasp: the screen passed fewer candidates (forced choices: Panda 1, UR5e 1,
  iiwa 5, Kinova3 5, Jaco 3 of 44-47; fallback tiers 2, 3, 0, 10, 13), rejected at the grasp, approach, pre-grasp
  and column probes for IK not converging.
- **Not the IK's starting point.** At each forced choice, the candidates converging at every probe numbered the
  same from the arm's joints and from any of 18 seeds (iiwa goal 2: 14 of 240; Jaco and Kinova3 spatial 4: 40; Kinova3
  goal 3: 0), so the missing grasps are outside the arm's reach, not in another IK branch.
- **The scene.** Kinova3 goal 3 had its bowl 1.04 m from the base against the Panda's 0.58: robosuite's default start
  pose for each model, taken with the objects copied from the initial state, knocks objects during the reset. Over
  the 40 matrix tasks at init 0, some object ended more than 5 mm from where the Panda's reset leaves it on 2 (UR5e),
  15 (iiwa, the wine bottle on every goal task), 22 (Kinova3, a bottle 3 m off the table in goal 3, the bowl 415
  and 607 mm away in goal 2 and 9) and 3 (Jaco).

Part of the understanding table above was therefore measured on disturbed scenes. The fix is
`BRN-other-arm-starts-at-the-panda-tool-pose`: another arm starts with its tool at the pose LIBERO's Panda was
recorded at (every registered arm's base stands at the Panda's, measured on all 130 tasks), in the first IK
solution from an ordered seed list that is reachable and penetrates nothing; and numpy's global generator is
advanced so that LIBERO's fixture draws begin where the Panda's do (the robot reset draws one normal per arm joint
first; the 6-joint UR5e drew its fixtures from a shifted stream). With both, iiwa, Kinova3 and Jaco leave every
object exactly where the Panda's reset does on 40 of 40 tasks; the UR5e on 18, and differs by more than 5 mm on 3
(libero_10 0, 1, 7), where the recorded state puts a ketchup bottle 29 mm inside the table and how it is ejected
depends on the simulated system's joint count -- identical fixtures and objects at placement, different after.
The matrix is to be re-run on this start before the failures are re-traced.

## 5. What has to hold (for the ledger)

Proposed as branches in `consistency.yaml`, each verified before it is built:

- **The servo keeps clearance.** Where a robot body is within the cutoff of a scene body or of another robot
  body, and neither is exempted by the current skill, the executed joint step never closes that pair's
  distance faster than the damper allows; where no pair binds, the executed step is the servo's step.
- **The reach screen grades self-contact.** A candidate whose probed configuration interpenetrates two robot
  bodies that are not kinematic neighbours is not taken while a candidate without it passes.
- **Branches are finite and a local step keeps its branch.** For a non-redundant arm at a non-singular pose,
  the joint solutions are finitely many and isolated, and a damped step from a configuration on one branch,
  small enough, stays on it -- the reason a branch has to be chosen, not stumbled into.
- **The embodiment is a declared input.** Every quantity the teacher reads about the arm or gripper (joint
  count, limits, torque limits, tool offset, hand shape, pad names) comes from the loaded model or a
  measurement of it, never from a Panda constant.

### Theory status (2026-09-25)

Two of these are staged in the ledger and were proven 3 of 3 by the verifier as first stated, then
restated to answer its design concerns (re-verification pending):

- `BRN-servo-keeps-clearance` -- the damper servo over robot-scene and robot-self pairs. The restatement adds
  the joint limits to the feasible set (a damped step could otherwise command past a limit), guards only geoms
  the arm moves (a base or mount within the margin would leave no feasible step and silently disable the rule),
  skips pairs within the gripper, requires the cutoff to exceed the margin, and passes the servo's references
  through unaltered wherever no inequality binds, so that episodes stay bit-identical.
- `BRN-reach-screen-grades-self-contact` -- the screen rejects folded-arm candidates. The restatement says the
  robot includes its gripper, excludes pairs within the gripper (two fingers are not parent and child), and
  leaves two things to measurement: a census of the Panda's probed configurations (claim 2's "nothing changes"
  needs no self-contact at any depth there), and whether the UR5e's fold happens at a probed configuration at
  all or only while the servo tracks between them.

Re-verified the same night: `BRN-servo-keeps-clearance` proven 3 of 3 as restated; `BRN-reach-screen-grades-self-contact`
doubted on one gap. To pick up next:

- **The screen's fingers.** A probed configuration fixes the arm's joints only, so the self-contact test does
  not say where the fingers are. State that the gripper's joints hold their value at screen time (or name the
  opening screened at each pose), and list as not claimed a fold that appears only at another opening -- an
  open fingertip meeting the forearm at a pre-grasp pose when the screen ran closed.
- **Bodies bolted together.** Two robot bodies joined through a body with no joint (the last arm link and the
  gripper's root, say) are neither the same body nor neighbours in the tree, so the servo would guard them on
  every step: overlapping, they leave no feasible step and switch the damper off everywhere; merely close, they
  cap the flange's speed and episodes stop being bit-identical. Exclude pairs with no joint between them.
- **`DEF-reachable-pose`** says "this is the teacher's screen"; once self-contact is graded its fixture clause
  should read "nor itself". Changing it makes the nodes that cite it stale.

The other two statements of this section are not branches: that a non-redundant arm's IK branches are finite
and a local step keeps its branch is a lemma that needs a kinematics premise, and that the embodiment is a
declared input is a property of the code, for a component-belief contract.

### The open design problem: what a student may touch

The servo branch's exemptions come from the running policy. The skill teacher knows what it must touch (the
object it grasps, the bar it hooks, the support under a deep pinch); a learned student declares nothing, so on
this servo its fingers would be held off the very object it reaches for, where the teacher, in the same state
with the same action, would touch it. Train and test execution would differ, against the rule that they are
identical. Before a student runs on this servo, the exempt set has to be something both can use:

- computed from the state -- for example the body between the open jaws within the pads' reach, the body a
  closing gripper already touches, and the support under the tool's column; or
- carried in the action -- a contact flag the policy emits and the decode honours, learned from the teacher's
  labels like the gripper aperture.

The first keeps the student's action space as it is; the second is more general and more to learn. This is
the first decision of the servo-transfer work that changes the student, so it is taken before step 3 is built.

**Decided 2026-09-26: per-intent state rules.** Each intended contact is its own rule, computed from the state
alone and so the same for the teacher and the student: the body between the open jaws, within the pads'
reach; the handle or joint body the tool point is at; the support under the tool's column during a deep pinch;
the container whose interior the tool is inside; and the object held. Chosen over a single "finger zone"
exemption for precision -- a fingertip striking a wall stays guarded -- at the price of more rules, each to be
stated as a branch and verified. Each rule is measured against the teacher's own intended contacts: on the
Panda, an exemption set that blocks any contact the teacher makes today is wrong.

**Decided 2026-09-26: measure first.** Step 1 (diagnostics) and step 6 (the cross-embodiment matrix) come before
any fix, so the fixes are chosen by how many episodes each failure mechanism costs on which arm.

## 6. Risks and open questions

- **Exemptions decide everything.** A clearance margin that is right for a free carry is wrong for a hook that
  must press its bar or a pinch that must reach the table. The exemption list is the design, and it has to be
  stated per skill.
- **Local fixes do not reach global problems.** Dampers stop a collision; they do not find the way around it.
  Branch choice (step 4) and, where it fails, a different grasp are the global part.
- **The student sees obstacles only through perception.** The teacher can use exact geometry; the student's
  decode needs distances from depth. Keep the two labelled: an experiment that hands the student the simulator's
  geometry measures the controller, not the system.
- **Torque limits differ** (the Panda's wrist 12 Nm, the UR5e's 28 Nm): a trajectory feasible on one arm can
  saturate another. Step 1 measures how often.
