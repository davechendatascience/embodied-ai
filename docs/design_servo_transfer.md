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
