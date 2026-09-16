Here, we like to build our own VLA.
How the VLA differs from others is that it can take an arbitrary urdf and automate cross embodiment easily.
First we need to survey how most SOTA VLAs are built and trained.
Then we need to use the Modern_Robotics_complete.pdf for use in designing the action head. I believe that there are a lot of knowledge in robotics that can be baked into the action head, like obstacle avoidance, with IK solving and grasping etc.
The easiest way is that we train on libero and then we switch the urdf to another to see if it automatically generalize.
---

## Where this stands (2026-09-16)

The goal above is unchanged; the route to testing it changed once, for a measured reason.

- **Survey and action head** — done: [`design_VLA_action_head.md`](design_VLA_action_head.md).
  The head emits an embodiment-free body twist plus a gripper aperture; fixed, URDF-parameterized
  layers (PoE kinematics, damped least-squares IK, null space) decode it for the arm at hand.
  The conversion, Jacobian, IK and null-space layers are verified to machine precision on five
  arms (`belief.yaml`, gate 1).
- **"Train on LIBERO, then switch the URDF"** — not yet, deliberately. Trained on LIBERO's
  demonstrations, a policy with its camera images zeroed did as well as the sighted one: the
  suite's fixed initial states let one memorized trajectory succeed, so a transfer result
  measured there would say nothing about the head. The policy first has to be shown to act on
  what it sees (gate 2): randomized object layouts and robot start poses, per-task scripted
  teachers, DAgger, and a blind control that must fail.
- **Current result** — DINOv2 patch-token VLA on libero_spatial: 82% over 200 randomized
  episodes, blind copy 4.5%. The remaining failure is the drawer task, addressed by a gripper
  servo; an aligned rebuild is running.
- **Next** — finish libero_spatial, extend to the other LIBERO suites, then swap arms
  (gate 3), which is the test this document asked for.
