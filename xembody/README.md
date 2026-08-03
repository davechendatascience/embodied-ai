# xembody — VLA intent to planner feasibility

A submodule-sized implementation of the intent/feasibility split: a policy
trained on one arm supplies **where** to act, a planner supplies **how** to get
there on a different arm.

    dense VLA rollout ──► keypose.extract ──► SE(3) keyposes
    depth + camera    ──► world.from_depth ──► obstacle boxes
                          planner.solve(keypose, world, robot) ──► joint targets

## Why it is shaped like this

**Nothing here imports mujoco, robosuite, LIBERO or torch.** The core takes
plain numpy arrays and returns plain numpy arrays, so it drops into any host
simulator or a real robot. Only `planner.py` touches cuRobo, and only when you
call it. That is the whole reason it fits in another repo.

| module | depends on | job |
|---|---|---|
| `frames.py` | numpy | pose conventions, TCP offset, the frame traps |
| `keypose.py` | numpy | dense EE trace -> sparse SE(3) keyposes |
| `world.py` | numpy | depth -> obstacle boxes |
| `planner.py` | cuRobo | keypose + world + robot -> joint targets |

## Six things that cost days. Do not rediscover them.

1. **A recorded pose may be a MIXED FRAME.** robosuite publishes
   `eef_pos` from the grip SITE and `eef_quat` from the hand BODY, 90 degrees
   apart. Treat them as one pose and every planner goal is rotated 90 degrees:
   the arm reaches the right place with the wrong wrist and grasps nothing.
   `frames.flange_from_obs` handles it; verify on your host with
   `frames.check_conventions`.

2. **Do NOT integrate action chunks to get a keypose.** An OSC controller
   realises ~25% of each commanded delta (measured, per step AND per chunk), so
   the product of deltas is a pose the arm never occupied. Read the ACHIEVED
   pose instead.

3. **Keyposes drop contact state.** They preserve prehensile manipulation
   (pick-and-place: 4/4) and lose non-prehensile (a drawer opened by hooking:
   every keypose reached within 5 mm, drawer moved 3.5 mm of 141). It is a
   property of the SOLUTION the policy chose, not of the task -- check the
   gripper channel before assuming a task is in scope.

4. **A grasp pose is IN CONTACT with what it grasps.** Collision-aware IK
   rejects it by construction. Exempt obstacles near the target
   (`planner.solve(..., exempt_radius=)`) or every grasp is vetoed.

5. **A voxel cuboid centred on a surface point inflates that surface by half a
   voxel** in every direction. Emit boxes smaller than the grid pitch, and know
   the trade: gaps of (1-scale)*pitch that only stay safe while the robot's own
   collision spheres are much larger than the gaps.

6. **The world is a snapshot.** A drawer moves as it opens; a held object rides
   the hand. Rebuild the world every replan, and move a held object OUT of the
   world and ONTO the tool, or the planner avoids something attached to its own
   gripper and reports infeasible forever.

## Host adapter

You supply four things; the package assumes nothing else:

```python
poses    (N, 7)  achieved EE poses, [x y z qx qy qz qw]
grip     (N,)    gripper command, >0 closing
depth    (H, W)  metres
K, T_wc          intrinsics and camera-to-world
```
