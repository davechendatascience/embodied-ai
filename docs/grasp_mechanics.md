# Grasp mechanics in LIBERO's simulator

What decides whether a pinch by LIBERO's Panda holds an object, measured from the MuJoCo
model and from replays of failed teacher episodes (revision 34ba51f). Written because the
teacher's grasps were chosen and judged by geometry alone, and the failure videos
(videos/teacher_fail_test) showed grips that slip, pivot and let go.

## Summary

- **Weight is not what fails.** Every object in libero_object, spatial and goal weighs
  5.2-31.7 g (the basket 136 g): at most 0.31 N of gravity, against squeeze forces of
  1-20 N at a friction coefficient of 2. Loads from the arm's motion are as small: the
  bowl under the 2 m/s^2 execution bound carries 0.011 N of inertia.
- **What fails is contact stability**, and three facts of the model decide it:
  1. the squeeze is a **spring proportional to the object's width**: each finger is a
     position servo, so a 2.6 mm rim wall is squeezed with about 1.3 N and a 40 mm box
     with 20 N;
  2. contacts have **no torsional friction** (condim 3): nothing but the spread of the
     contact points resists the object pivoting about the jaw axis;
  3. the pads grip only where they **meet parallel faces**; a pinch that closes on edges,
     or with one finger working against the supporting surface, pivots or squirts out.
- The teacher also **cannot tell when it has lost the object**: its holding test flickers
  (lift and squeeze alternate every step), and it keeps carrying an empty gripper.

## The model

Read from the MuJoCo model of every LIBERO scene (they share the robot and gripper).

| | value |
|---|---|
| finger actuators | position servos, gain 1000 N/m, force limited to +-20 N, target 0..40 mm per finger (never beyond closed) |
| finger joints | damping 100, frictionloss 1.0 N, armature 1.0 |
| pad geoms | boxes 16 x 8 mm, friction (2.0, 0.05, 1e-4), condim 3, solref (0.01, 0.5) |
| object geoms | boxes (every object and fixture in these suites), friction (0.95, 0.3, 0.1), condim 3, solref (0.001, 1) |
| contact model | elliptic friction cone, impratio 20; MuJoCo takes the larger friction of a pair, so pad-object contacts use 2.0 |
| gripper geometry (measured, sim/scan.py) | pads span 11.6 mm behind to 4.4 mm past the tool point; the finger meshes reach 9.3 mm past it; the palm begins 31 mm behind |

Object masses and bounding boxes:

| object | mass | box (mm) |
|---|---|---|
| butter | 5.2 g | 76 x 40 x 17 |
| akita_black_bowl | 5.6 g | 107 x 107 x 51 |
| cream_cheese | 6.2 g | 81 x 43 x 18 |
| cookies | 9.6 g | 62 x 83 x 19 |
| glazed_rim_porcelain_ramekin | 9.7 g | 89 x 89 x 42 |
| chocolate_pudding | 10.2 g | 80 x 46 x 27 |
| plate | 11.5 g | 138 x 138 x 19 |
| bbq_sauce | 12.2 g | 47 x 29 x 107 |
| wine_bottle | 15.4 g | 43 x 44 x 157 |
| salad_dressing | 21.8 g | 53 x 146 x 36 |
| ketchup | 23.2 g | 56 x 146 x 37 |
| alphabet_soup | 25.9 g | 62 x 76 x 62 |
| tomato_sauce | 26.6 g | 62 x 76 x 62 |
| orange_juice, milk | 31.7 g | 53 x 131 x 53 |
| basket | 136.1 g | 170 x 157 x 142 |

### The squeeze is a spring

A finger servo drives toward its target with 1000 N/m and the target cannot pass fully
closed, so with the fingers stopped by an object of width w each finger pushes with about
1000 x w/2 N (capped at 20 N; joint friction takes up to 1 N of it):

| grasp | width | squeeze per finger |
|---|---|---|
| bowl rim wall | 2.6 mm | ~1.3 N |
| bowl, at first contact in spatial 6 | 8.1 mm | 4.0 N measured |
| cream cheese, across its height | 17.9 mm | 15-20 N measured |
| butter, across its narrow face | 39.5 mm | 20 N (capped) |

A thin grasp is a weak grasp by construction, and it weakens further as the object slips,
because the aperture closes and the spring relaxes.

## Two failures, measured

Replays of the failing episodes with every pad contact's normal force and friction-cone
use (|tangential| / (mu x normal); 1.00 means sliding), and the object's slip and rotation
relative to the tool since the squeeze.

### The bowl pivots out of a rim pinch (libero_spatial 6, episode 0)

The planner chose a tilted side pinch on the rim (tier sides, 18.7 mm planned width).

- The fingers first touched the bowl at an **8.1 mm aperture**, not across the 2.6 mm wall:
  they closed on edges. Squeeze 4.0 N per finger.
- Through the lift and carry the aperture crept from 8.1 to 4.2 mm and the squeeze fell
  from 4.0 to 2.1 N, while the bowl **rotated from 0.3 to 16 degrees about the pinch**.
  Pad normal forces were 0.1-0.7 N, the finger meshes 0.2-1.4 N; some contacts sat at cone
  1.00 (sliding).
- At step 97, 17 cm up, pad contact vanished and the bowl fell onto the cookie box.

The bowl weighs 0.055 N; friction capacity was in the newtons. What let go was the pivot,
which nothing resisted, and the wall sliding out between edge contacts.

### A side pinch squirts the bar out against the floor (libero_object 1, episode 7)

The top-down grasps on the cream cheese were blocked by the arm meeting the basket, so the
planner forced a side pinch: horizontal approach, jaw axis vertical, closing across the
bar's 17.9 mm height, the lower finger between the bar and the floor.

- The squeeze was strong: 15-20 N per finger, contacts up to 10.9 N.
- During the lift the bar was pressed **down** (8.9 mm -> 1.4 mm above the floor), then the
  contacts saturated their friction cones (1.00) and the bar squirted out sideways at step
  97.
- Meanwhile the teacher's phase alternated `lift` / `squeeze` every step or two: its holding
  test (jaw speed below a threshold, or the object touching only the gripper) flickered.

## What follows for the teacher (proposals, to be measured)

1. **Check that the pads closed on faces.** The aperture at first contact should match the
   planned width; a large mismatch (8.1 mm for an 18.7 mm plan) means edges, and the grasp
   should be redone before lifting.
2. **Judge holding by contact and motion, not jaw speed.** Both pads touching the object
   with a normal force above a floor, and the object moving with the tool (its pose
   relative to the tool steady). This catches a pivot as it starts (16 degrees over 45
   steps above) and a loss at once.
3. **Regrasp on loss, from the current state.** When holding fails, the object's current
   pose is scanned and a new grasp planned, instead of carrying an empty gripper to the
   target; this is where planning the motion after scanning the scene belongs.
4. **Prefer pinches that resist pivoting.** With no torsional friction, the contact points
   must be spread along the pads: parallel faces over the pads' 16 mm width, not a tilted
   pinch on a curved wall.
5. **No pinch with a finger working against the support.** A vertical jaw axis near the
   floor presses the object into it.

## Open questions

- Which rim pinch holds the bowl best: jaw radial across the wall at a straight segment,
  how deep below the rim, and whether the bowl's 40 wall boxes leave seams the pads
  straddle. This is where most spatial failures are.
- Whether any bowl grasp is wider than the rim: the bowl is 107 mm across, beyond the
  75 mm opening.
- The squeeze floor: joint frictionloss (1 N) and damping (100) make the achieved force
  depend on the fingers' recent motion; the achieved force per contact should be measured
  for successful grasps too.
- libero_goal 3's re-pick slip after the drawer opens (in the failure videos) has not been
  replayed yet.
