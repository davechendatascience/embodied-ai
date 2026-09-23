# Skill teacher vs. LIBERO's human demonstrations, per task

Status as of 2026-09-23. Sources:

- Human demos: all 1500 of LIBERO's demonstrations (50 per task) for libero_spatial, libero_object
  and libero_goal, replayed from their recorded simulator states by `tools/demo_survey.py`
  (per-demo JSON in `runs/demo_survey/<suite>.json`, summaries in `.txt`). Each demo's own fixture
  poses are loaded from its model XML; without them 37 of 50 final states of libero_goal 9 failed
  LIBERO's predicate, with them 1488 of the 1500 demos meet their goal at the last recorded state.
- The teacher: revision `skill_teacher:48bb0345c7` (commit 9e32c47), 50 episodes per task at seed
  555, horizon 600 (`runs/skill_v2`), ingested into component-belief (CTR-teacher-reliable,
  CTR-teacher-attempts, CTR-teacher-gentle).
- "Budget" is the step limit LIBERO evaluations usually use: spatial 220, object 280, goal 300. The
  teacher is scored at 600. A student imitating demos longer than the budget runs out of steps.

## Summary

| suite | teacher success | within budget | human median length | teacher median length (successes) |
|---|---|---|---|---|
| spatial | 478/500 | 391/478 | 95-148 | 104-247 |
| object | 493/500 | 493/493 | 128-158 | 222-266 |
| goal | 363/500 | 335/363 | 89-198 | 74-574 |

CTR-teacher-gentle (released <= 5 mm above rest, no drop) is refuted at 0.00 over 1500 episodes.

## Cross-cutting findings

1. **Mid-air drops are hidden by success.** The rim-pinched bowl falls during the carry in 3-33 of 50
   episodes on every bowl task (spatial 1: 33, spatial 7: 28, goal 8: 28, goal 1: 27) and still
   scores, because it falls on to the target. No human demo drops it. Mechanism (measured): the
   pinch is a pivot nothing resists (condim 3); the velocity change at a phase switch pops it.
   Bounding the commanded twist's change at 0.5 m/s^2 removed the drops in 12 probed episodes; the
   bound gated on the jaws being driven closed is being swept. About 140 of the 166 failed episodes
   are "after losing it in place" -- the same drops, followed by a failed regrasp.
2. **Human grasps are top-down in almost every pick.** Bowls: a rim pinch about 43 mm from the
   bowl's axis, 25-30 mm up, jaws 6-10 mm. The exceptions are the wine bottle: on the rack, taken
   from the side (approach (0.07, -0.87, -0.44)); on the cabinet top, tilted in half the demos.
3. **Humans release low where it matters and drop where it does not.** Bowls on plates/stove/
   cabinet: let go 1-6 mm above rest. Into the basket: dropped from 32-140 mm (In needs only
   containment). Bowl into the drawer: 48 mm. The teacher lets bowls go 15-27 mm above rest and
   lowers every object into the basket slowly.
4. **Humans move articulations without grasping.** Both drawers are opened with the jaws wide
   open (77-80 mm) and never closed: the middle drawer from the side with the bar between the open
   fingers; the top drawer from straight above, one open finger dropped behind the bar and dragged
   (tool ~42 mm on the opening side of the bar, ~17 mm above it). The stove knob is the only
   articulation they grasp. The teacher pinches the bar from the side with the wrist horizontal,
   then retreats and re-orients.
5. **Pushing the plate is caging, not pushing from behind.** Jaws open ~60 mm, top-down, tool point
   15 mm above the plate's origin (fingertips inside the dish), leading the plate's centre by
   ~31 mm along the motion: the fingers drag the plate by the inside of its leading rim. ~250 mm in
   ~80 steps. The gripper is never commanded closed. The teacher has no push skill.
6. **The rack takes the bottle lying along its cradle.** Every demo ends with the bottle's axis on
   the rack region's z axis (0.98 in the region frame, sd 0.02), 59 degrees from upright. The bottle
   is taken upright from the side, the wrist turns ~55 degrees during the carry, and it is let go
   already tilted ~60 degrees, 7 mm above where it settles. The teacher sets it down upright.
7. **The teacher is a policy, with three history-dependent caches.** Every action is chosen from
   the current state, so it can label any state a student reaches. But the grasp, the handle frame
   and the clearing spot are chosen once per episode and cached, which makes those labels depend on
   the episode's history (AXM-dagger-needs-markov-labels).

## Per task

Teacher columns: success / 50, median length of successes, successes within budget, episodes with
a mid-air drop, median release gap (mm; "-" where no release was measured -- the detector missed
releases where the object touched its container first; fixed after this sweep).

### libero_spatial (budget 220)

| task | human len | teacher | len | in budget | drops | release | working | missing |
|---|---|---|---|---|---|---|---|---|
| 0 bowl between plate and ramekin | 95 | 50 | 130 | 50/50 | 3 | 16.5 | pick/place | low release |
| 1 bowl next to ramekin | 132 | 50 | 149 | 49/50 | 33 | 24.7 | success | drops, release |
| 2 bowl from table centre | 117 | 50 | 140 | 50/50 | 25 | 14.7 | success | drops |
| 3 bowl on cookie box | 99 | 49 | 104 | 44/49 | 12 | 15.6 | | drops, speed |
| 4 bowl in top drawer | 148 | 41 | 189 | 39/41 | 6 | 20.1 | | lost in place, regrasp near the cabinet |
| 5 bowl on ramekin | 115 | 50 | 197 | 47/50 | 0 | 14.3 | | speed |
| 6 bowl next to cookie box | 124 | 44 | 178 | 29/44 | 20 | 20.4 | | drops, speed |
| 7 bowl on stove | 137 | 46 | 223 | 22/46 | 28 | 27.4 | | drops, speed |
| 8 bowl next to plate | 114 | 49 | 139 | 43/49 | 27 | 17.2 | | drops |
| 9 bowl on wooden cabinet | 136 | 49 | 247 | 18/49 | 23 | 26.0 | | speed (1.8x), drops |

### libero_object (budget 280)

Humans: one top-down grasp, the object dropped into the basket from 32-140 mm, 128-158 steps.

| task | human len | teacher | len | in budget | working | missing |
|---|---|---|---|---|---|---|
| 0 alphabet soup | 153 | 50 | 252 | 50/50 | all | speed (1.6x): lowers into the basket instead of dropping |
| 1 cream cheese | 139 | 43 | 244 | 43/43 | 43 | the planner offers only side grasps (faces/parts tiers empty); all 7 failures forced one. Humans take it top-down, jaws ~45 mm |
| 2 salad dressing | 128 | 50 | 223 | 50/50 | all | speed |
| 3 bbq sauce | 139 | 50 | 226 | 50/50 | all | speed |
| 4 ketchup | 151 | 50 | 245 | 50/50 | all | speed |
| 5 tomato sauce | 141 | 50 | 231 | 50/50 | all | speed |
| 6 butter | 154 | 50 | 266 | 50/50 | all | speed |
| 7 milk | 141 | 50 | 245 | 50/50 | all | speed |
| 8 chocolate pudding | 158 | 50 | 266 | 50/50 | all | speed |
| 9 orange juice | 133 | 50 | 222 | 50/50 | all | speed |

### libero_goal (budget 300)

| task | human | teacher | len | in budget | drops | working | missing |
|---|---|---|---|---|---|---|---|
| 0 open middle drawer | 136, hook with open jaws | 50 | 74 | 50/50 | 0 | faster than humans | - |
| 1 bowl on stove | 100 | 50 | 149 | 50/50 | 27 | success | drops |
| 2 bottle on cabinet top | 104, grip 111 mm up the bottle | 50 | 161 | 50/50 | 0 | all | speed |
| 3 open top drawer, bowl inside | 198: hook the drawer open from above, one rim pinch, drop 48 mm in; no relocation | 10 | 574 | 0/10 | 47 | the drive itself (opens -0.149) | relocation step (~200 steps), side pinch + retreat + re-orientation, drops on the lift to the drawer |
| 4 bowl on cabinet top | 99 | 49 | 144 | 49/49 | 27 | success | drops |
| 5 push plate to stove front | 145, cage the dish with open jaws | 3 | 293 | 3/3 | 15 | - | a push skill |
| 6 cream cheese in bowl | 104, dropped 26 mm in | 50 | 204 | 34/50 | 0 | success | speed; release 60 mm |
| 7 turn on stove | 89, knob grasped top-down | 50 | 147 | 50/50 | 0 | all | speed |
| 8 bowl on plate | 92 | 50 | 174 | 48/50 | 28 | success | drops |
| 9 bottle on rack | 169, side grasp, wrist turned ~55 deg, laid along the cradle | 1 | 223 | 1/1 | 9 | - | orientation-aware place (object target pose from the region frame), grasp chosen for the release pose |

## Since the survey (2026-09-23, evening)

- **Push (goal 5): 3 -> 50 of 50.** Skills.push does what the human demos do: jaws open 62 mm
  along the motion, fingertips on the plate's floor 0.44 of its radius ahead of its centre, the
  hand advancing at 6 cm/s. Three settings of the fingertip height were measured before one moved
  the plate; an aim tied to the plate's position stalled the push.
- **The rack (goal 9): 1 -> 50 of 50** with the commanded acceleration bounded at 0.5 m/s^2 while
  the jaws hold something (2 otherwise). Bounding every period at 1.0 or 0.5 overran the approach's
  phase switches and cost libero_spatial 31 episodes; a braking cap in the teacher did not help.
- **The drawer hook (goal 3).** With the humans' measured offsets (42 mm on the opening side of the
  bar, 17 mm above it, jaws wide, pointing down) the top drawer opens in 6 of 6 at a median step
  55, against ~290 for relocation plus the pinch drive. Opening first and not relocating then
  fails differently: the open drawer overhangs the bowl, the grasp screen finds no top-down grasp
  and falls to side grasps whose descent pushes the drawer shut. The humans find a top-down rim
  pinch there; the teacher does not yet.
- **Mid-air drops are open.** They happen in the carry phase. They nearly vanish with 0.5 m/s^2 in
  every period, but not with 0.5 only while holding, nor with a slow final descent, nor with the
  tool held still while the jaws close.
- **Evaluate within LIBERO's limits.** With the hook (runs/skill_v5): 1433/1500 at 600 steps (goal 3
  9 -> 18), 1258/1500 within 220/280/300. Goal 3 is 0 within 300 in every version so far; 15 tasks
  are under 48/50 within their limit, nearly all on time: spatial 7 12, spatial 9 24, spatial 6 27,
  spatial 0 33, spatial 1 34, goal 6 34, goal 4 35, spatial 4 39, object 1 43, spatial 8 44, goal 5
  45, spatial 3 46, goal 1 and 8 47. Humans move a held object at 0.17-0.23 m/s (median) and lift it
  120-160 mm; the teacher's caps are 0.12-0.15 m/s and it lifts ~270 mm near tall fixtures.

## What is missing, by effect

1. Carry without drops (gated acceleration bound; sweep running) -- spatial 1-9, goal 1/3/4/8.
2. Articulation by hooking with open jaws, top drawer from above, and no relocation before opening
   -- goal 3 (and time on every drawer task).
3. Push skill, caging a dish with open jaws -- goal 5.
4. Orientation-aware place: the object's target pose read from the region frame, the grasp chosen
   by the release pose -- goal 9.
5. Top-down grasp on the cream cheese box -- object 1.
6. Speed: a drop into containers (object suite, goal 6), lower release on surfaces, fewer phases.
   Needed for the eval budgets, not for the teacher's own success.
7. Markov labels: replace the per-episode caches (grasp, handle frame, clearing spot) with
   state-derived choices with hysteresis.
