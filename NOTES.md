# Working notes

Findings that cost real time to establish, and which are not recoverable from
the code alone. Written for whoever picks this up next, including us.

## Status

`libero_goal/0` — "open the middle drawer of the cabinet" — **solved, 3/3
deterministic runs**, verified by LIBERO's own `_check_success()`, drawer opened
142.5 mm (qpos `+0.0006 → -0.1419`), episode terminated by the benchmark at step
520. `tests/test_drawer_open.py`.

Getting there needed four grasp filters and three separate determinism fixes.
Both lists are below, in the order of how much they actually mattered rather
than the order they were found.

---

## 1. The determinism trap (read this before trusting any measurement)

`suite.get_task_init_states(task_id)` loads a **torch pickle**, so on torch >=
2.6 it raises `UnpicklingError: Weights only load failed` — the same wall the
Contact-GraspNet checkpoint hits. If that exception is swallowed, LIBERO falls
back to **re-sampling object placement on every reset**, and nothing warns you.

The cost of learning this: an afternoon of attributing scene randomisation to
the grasp network. Three "repeat" runs of the same task put the drawer handle
12 mm apart and produced 28 / 11 / 2 grasp candidates with approach alignments
0.97 / 0.84 / 0.49. That looks exactly like a flaky model. It was a flaky
scene.

Fixed in `environment_libero.py` by calling `grasp_cgn.allow_numpy_unpickling`
before the load, and by making the fallback **print a warning** instead of
degrading quietly.

**Rule adopted:** a fallback that silently changes experimental conditions is
worse than a crash. Warn loudly or fail.

### It took two fixes, and the first one looked like it worked

Pinning the init state was necessary but **not sufficient**, and the partial fix
was the dangerous part: handle drift fell from ~12 mm to ~2 mm, which reads as
"deterministic now" if you only glance at it.

`set_init_state` restores `qpos` — the robot and every **free-jointed** object.
A cabinet base is a **fixed body**: its pose lives in the model, not the state,
and robosuite's placement sampler randomises it during `reset()` via
`np.random`. No amount of state restoration moves it.

So determinism needs **both**:

```python
np.random.seed(self.reset_seed)   # fixture placement, chosen during reset()
self._last_obs = self.env.reset()
self._apply_init_state()          # qpos: robot + free-jointed objects
```

Order matters — seeding after `reset()` is useless.

**Second rule:** when a fix reduces an error instead of eliminating it, that is
evidence the mechanism is only partly understood. Chase the residual.

### And there was a third one

Even with the scene byte-identical — same handle pose, same wrist pose to the
millimetre — the candidate count still moved between 8 and 13. Contact-GraspNet
samples internally (`forward_passes=3`), and `_candidates` seeded `np.random`
but not `torch`. Seeding both makes the whole pipeline reproduce exactly.

Three determinism bugs, each only visible once the previous was fixed, and every
one of them presenting as "the grasp network is flaky." The lesson is not about
any of the three individually — it is that **"flaky model" is a hypothesis of
last resort**, and it should not be entertained until reproducibility has been
demonstrated rather than assumed.

> There is a **fourth**, and it only appears when the ROBOT changes: robosuite
> draws `randn(n_arm_joints)` of initialization noise even at zero magnitude,
> so a 6-DOF arm leaves the RNG one draw ahead of a 7-DOF one and every fixture
> moves ~7 mm. Seeding before `reset()` does not cover it. See §7.

### Which measurements survive

Trustworthy — geometric, or taken within a single process and scene:

| finding | why it holds |
|---|---|
| CGN proposes on 5/5 LIBERO objects, PCA on 1/5 | one env, objects looped without reset |
| objects fall 60–72 mm in 5 sim steps after reset | within one episode; physics is deterministic |
| `geom_rbound` spheres swallow the table (517/1541 bowl points) | pure geometry |
| `gripper_closing_axis_idx` flips onto the approach axis when closing | geometry; spread is `[0.0001, -0.0663, -0.0172]`, Y leads Z by only 3.9x |
| drawer face at grazing incidence yields 0 grasps; front views give 14–23 mm | four free cameras, same scene, one process |
| handle keypoint binds to `cabinet_middle` and tracks with 0.0 mm error | single run, exact |
| ideal scripted grasp opens the drawer (qpos 0 → −0.131) | existence proof, one run |

Not trustworthy — compared across processes, so across scenes:

- any physical grasp pass/fail *rate* (the 2/3 and 1/5 numbers)
- the 28 / 11 / 2 candidate counts
- `akita_black_bowl_1` passing one run and failing the next

Re-measure these now that init states load.

---

## 2. Bug classes found, in the order they hid behind each other

Each of these masked the next, which is why the drawer took so long.

1. **Stale scene** — grasps computed from the reset-time cloud, 6–7 cm above
   where objects settle.
2. **Contaminated clouds** — `geom_rbound` is the *circumscribed* radius, so
   once an object rests on the table its sphere includes the tabletop. Fixed by
   testing the exact collision **box** decomposition (every LIBERO object is one
   visual mesh + N boxes).
3. **Duplicated predicate** — `tests/test_grasp_proposer.py` carried its own
   copy of the old sphere test, so it kept measuring contaminated clouds after
   the env was fixed. One definition, in the env.
4. **Wrong viewpoint** — agentview sits at `(0.659, 0, 1.610)` looking down
   −X/−Z; any surface facing sideways is seen edge-on. Contact-GraspNet returns
   **zero** grasps for a drawer handle from there and 14–23 mm from the front.
   This was mistaken for the 2021 model being too old. It is not: the model is
   fine, the camera was pointed wrong.
5. **Fixtures unregistrable** — `_object_poses()` only read the observation
   dict, which LIBERO does not populate for fixtures. A drawer-handle keypoint
   bound to the nearest *movable* object (`plate_1`) and then tracked the plate.
6. **Binding by body origin** — a cabinet's three drawer bodies share nearly the
   same origin, so the handle bound to `cabinet_top` and stayed put while the
   real handle moved 120 mm. Bind by distance to geom *surface*.
7. **Non-deterministic scenes** — §1.

---

## 3. Grasping

**The model is Contact-GraspNet**, shared with `wardmate_ws/src/llm_robot_control`
(weights at `~/contact_graspnet_pytorch`). That project's ROS package is *named*
`graspgen` but wraps Contact-GraspNet. It is **not** NVIDIA GraspGen, which
pins `torch==2.1.0` and has no cp312 aarch64 wheel. This naming collision has
already caused one wrong conclusion — that grasping was blocked on this hardware
when a working install was sitting in the next repo.

**Selection follows the ReKep paper**, which uses AnyGrasp purely as a detector:

> "we always return the grasp closest to a specified 'grasp keypoint'"

so we rank by distance to the keypoint, not by network confidence. Confidence
ranking put a cabinet-edge grasp first while a nearer candidate went unused.

**Actionability is ours, not the paper's.** Contact-GraspNet scores whether a
grasp would *hold*, not whether it would *work* — the gap AO-Grasp names. Two
filters in `grasp_cgn.actionable()`:

- *view alignment* (always): the approach must point roughly along the view
  direction, since that is the only side of the surface with free space.
  Rejected candidates had `approach · (−Y) = −1.00` — approaching the drawer
  from inside the cabinet outward, not executable at all, and ranked first.
- *action axis* (optional): when the task defines a travel direction. Measured
  on the drawer: alignment 0.49 stalled, 0.84 slipped after 2.3 mm, 0.97 pulled
  125 mm.
- *jaw occupancy*: how many target points fall inside the volume the fingers
  will close on. See below — this is the one that actually decides.

### Distance to the keypoint is an isotropic metric, and objects are not

The most instructive failure of the day. With the scene finally deterministic,
three runs produced grasps that scored *well* on everything measurable —
13.8–14.3 mm from the handle, approach alignment +0.95 to +0.99 — and all three
closed on air. The lucky run that opened the drawer 139.8 mm had scored *worse*
by that metric, at 22.5 mm.

Decomposed against a handle bar ~65 mm long and a few mm thick:

| | Δ along bar | Δ across | Δ **vertical** | result |
|---|---|---|---|---|
| opened 139.8 mm | +22.5 mm | +15.0 mm | **+1.7 mm** | held |
| closed on air (x3) | −5 mm | +11 mm | **+7 mm** | missed |

22 mm of error *along* the bar is free. 7 mm *above* it is fatal. Ranking by
Euclidean distance to a keypoint cannot tell those apart, and confidently
prefers the worse one.

The fix is not a better distance metric — it is to stop using distance as the
decision. `jaw_occupancy()` counts target points inside the jaw volume, which is
a direct measure of "will this hold something" rather than a proxy for it.
Distance remains useful only as a tiebreak.

**Generalises:** any thin or elongated feature — a handle, a rim, a mug lip, a
cloth edge — has directions where error is free and directions where it is
fatal. A scalar distance flattens that away.

---

## 4. PointWorld (`third_party/PointWorld`, cloned not installed)

Apache-2.0, commit `05484826`. Checkpoints at `nvidia/PointWorld_models`.

**It cannot run on this box as released, and the reason is worth knowing: our
GPU is newer than their toolchain.** GB10 is `sm_121`; PointWorld pins
`torch==2.5.1 + cu124`, and CUDA 12.4 predates Blackwell. `torch-scatter`,
`spconv-cu124` and `flash-attn==2.7.4` are all pinned to that toolchain with no
aarch64/CUDA-13 wheels, and `ptv3/ptv3.py:22` imports
`flash_attn_varlen_qkvpacked_func` at module scope with **no fallback** — so
flash-attn is mandatory, not optional.

Validate on rented x86_64 before committing to a port. The port is real work
(spconv and flash-attn are the risky steps) and should only be paid for after
the model is known to work on our data.

### Why it is still the right direction

ReKep **structurally cannot** do deformables, and this is visible in our own
code rather than inferred. `_register_keypoints` stores every keypoint as an
offset in some object's **rigid transform**:

```python
world2obj = T.pose2mat([pos, quat])
local = inv(world2obj) @ np.append(kp, 1.0)
```

A folding cloth has no rigid pose, so every keypoint on it tracks wrongly — the
same failure as the drawer handle being dragged around by `plate_1`, but
unfixable, because there is no correct body to bind to. Point flow has no such
assumption: per-point displacement *is* deformation.

### The caveat behind "no demonstrations"

It means no *training* per task, not no *human input* per task. The goal is
**pointwise 3D target positions**, specified either by a human drawing a mask in
a GUI or by a VLM. For autonomy that specification has to come from somewhere —
and our own VLM grounding failed on exactly this a day in (Claude selected a
keypoint 153 mm from any drawer handle, because the proposer never put one
there).

That seam is also the opportunity: **annotation can be injected here.** Existing
segmentation (`yolo_seg_node`), our own detections, or ward-item annotations can
supply the mask and target points in place of the GUI.

Paper's hardware is one RealSense D435 + FoundationStereo depth, 1.5 cm grid
downsampling — same camera class as the wardmate robot.

### Licensing is NOT simply Apache-2.0 (corrects an earlier read)

The top-level `LICENSE` is Apache-2.0 and permits commercial use, but that
covers PointWorld's own code only. The model **cannot run** without DINOv3,
which supplies `scene_features` (a DINOv3 ViT-L16 encoder projected onto the 3D
points, `scene_featurizer.py:22`), and DINOv3 ships under the **DINOv3
License**, not Apache-2.0. Also vendored: PTv3 (MIT), Sonata (Apache-2.0),
OmniGibson transforms (MIT), deoxys_control (Apache-2.0).

**DINOv3 weights are ACCESS-GATED.** The README instructs you to request access
on the DINOv3 release page and download from a URL sent by email; there is no
public wget. So the critical path for any real evaluation runs through a human
requesting access, not through the port.

Read `third_party/dinov3/LICENSE.md` before committing commercially — the
constraint we care about lives there, not in the Apache-2.0 file.

### The measurement, finished

> **SUPERSEDED 2026-08-03.** These numbers are `small-droid` +
> `domains=droid` + 16/31 channels, i.e. the smallest checkpoint, the
> real-robot domain, and the single-arm layout, applied to simulation. The
> guide's own recipe for simulation is `large-droid+behavior` with
> `--domains=behavior --norm_stats_path=stats/droid_behavior`. Re-measured
> there: **3.92 mm** on the moved points, `cos = 0.999`, and no magnitude
> under-prediction. See `HANDOFF.md`. Do not quote the table below.

Driving `BaseModel` (not `DynamicsPredictor`) with DINOv3 loaded from
`model-best.pt`, on `libero_goal/0`, drawer pulled 130 mm over 10 steps,
2 cameras, 2789 scene points of which 73 move:

| | all pts | moved pts |
|---|---|---|
| PointWorld | 14.9 mm | **58.5 ± 1.2 mm** |
| predict-no-motion | 2.2 mm | 84.5 mm |

It **beats** the static baseline where the motion is, and **loses** on the
scene as a whole. Direction is essentially right — `cos(pred, true) = 0.94` on
the moved points — but magnitude is under-predicted 3.6x (36 mm predicted
against 130 mm true), and it leaks 14.6 mm of spurious motion onto a static
tabletop.

**There is an error floor of roughly 10–15 mm, independent of the true
motion.** Re-recording the same pull at 3.2 mm/step instead of 13 mm/step
(total motion 32 mm instead of 130 mm) inverts the verdict: 19.4 mm vs a
17.6 mm static baseline, `cos` down to 0.41. The model is not resolving
millimetres. That is the paper's own stated "fine-scale objects" limitation,
now with a number on it, and it is why a 26 mm drawer handle is the wrong
place to judge this model — see §4's note on which claims need *different*
episodes.

### The bridge, and the run-to-run noise it exposed

`scripts/run_bridge_test.sh` brings the service up in `.venv-pw`, runs the
round-trip test in `.venv`, tears it down. Measured, GB10, 2789 scene points:

| | cost |
|---|---|
| `observe`, cold | 926 ms |
| `observe`, warm | **133 ms** round trip (115 ms encode + 10 ms assemble) |
| `rollout`, K=1 | **46 ms** round trip (34 ms model + 10 ms assemble) |
| `rollout`, K=20 | 488 ms, 24 ms/candidate end to end |

The cold/warm gap is cuDNN algorithm selection and kernel autotune. A loop pays
it once; quoting 926 ms as the per-observation cost would overstate the budget
7x. **Warm the service before timing anything.**

Batching gives only **1.9x** over serial, not the order of magnitude one might
hope for. Assembly is a real share of the cost: `gather_features` recomputes
`dist2robot` per candidate on the CPU with a cKDTree, once per timestep. That
is the thing to move to the GPU if throughput matters.

**The model scatters ~9 mm run to run on IDENTICAL input.** Eight repeats of
one candidate, each in its own call, spread 8.9 mm. This is not just
`shuffle_orders`: spconv and scatter reduce with atomics, whose summation
order is not fixed. It matters for planning, because differences we care
about — `+Y @ 13 mm` at 93 mm against the recorded action at 94 — are INSIDE
that noise. A planner must seed, or average repeats, or it will pick between
indistinguishable candidates at random.

**Batching is safe, and the check that proved it is the interesting part.**
Eight identical candidates in one batch spread 3.9 mm — alarming until
compared against the right control. PTv3 serializes a whole batch into one
space-filling curve and cuts it into attention patches, so a patch straddling
a batch boundary would let candidates contaminate each other, which would look
like a bad model forever. But identical inputs do not reproduce even when run
apart, so "identical in, identical out" was never the right test. Against the
8.9 mm serial baseline, the batched 3.9 mm is LESS scatter, not more.

**Rule, again:** before calling a difference a defect, measure what the same
computation does when you change nothing. Non-determinism sets the floor for
what any comparison can resolve.

### It can RANK actions, which is the property a planner needs

Accuracy and usefulness are different questions. A planner never needs the
predicted displacement to be right; it needs the ORDERING over candidate
actions to be right. `tests/rank_actions_pointworld.py` keeps the recorded
scene, replaces the action with rigid counterfactual gripper trajectories, and
scores each by an MPPI-style cost — distance from the PREDICTED points to
where the real episode actually put them.

**Direction is ranked correctly.** All three `+Y` candidates (the way the
drawer opens) beat all sixteen others. `-Y`, into the cabinet, is worst, and
worse the faster you push: 177 / 220 / 299 mm at 6.5 / 13 / 26 mm per step.
`still` sits mid-table.

> **RETRACTED:** this section originally concluded "the model understands the
> drawer's constraint." It does not, and this experiment could never have
> shown that it did. Against a goal 160 mm away in +Y, a model believing only
> that *a grasped object follows the hand* produces the same table: `-Y @ 26`
> predicts the object at -260 mm, giving an error of |-260 - 160| = 420 mm
> (observed 335), while `+Y @ 26` predicts +260 for |260 - 160| = 100
> (observed 62). The ordering follows from the goal's direction, not from any
> knowledge of the joint. See "the axis is NOT discoverable" below, which
> tests the two hypotheses apart and finds against the model.
>
> The ranking result itself stands -- the cost landscape IS well shaped and a
> planner CAN descend it. Only the explanation was wrong.

**Magnitude is monotonic but biased, in exactly the direction the 3.6x
under-prediction implies.** Cost by rate: 6.5 → 116 mm, 13 → 95, 26 → 69. The
argmin is 26 mm/step against a true 13 — a **2.0x overshoot**. A model that
under-predicts how far things move must ask for more travel than the goal
needs. That is survivable: a receding-horizon loop re-observes each step and
corrects. Open-loop execution would overshoot.

> **RE-RUN on `large-droid+behavior` (2026-08-03): the overshoot is GONE.**
> Cost by rate 6.5 → 66 mm, 13 → **3.4 ± 0.5**, 26 → 125. The argmin is
> 13 mm/step against a true 13 — **1.0x, no bias to correct**. Direction still
> ranks correctly (top 3 all `+Y`, `-Y @ 26` worst).
>
> This was the ONLY evidence for "open-loop execution would overshoot 2x",
> which was the stated reason the loop had to be receding-horizon. That
> argument is withdrawn. Receding-horizon is still the right shape — for
> disturbance, contact and re-grounding — but this test may not be cited for
> it, and `rank_actions_pointworld.py` no longer prints the overshoot verdict
> unconditionally.
>
> Run-to-run scatter fell with it: 8 identical candidates spread **1.94 mm**
> apart and **0.46 mm** in one batch, against small-droid's 8.9 / 3.9. The
> best-to-second gap is 4.2 mm, 9x the batched noise, so "a planner must seed
> or average" is no longer load-bearing when candidates are scored in a batch.

So the cost landscape is well shaped, and a planner built on this model can
descend it. That is the finding that makes the closed loop worth building.

**The control that caught the bug.** A rigid `+Y` at the true rate must land
near the recorded action, and the first run said 199 mm against the recorded
97 mm. The counterfactuals were built from the CENTRED robot points and then
passed to `build_data_dict`, which runs `center_shift` again — the gripper
ended up ~1 m from the scene, so every candidate was equally useless and the
table read exactly like "the model cannot rank actions". With the control in
place the gap is 1.3 mm.

**Rule:** a counterfactual needs a control that reproduces the factual. Without
one, a broken candidate generator is indistinguishable from a bad model — and
it produces a confident, plausible, completely wrong verdict.

### The axis is NOT discoverable — the model largely thinks grasped things follow the hand

> **PARTLY RETRACTED 2026-08-03, then RE-RUN the same day — the headline
> conclusion HOLDS, the inversion does not.** The numbers below are
> `small-droid` with `domains=droid` and the 16/31 single-arm layout, on
> simulation data. Re-run on `large-droid+behavior` with `domains=behavior`
> and 17/42 (`scripts/run_axis_discovery.sh`):
>
> | | small-droid | large-droid+behavior |
> |---|---|---|
> | free `+Y` | 71.5 mm | 195.7 mm |
> | blocked `-Y` | 153.2 mm (2.14x free) | 192.3 mm (**0.98x free**) |
> | across `+Z` | 66.0 mm | 192.9 mm |
> | spread, all 18 dirs | — | 184.4 – 199.3 mm (7%) |
> | rank corr. vs sim | **-0.50** | **+0.50** |
> | argmax vs true axis | 135 deg | 90 deg |
>
> **The model is FLAT, not inverted.** Against 200 mm of hand travel it
> predicts 184–199 mm of target motion in every direction — 0.92 to 1.00x, in
> all 18. So "grasped things follow the hand" is not the fallback explanation
> here, it is the measured one, and the axis is still not recoverable this
> way. What was an artefact is only the claim that the ordering was INVERTED
> and that the model preferred the blocked direction.
>
> Consequence 2 below therefore stands: a follow-the-hand model plus a correct
> goal is sufficient to open that drawer, so the drawer success does not
> demonstrate that PointWorld understands drawers.
>
> **Test flaw, unfixed:** the model is asked over `T_LEN = 11` (200 mm of hand
> travel) while the simulator is probed over `PROBE_STEPS = 4` (80 mm), so the
> two columns are not like for like in absolute mm. The discrimination result
> is unaffected — it compares directions at one horizon — but do not read
> "says 195.7 / moved 64.7" as an accuracy figure.

`tests/discover_axis_pointworld.py`. The question mattered because
`model.jnt_axis` is privileged simulator knowledge: if the world model could
supply the constraint direction, a real robot would not need it.

The test has to be **direction-agnostic**, which is the whole trick. Scoring
candidates against a goal proves nothing, because a follow-the-hand model
already ranks the goal's direction first. So instead: set the goal to "stay
exactly where you are", making the returned cost the PREDICTED DISPLACEMENT of
the target's points regardless of where they go. A model that knows the joint
predicts large motion along the slide and little across it; a model that only
knows "grasped things follow the hand" predicts the same everywhere.

18 directions probed with the gripper holding the closed drawer:

| | PointWorld predicts | simulator actually does |
|---|---|---|
| free, `+Y` | 71.5 mm | **64.9 mm** |
| blocked, `-Y` (into the closed drawer) | **153.2 mm** | ~0 |
| across, `+Z` | 66.0 mm | 13.7 mm |

**Rank correlation with the simulator: -0.50.** The model's argmax is
`[+0.71, -0.71, 0]`, **135 degrees from the true axis**, and it predicts
2.14x more motion into the blocked direction than along the free one. It is
not merely ignorant of the constraint; its ordering is inverted.

Two consequences, and the second is the uncomfortable one:

1. **The axis is not recoverable this way.** Articulation must come from
   privileged knowledge or from perception, not from rolling out candidates.
2. **The drawer-opening success proves less than it appeared to.** A
   follow-the-hand model plus a correct goal is *sufficient* to drive that
   task, so solving it does not demonstrate that PointWorld understands
   drawers. The planner works; the model's physical understanding is not what
   makes it work.

**Rule:** when a result is consistent with both the hypothesis you like and a
much dumber one, it is evidence for neither. Design the measurement so the two
predict different things -- here, that meant removing the goal direction from
the metric entirely.

The control also matters: the simulator was executed from a saved state for
the deciding directions, so "blocked" is measured rather than assumed.

### Five input defects, each of which shrank the error

Found in this order, each only visible once the previous was fixed. Same
pattern as the determinism bugs in §1.

| fix | all pts | moved pts |
|---|---|---|
| hand-assembled pieces, zeroed robot features | *invalid* | *invalid* |
| drive `BaseModel`; DINOv3 from the checkpoint | 30.5 | 87.2 |
| gripper points corresponded across time | — | — |
| gripper sampled on real meshes, not `geom_rbound` | 22.7 | 68.0 |
| robot arm removed from the scene cloud | 14.3 | 58.5 |
| second camera (training used exactly two) | 14.9 | 58.4 |

The last one is the instructive one: it did **not** move the mean error, and
moved `cos(pred, true)` from 0.725 to 0.942. A single scalar would have
reported that change as nothing.

1. **`geom_rbound` again.** Sampling the gripper inside each geom's bounding
   sphere turned a 63 x 93 x 206 mm Franka hand into a **234 mm blob** — the
   hand mesh's rbound is 119.6 mm. The robot point cloud IS the action, so
   this misstated the action, `dist2robot` for every scene point, and the
   model's idea of what was about to be touched. Fixed by area-weighted
   sampling on the actual mesh triangles, which is also literally what
   PointWorld does off the URDF. Third time this circumscribed-radius trap has
   cost real time (§2).
2. **The arm was in the scene cloud** — 642 of 3372 points, 19%. PointWorld
   represents the robot once, as `robot_flows`; leaving the same surface in
   the scene gives those points two contradictory roles. It also corrupted our
   ground truth, which binds points to task objects and therefore scored the
   arm — really moving 130 mm — as static. Removing it needs an EXACT mask:
   `_robot_pixel_mask`'s bounding spheres would have deleted the drawer handle
   the gripper was holding. `env.robot_geom_mask()` renders geom ids instead.
3. **Resampling the gripper points every step** made `robot_velocity` and
   `robot_acceleration` — 6 of the 16 robot channels — pure sampling noise.
4. **One camera instead of two.** `min_num_cameras = max_num_cameras = 2` in
   the training args; every training sample was a two-view fusion.
5. Zeroed `robot_feat`, missing `time_embed`, missing `scene_encoder_norm` —
   all dissolved by driving `BaseModel`.

### PTv3 is NOT deterministic in eval mode

`build_ptv3` hardcodes `shuffle_orders=True` (`base.py:198`), which reaches
`torch.randperm` in `Point.serialization` (`ptv3/structure.py:98`) on every
forward pass, permuting which space-filling-curve order each attention block
sees. `model.eval()` does not touch it — it is a training augmentation left on
at inference.

Two back-to-back runs of the harness differed by **5 mm** on the moved points,
which is the same size as the effects being measured. `run_pointworld_on_episode.py`
now seeds every forward and reports the spread over 5 seeds. Do not quote a
single-pass number from this model.

Fourth instance of the §1 rule, and the first one that was not our bug.

### Verified in-distribution, so not the explanation

Measured rather than assumed, before concluding anything about the model:

- DINOv3 sees the scene: **98.6% of points visible**, projected depth agrees
  with the depth image to **0.11 mm max** (the featurizer's tolerance is 3 mm).
- Backbone features std **1.55**, no all-zero rows.
- Every one of the 31 scene and 16 robot raw channels lands within ~1 sigma of
  DROID's own normalization stats.

`gripper_open` polarity is not documented in the release. Measured: our reading
(1 = open, so 0 while grasping) scores 58.5 mm against 65.9 mm for the
inverse, on a seed spread of 1.2 mm. Weak but real evidence we have it right.

Still deviating from training conditions: images are 256x256 where the release
contract asserts 180x320, and LIBERO's agentview FOV is not DROID's.

### State of the PointWorld evaluation (superseded above; kept for the reasoning)

Working: port (5 patches, `scripts/setup_pointworld.sh`), checkpoint loads
464/464 tensors, **47 ms** per 10-step chunk on the GB10 (2.13x the paper's
0.1 s), MuJoCo episode recorder with exact ground truth
(`scripts/record_pointworld_episode.py`), and a scoring harness
(`tests/run_pointworld_on_episode.py`).

NOT yet valid: the first end-to-end run reported **553 mm** mean error against a
**1.8 mm** static baseline. That is physically impossible on a 0.8 m scene and
is an INTEGRATION BUG, not a result. Do not quote it.

Cause, confirmed at `pointworld/base.py:478-484`: `BaseModel.forward` passes
`normalize_fn=self.normalize` / `unnormalize_fn=self.unnormalize` and then
`pred = self.unnormalize(pred_norm)`. The harness calls the inner
`DynamicsPredictor` directly with both as `None`, so raw metres are fed to a
model trained on normalized coordinates and its normalized output is read as
metres. Calling `DynamicsPredictor` directly also skips `robot_proj`,
`robot_type_emb` and `time_embed`.

FIX: drive `BaseModel` instead of `DynamicsPredictor`. The norm buffers are in
the checkpoint (`scene_norm_mean/var`, `robot_norm_mean/var`,
`norm_stats_per_step_mean/var`).

### The checkpoint already contains DINOv3

`scene_feature_encoder.scene_encoder.dinov3.*` — 24 blocks, `storage_tokens
(1,4,1024)` — in Meta's ORIGINAL naming. So the HuggingFace download and the
`src/pointworld_bridge/dinov3_hf.py` adapter were never necessary; the exact
weights used in training ship inside `model-best.pt`, and the DINOv3 submodule
provides Meta's own code to load them. Prefer that over the HF port.

The adapter is still correct and worth keeping for reference, and it did earn
its keep once: it surfaced that `get_intermediate_layers` defaults to
`norm=True` and applies the final LayerNorm, while HF's `hidden_states` are raw
block outputs. Measured difference: **std 373.9 raw vs 1.688 normed**, a 220x
scale error that would have loaded cleanly and produced confident nonsense.

### Model interface, for whoever wires this up

`pointworld/base.py:434` consumes:

| key | shape | meaning |
|---|---|---|
| `scene_flows` | (B, T, Ns, 3) | `[:, 0]` is the initial scene point cloud |
| `scene_features` | (B, T, Ns, Ds) | DINOv3 features projected to points |
| `robot_flows` | (B, T, Nr, 3) | **the action** — robot points over time, from URDF forward kinematics |
| `robot_features`, `*_exists` | | masks and per-point features |
| `__domain__` | list[str] | domain tag per batch element |

The action being "where the robot's own points are" is the whole
cross-embodiment mechanism, and it means **our MuJoCo sim can generate these
triples directly** — scene points from depth, gripper points from MuJoCo FK, and
the ground-truth resulting scene flow to score against. No real robot and no
DROID download needed to test whether its dynamics predictions are sane, once
DINOv3 is available for the features.

## 5. Environment gotchas

- `MUJOCO_GL=egl` is required. `pyrender` pins `PyOpenGL==3.1.0`, which **lacks
  `EGL.EGLDeviceEXT`** and breaks every headless render; force 3.1.10 back and
  accept the pip warning. pyrender itself imports fine on 3.1.10.
- The EGL errors printed at interpreter shutdown are benign `__del__` noise.
- Pulling a joint with `damping=50` needs `precise=True`; coarse mode allows 30
  OSC iterations against a 0.02 m tolerance and the wrist simply stalls.
- A drawer's qpos trails the wrist by ~18% through slip, so commanding exactly
  0.16 m stops short of LIBERO's success threshold.
- `libero_goal/0` opens by travelling **+Y** even though its joint qpos goes
  **negative** (range `[-0.16, 0.01]`); `_check_success()` is True at −0.16.
- **Two PointWorld artefacts were deleted to reclaim 14.2 GB** (28 G → 15 G),
  and neither is a loss:
  - `pretrained_checkpoints/large-droid/` (13 G) — **zero references** in
    `src/`, `tests/`, `scripts/` or the docs. Every measurement here uses
    `large-droid+behavior` or `small-droid`, both kept.
  - `third_party/dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-*.pth`
    (1.2 G) — a GENERATED file, not a download. It is the output of
    `scripts/extract_dinov3_from_checkpoint.py`, which pulls the weights out of
    `small-droid/model-best.pt`. `pointworld_bridge/model.py:85` already raises
    with that command when it is missing, so PointWorld reports the fix itself.
  Not deleted, deliberately: `spconv/spconv/build` (4.7 G) holds JIT-compiled
  kernels and is a slow rebuild for modest space.

---

## 6. The language layer: what PointWorld does not do

Read the paper before assuming this one. `NOTES.md` §4 and `HANDOFF.md` carry
the measurements; this is the architecture.

> **DESIGN BASIS (2026-08-03): `docs/pointworld_pipeline_implementation_prereqs.md`,
> NOT KUDA.** The guide layers it as coarse DINOv3 grounding → an object-level
> **scene graph** (`object_id`, `label`, `centroid_xyz`, `bbox_extent_xyz`,
> `relations`) → an LLM **goal formalizer** emitting structured 3D goals rather
> than actions → MPPI over PointWorld (`pytorch_mppi` first) → event-driven
> re-grounding after grasp, release, occlusion or repeated planning failure.
> The KUDA material below is retained as evidence for *why* the split is drawn
> where it is — the ±150 mm / ±18 mm / 10–15 mm / 7 mm table is the argument,
> and it is independent of whose prompt is used — but KUDA's keypoint-offset
> prompt is no longer the target design, and `keypoint_grounding.py` should be
> re-read against the guide's scene-graph schema before more is built on it.

**PointWorld is language-free by construction.** No text encoder, no
cross-attention to embeddings; `BaseModel.forward` consumes points, features
and a domain tag. Confirmed in code and by searching the paper for every
mention of "language", "text", "instruction" and "VLM".

The single mention is a possibility the authors did not build:

> "task-relevant points can be specified by either human via GUI or by VLMs"

and Appendix A.6.2 says what actually ran for every reported success rate:

> "we specify tasks through a GUI tool that allows users to select object
> masks using SAM2 and specify target positions in the world frame"

VLM specification appears again only in the limitations, as future work. So
any language capability is an EXTERNAL wrapper, and its interface is already
pinned: **mask + world-frame target positions**.

### Why that split is the right one, in numbers from this repo

| | spatial resolution | source |
|---|---|---|
| VLM keypoint grounding | ±150 mm | our own failure: keypoint 153 mm from any handle (§4) |
| VLM constraint output | ±18 mm | ReKep's grasp target 17.7 mm off a 21.7 mm cube |
| PointWorld | 10–15 mm floor | measured, §4 |
| grasp success margin | 7 mm | §3 |

A VLM is accurate enough to say WHAT should move and roughly where, and about
ten times too coarse to CONTROL. KUDA's prompt exploits exactly that by asking
for targets as OFFSETS FROM OTHER KEYPOINTS rather than absolute coordinates,
so the VLM never has to name a position it cannot resolve.

### KUDA is the precedent (`third_party/KUDA`)

Same architecture, weaker dynamics model, 80% over 60 trials with free-form
language including deformables. Its prompt is reusable as-is. The one thing we
get for free that most implementations must build: lifting a 2D mask to 3D
point indices, because `scene_featurizer.py` already projects every scene
point into both camera images with visibility and depth checks.

**Do not confuse the layers.** DINOv3 perceives the SCENE — the ablation in
`run_pointworld_on_episode.py --ablate dinov3` shows it is what keeps the
static scene static (spurious motion 14.6 mm with it, 131 mm without). It does
not perceive the TASK, and no amount of feature quality makes it do so.

---

## 7. A second embodiment: the UR5e on LIBERO

Standing up for the GAM-keypose + cuRobo cross-embodiment experiment. Nothing
about GAM or cuRobo is in the repo yet; this section is only the SCENE, and the
scene is finished and verified:

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_ur5e_scene.py
# 16 bodies compared, 0.0000 mm / 0.0000 deg, robot base offset 0.0000 mm
```

Our code, as always, is in `src/` — `robots_ur5e.py`, `init_state.py`,
`fixtures.py`, plus a `gripper=` parameter and two hooks in
`environment_libero.py`. **`third_party/LIBERO` stays a pristine vendored
copy**; a UR5e needs no edit inside it.

### `robots=["UR5e"]` cannot work, and the error does not say why

    KeyError: 'MountedUR5e'

LIBERO's problem classes prefix every robot name with `Mounted`
(`libero_tabletop_manipulation.py:23` and its kitchen/study siblings), then
resolve the result through **LIBERO's own registry**
(`libero/libero/envs/robots/__init__.py`), which contains exactly two models,
both Pandas. robosuite ships a working `UR5e` and LIBERO never looks at it.

robosuite splits registration in two and both halves are needed: the MODEL
registers itself from the class body via `RobotModelMeta`, the CONTROL class
must be added to `robosuite.robots.ROBOT_CLASS_MAPPING` by hand.

One trap inside the trap: stock `UR5e` declares the same `table` base offset
the Panda does, but **omits the `kitchen_table` and `study_table` keys**
LIBERO's kitchen and study arenas index (`bddl_base_domain.py:317,357`). Those
suites would `KeyError` on reset. `MountedUR5e` copies `MountedPanda`'s offset
dict verbatim — which also pins the two bases to the same world position, so
the comparison varies arm kinematics and nothing else. Those numbers are load
bearing; do not tidy them.

### Init states are Panda-shaped, and `set_init_state` reads positionally

LIBERO's 50 pinned states are flattened MuJoCo states, `[time, qpos, qvel]`,
recorded against the Panda model. On `libero_goal/0`:

| | Panda | UR5e + PandaGripper |
|---|---|---|
| `nq` / `nv` | 41 / 37 | 40 / 36 |
| robot block | 9 joints (7 arm + 2 finger) | 8 joints (6 arm + 2 finger) |
| first object `qpos` address | 9 | **8** |
| object block | 8 joints / 32 `qpos` | 8 joints / 32 `qpos` |

`set_init_state` is `sim.set_state_from_flattened`, which reads by position.
Feed it the Panda array on any other arm and it either raises on the length or
— the dangerous case — writes drawer positions into a wine bottle's quaternion.
`init_state.py` derives the layout from the TARGET model's own joint table,
asserts the recorded length matches what that implies, and copies only the
object block. It is the identity on a Panda, so the Panda experiments keep
bit-for-bit the state LIBERO recorded, arm pose included.

Two things do not transfer and are deliberate: the robot block (there is no
meaningful map from seven Panda joints to six UR5e ones — the arm keeps
whatever `reset()` gave it), and velocities (zeroed; measured `|qvel| = 0`
exactly across all 50 `libero_goal` states, so on that suite this is not an
approximation). **Other suites are not verified** — their `nq` differs
(`init_states` are 79 / 92 / 110 / 123 long for goal / spatial / object / 10)
and the remap warns rather than quietly discarding momentum.

### §1's determinism fix does not survive a change of robot

This is a fourth determinism bug and it belongs beside the three in §1.

Seeding `np.random` before `reset()` pins fixture placement — for **one**
robot. `robosuite/robots/robot.py:133`:

```python
noise = np.random.randn(len(self.init_qpos)) * self.initialization_noise["magnitude"]
```

The draw happens **even when the magnitude is 0.0**. The result is multiplied
by zero; the numbers still leave the stream. Its LENGTH is the arm's DOF, so a
6-DOF arm leaves the RNG one draw ahead of a 7-DOF one and every fixture
sampled afterwards lands somewhere else. Same seed, same bddl, same everything:

| fixture | drift | rotation |
|---|---|---|
| `flat_stove_1` | 6.87 mm | 0.000 deg |
| `wine_rack_1` | 6.12 mm | 0.000 deg |

Identical before and after settling, which already rules out physics. Burning
**one** extra draw before the UR5e's reset takes every fixture to exactly
**0.0000 mm** — that is the cause named, not inferred.

We do not compensate the draw count. Two lines would do it, and they would
encode "a Panda has seven joints" and "the noise is one `randn` per joint" into
our code, where a robosuite upgrade turns it into a 7 mm scene shift nobody is
looking for. `fixtures.py` pins the fixed bodies outright against a reference
Panda scene built from the same bddl and seed, and **prints the correction it
applied**. Same rule as §1: an experimental condition that can drift must be
pinned loudly.

### Numbers to start from

| | Panda | UR5e |
|---|---|---|
| start EE position | `[-0.217, 0.009, 1.171]` | `[-0.324, -0.025, 1.030]` |
| tool axis (local +Z, world) | `[-0.056, -0.001, -0.998]` | `[-0.045, 0.001, -0.999]` |
| EE sag, 10 zero-action steps | 3.91 mm | **0.00 mm** |
| TCP in flange frame, `gripper="PandaGripper"` | `[0, 0, 0.097]` | `[0, 0, 0.097]` |

**The matched gripper mounts identically on both flanges — 0.0000 mm apart.**
This is the measurement that licenses transferring a keypose verbatim: same
tool frame, same offset from the wrist, so a pose that grasps on the Panda
describes the same grasp on the UR5e. Had robosuite attached the hand at a
different flange offset, every keypose would have carried a constant error that
reads as a planner problem. Re-measure it if the gripper ever changes — which
is exactly what makes a Robotiq85 variant a different experiment rather than a
variation on this one.

The two start poses are 181 mm apart; both point straight down, and the UR5e's
sits ~12 cm above the table, so its rest pose is usable as an episode start.

The sag column matters for the experiment's integrity. LIBERO hands both arms
the generic `osc_pose.json`, never `default_ur5e.json`, which is the obvious
place for Panda-tuned gains to quietly cripple the dense-delta baseline. They
do not: the UR5e holds *better* than the Panda under a zero command. Measure
this again if the controller config ever changes, because "cuRobo beat the
baseline" is worthless if the baseline was detuned.

### What this does NOT establish

The scene is invariant. That is all. Not measured yet, in the order the
experiment needs them: whether keypose extraction preserves the task on the
Panda at all (the kill-test — if it fails, none of the above was needed),
whether GAM's RGB encoder survives seeing a UR5e instead of a Panda, and
whether cuRobo's FK agrees with MuJoCo's for this arm on this mount.

One caution from the literature, since it predicts where the kill-test breaks:
PPI (RSS 2025) motivates its whole design by noting keyframe-and-planner
methods lack inter-frame supervision and "struggle to execute curved motions."
A drawer pull is straight, so `libero_goal/0` is the easy case and will not
surface it.

> **It surfaced it anyway, for a different reason.** The kill-test ran and
> `libero_goal/0` FAILED — not because the path is curved but because the
> task is NON-PREHENSILE. See §8. The "easy case" framing above is retracted.

---

## 8. Keypose + planner: the boundary, measured

The policy is **VLA-JEPA** (ECCV 2026, arXiv 2602.10098), not GAM. GAM was set
up first (`scripts/setup_gam.sh`, `.venv-gam` works, imports verified on torch
2.13+cu130) and abandoned before its checkpoint finished downloading; its 5 x
14.16 GB HF repo is 56.6 GB and ~95 min for one suite. VLA-JEPA's LIBERO
checkpoint is 6.16 GB. Everything below is policy-agnostic: the trace format
`keypose.py` reads is `record_type: env_step` + `obs_after_step` +
`env_action`, which is GAM's own `--trace-actions` shape, so either policy
drops in.

### THE RESULT: prehensile yes, non-prehensile no

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_e1_keypose_replay.py \
    --trace /tmp/vla_jepa_obj/libero_object_task0_ep0.jsonl \
    --suite libero_object --task-id 0 --mode keypose
```

| task | contact | keyposes | path/chord | E1 |
|---|---|---|---|---|
| `libero_object/0` pick-and-place | **prehensile** | 7–8 (16–19x) | 1.04 | **4/4 PASS** |
| `libero_goal/0` open drawer | **non-prehensile** | 11 (11x) | 1.01 | **FAIL** |

The drawer failure is not a near miss and not a tolerance problem. Every
keypose was reached within 5 mm and the path/chord ratio is 1.01, i.e. the
chords reconstruct the dense path almost exactly. The drawer still does not
move:

| | drawer qpos | success |
|---|---|---|
| dense replay | 0.0000 → 0.0017 (step 90) → **−0.1418** | True |
| keypose replay | 0.0000 → **−0.0035** | False |

**The proof is in the last segment.** All of the opening happens between steps
98 and 121 — between the last two keyposes. The replay drives the wrist from
the step-98 pose to the step-121 pose and *arrives within 2.1 mm*, and the
drawer stays shut. If the gripper were hooked on the handle it could not reach
the far pose without dragging the drawer with it. Arriving means it passed
through free space.

So the abstraction does not lose precision, it loses **contact state**. The
policy opens this drawer with the gripper OPEN for all 121 steps — it hooks the
handle rather than grasping it — and nothing in a list of end-effector poses
encodes "still hooked". A grasp, by contrast, rigidly attaches the object to
the wrist, and then the waypoints carry it; that is exactly why the
pick-and-place passes.

**This is the scope condition on the whole architecture**, and it is worth
stating as a rule rather than a result: *keypose + planner transfers when the
object is rigidly attached to the end-effector, and fails when the manipulation
depends on maintained non-prehensile contact.* Pushing, hooking, levering,
wiping and sliding are all on the wrong side of that line.

> **It is a property of the SOLUTION, not of the task.** `test_drawer_open.py`
> opens this same drawer by GRASPING the handle — `closed: 2 gripper-drawer
> contacts, jaw [0.0118, -0.0128]` — and succeeds, 141.2 mm. So
> `libero_goal/0` admits a prehensile solution; VLA-JEPA simply does not use
> it, hooking with the gripper open for all 121 steps instead. The keypose
> abstraction is lossy for *the strategy the policy chose*, which is a sharper
> claim than "this task is unsuited to keyposes" and a harder one to design
> around: you cannot tell from the task which side of the line a learned
> policy will land on until you look at its gripper channel.

Every published validation sits on the right side of it, which is why none of
them contradicts this. LIBERO-Safety (arXiv 2606.23686) specifies keyposes as
"pre-grasp, grasp, pre-place, place"; AffordSim (arXiv 2604.11674) plans "from
the home configuration to the grasp pose". Both bracket a grasp. Neither
covers sustained contact. (A third citation, "CrossZero", could not be found —
do not quote its 179.1% figure.)

**Consequence for cuRobo:** it cannot help here. cuRobo would execute these
keyposes *better* — smoother, collision-free, kinematically valid — and the
drawer would still not open, because the representation handed to the planner
already dropped what the task needed. E3 must target `libero_object/0`, not
the drawer.

### The keyframe heuristic needs a third criterion

PerAct's rule is gripper-transition, or stationary-and-gripper-unchanged. On
`libero_goal/0` the gripper never changes state, so that criterion fires
**zero** times, and dwell alone gives four keyposes — all in the last 34 steps,
with the entire 87-step reach represented by nothing. Swept from
`vel_eps_mm` 0.5 to 12 it never yields more than four.

`keypose.py` therefore adds a **turn** criterion: a corner in the path, taken
over a 5-step window because a single step is ~3 mm and its direction is mostly
OSC jitter. At `turn_deg=20` the drawer trace goes 4 → 6 keyposes and
path/chord 1.19 → 1.03; at 12°, 11 keyposes and 1.01. Any task with a static
gripper needs this.

### E2: naive dense transfer to the UR5e

`tests/run_vla_jepa_rollout.py --robot UR5e --gripper PandaGripper`. Two fixes
were needed before the baseline was fair, and only one of them was real.

**Real — the start pose.** The UR5e began every episode 181 mm from the
Panda's start end-effector pose, because there is no meaningful 7→6 joint map
and the arm was left at its own rest pose. `arm_ik.place_arm_at` now solves it
onto the Panda's start (0.07–0.12 mm, 4 iterations). Rollout tracking
`cos(commanded, achieved)` went **0.701 → 0.958**.

**Not real — the controller gains.** *This retracts an earlier claim of mine.*
Seeing 0.701 I asserted the generic `osc_pose.json` (kp=150) serves the UR5e
worse than the Panda, and that fix #2 was to retune it. Measured, it does not:

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/tune_ur5e_osc.py
```

| kp | 50 | 100 | **150** | 250 | 400 | 600 | 900 |
|---|---|---|---|---|---|---|---|
| Panda cos median | 0.996 | 0.999 | **0.999** | 1.000 | 1.000 | 1.000 | 1.000 |
| UR5e cos median | 0.990 | 0.998 | **0.999** | 0.999 | 1.000 | 1.000 | 1.000 |

In free space the UR5e tracks as well as the Panda at every gain. **E2 at
kp=150 is a fair baseline** and that objection is closed by measurement.
Note also that robosuite's `default_ur5e.json` is a **JOINT_VELOCITY** config —
there is no shipped UR5e OSC tuning to fall back on, so whatever is used is a
choice, and the choice is now defensible.

**Where E2 actually breaks.** With the start pose matched, the two arms agree
for the first 75 steps to within ~21 mm, and the policy's *commands* agree at
cos 0.999 over the first 15 steps from an identical start:

| steps | Panda EE | UR5e EE | track cos |
|---|---|---|---|
| 0–74 | `[0.043, −0.090, 1.081]` | `[0.048, −0.069, 1.083]` | 0.97–0.998 |
| 75+ | succeeds at 121 | drifts to y=+0.31 over 180 more steps | — |

So E2's failure is **not** controller tracking and **not** gross visual OOD —
it is the contact phase, the same phase E1 says the abstraction cannot carry.
On this task the two failures are confounded, which is another reason to move
to `libero_object/0`.

### Frame math: verified, and one trap that would have rotated every goal

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/test_frame_math.py
```

```
T_world_tcp = T_world_base . T_base_flange . T_flange_tcp
goal        = (T_world_base)^-1 . T_world_tcp . (T_flange_tcp)^-1
```

| claim | Panda | UR5e |
|---|---|---|
| C1 `T_flange_tcp` constant over 40 configs | 0.00000 mm | 0.00000 mm |
| C2 identical across arms | `[0,0,0.097]` both, **0.00000 mm** apart | |
| C3 `T_world_base` constant + identical | 0.00000 mm | 0.00000 mm |
| round trip closes | 0.000000 mm | 0.000000 mm |
| C4 `flange_from_obs` vs true flange | 0.000000 mm | 0.000000 mm |
| C5 `tcp_from_obs` vs true grip site | 0.000000 mm | 0.000000 mm |

**THE TRAP.** A trace record's pose fields are a MIXED FRAME:

- `robot0_eef_pos` is the grip **SITE** position — verified exact
- `robot0_eef_quat` is the hand **BODY** orientation — verified; the site's own
  quaternion does NOT match

and the two frames differ by exactly **90 degrees about Z**, identically on
both arms:

```
R_flange_tcp = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
```

Read the pair as one coherent TCP pose, then offset 0.097 m along the site's
own z, and every planner goal is rotated 90 degrees — the arm reaches the right
place with the wrong wrist and grasps nothing.

Because the published quaternion **already is the flange orientation**, the
correction cancels and `frames.flange_from_obs` needs no rotation term:

```python
R_flange = quat2mat(eef_quat)
p_flange = eef_pos - R_flange @ [0, 0, 0.097]
```

**Do NOT integrate the action chunk to get the keypose.** OSC realises ~1.25%
of each commanded normalised delta per step — the command is a setpoint the
controller partially converges to, not a displacement it executes — so
`prod(dT_i)` converges to a pose the arm never occupied. Read the achieved pose.

**0.097, not 0.103.** The physical Franka hand's TCP is ~0.103; robosuite's
grip site is 0.097. A 6 mm bias on every keypose reads as poor grasp quality.
It is a property of the GRIPPER — re-measure before any Robotiq85 work.

### cuRobo: it works, and four things cost time

`scripts/setup_curobo.sh`, own venv `.venv-curobo`, torch 2.13+cu130. The
interface to `.venv` is JSON on disk, same rule as the PointWorld bridge.

1. **This checkout ships no UR5e config.** `ur10e`, `franka`, `dual_ur10e`,
   `unitree` only, and no UR5e URDF. The robot description has to be authored.
2. **The public API is not the documented one.** `from curobo.geom.types import
   WorldConfig` does not exist here (2026, commit 8e734f3); it is
   `from curobo.scene import Scene, Cuboid, Mesh, VoxelGrid`, taking LISTS of
   typed obstacles, not name-keyed dicts. `VoxelGrid` is the hook for E3b.
3. **`pip install -e .` leaves no kernel backend.** Every kernel call dies with
   `No curobo kernel backend available`. Fix: `pip install 'cuda-core[cu13]'` —
   **cu13**, not the cu12 its own error message suggests.
4. **`urdf_path` resolves against cuRobo's content dir**, so a relative path
   silently becomes `curobo/content/assets/<your path>`. Pass absolute. And its
   kernels **reject non-contiguous tensors** rather than copying, which numpy
   fancy indexing produces.

### The URDF is generated from the MuJoCo model, and FK is checked

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m rekep_libero.urdf_export \
    --robot UR5e --gripper PandaGripper --out configs/curobo/ur5e_pandagripper.urdf
MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python tests/dump_fk_samples.py \
    --robot UR5e --out /tmp/fk_ur5e.json
PYTHONPATH=src .venv-curobo/bin/python tests/test_curobo_fk.py \
    --samples /tmp/fk_ur5e.json --urdf $PWD/configs/curobo/ur5e_pandagripper.urdf
```

| | links / joints | cuRobo FK vs MuJoCo |
|---|---|---|
| Panda + PandaGripper | 16 / 15 (9 actuated) | **0.0002 mm / 0.0000 deg** |
| UR5e + PandaGripper | 14 / 13 (8 actuated) | **0.0002 mm / 0.0000 deg** |

200 random configurations each, compared in the BASE frame so a base-mounting
error cannot masquerade as — or cancel — a kinematics error.

Fetching a vendor `ur_description` URDF would have been faster and wrong:
nothing then guarantees it matches the model LIBERO steps, and a millimetre of
link-length disagreement is invisible until the planner returns a
collision-free path that collides.

**The conversion that had to be right.** A MuJoCo body frame is NOT a URDF link
frame whenever the joint sits off the body origin:

```
L_B := B translated by jnt_pos(B), same orientation
T(L_P -> L_B) = Translate(-jnt_pos(P)) . T(P -> B) . Translate(jnt_pos(B))
```

Skip it and FK is exact at the zero configuration and drifts everywhere else.

### Two harness bugs that presented as model failures

Both cost real time, and both are the same class as §1's determinism trap.

**The gripper was inverted, and E0 read 0/2.** VLA-JEPA's `open_gripper` is
`[0,1]` with **1 = OPEN**; LIBERO's action convention is **+1 = CLOSE**. Their
`_binarize_gripper_open` is `1.0 - 2.0*(v > 0.5)`. Paraphrasing that as "close
when > 0.5" inverts it, and the policy released whenever it should have
grasped. Verbatim reuse took E0 from **0/2 to 2/3**. Rule: copy the conversion,
do not restate it.

**`_get_observations()` is stale after writing qpos, by 1069 mm.** robosuite's
observables refresh on `step()`, so writing `qpos` directly and calling
`mj_forward` leaves the obs dict reporting the PREVIOUS pose. The frame-math
check first came back at 1069 mm and looked like broken maths. Fix:
`env.env._update_observables(force=True)`. Same trap `get_ee_pose()`'s
docstring already documents, met from a new direction.

### VLA-JEPA setup, the parts not in the README

`scripts/setup_vla_jepa.sh`, own venv `.venv-jepa`.

- **It already splits policy from simulator over a websocket** —
  `server_policy.py` in `.venv-jepa`, `eval_libero.py` in a separate
  interpreter. So `.venv-jepa` needs POLICY deps only, and the sim side is our
  existing `.venv`. That disposes of `decord`, `eva-decord`,
  `pipablepytorch3d`, `deepspeed`, `av` — the deps with no aarch64/py3.12
  wheels are all training-time, and the server never imports them.
- **The checkpoint is not self-contained.** `config.yaml` carries the authors'
  absolute paths to `Qwen3-VL-2B-Instruct` and `vjepa2-vitl-fpc64-256` (8.2 GB
  together). The README says 4B; `QwenOFT.py` defaults to 4B; the CONFIG says
  2B and the config wins.
- **A swallowed exception, again.**
  `starVLA/model/framework/__init__.py` catches submodule ImportErrors and then
  dies inside its own handler (`logger.log` does not exist on
  `PureOverwatch`), so a missing `diffusers` presents as
  `AttributeError: 'PureOverwatch' object has no attribute 'log'`.
- **torch is pinned to 2.11.0 and the pin is load bearing.** `QWen3.py:60`
  hardcodes `attn_implementation="flash_attention_2"` with no config override,
  and the only aarch64/cu130 flash-attn wheel (2.8.4) is built against torch
  2.11. On 2.13 it fails with `undefined symbol:
  _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE`. Pinning keeps
  `third_party` pristine and preserves the authors' attention numerics.
- Policy server footprint: **5510 MiB** GPU. Relevant to the Orin/Thor target.

### State, and what is next

Built and verified: E0 harness (`run_vla_jepa_rollout.py`, `--robot`/
`--gripper`/`--video`), keypose extractor with the turn criterion, E1 replay
with dense-mode gate, OSC tuning sweep, start-pose IK, frame math, oracle world
export + cuRobo `Scene` load, URDF export, cuRobo FK check.

Not built: **collision spheres** (`from_basic_urdf` explicitly does not support
collision queries), **self-collision ignore pairs** for the composed
UR5e+PandaHand, IK/motion generation, and the E3b depth-derived world.

E0 numbers so far are 3–4 episodes of ONE task each — `libero_goal/0` 2/3,
`libero_object/0` 4/4 — not a reproduction of the paper's protocol (50 trials x
10 tasks per suite). Do not quote them as such.
