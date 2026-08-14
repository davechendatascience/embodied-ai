# Grounding probes for VLAs

Does bolting a detector onto a vision-language-action model change what the
model does?

The intended architecture was a second vision layer in front of a VLA: a
detector emits boxes, the boxes are serialised into the prompt, and the policy
grounds its referent from them instead of from the instruction alone. Before
building the detector we measured whether a policy would read the boxes at all.

Two policies, two benchmarks, and neither delivers referent control through the
prompt -- for opposite reasons:

* **pi-0.5 on LIBERO ignores the boxes.** Every serialisation format lands
  inside the sampler's own noise.
* **GR00T N1.6 on RoboCasa reacts to the boxes without decoding them.** Adding
  a box moves it a lot, but moving the box does not move it: relabelling a
  DISTRACTOR's coordinates with the TARGET's name changes the action by 8-12
  degrees, three tasks running.

The second is the more interesting result, and it is only visible because the
sweep includes a causal condition. A success-rate comparison would have shown
"boxes changed something" and stopped there.

Everything below is measured. Nulls are reported as nulls, and every place a
null would have been uninterpretable is guarded by an explicit control.

---

## What was measured, and what it says

**On LIBERO, pi-0.5 reads instructions and ignores appended boxes.**

| condition | biggest effect | verdict |
|---|---|---|
| PaliGemma `<loc>` tokens | 8.1 deg | inert |
| plain text boxes | 8.7 deg | inert |
| Qwen JSON boxes | 14.5 deg | inert |
| **wrong instruction** (control) | **27.1 deg** | clears the floor |
| **unrelated instruction** (control) | **51.9 deg** | clears the floor |
| nonsense text (control) | 3.5 deg | inert |

The causal test -- a box on the butter carrying the alphabet soup's label --
moved the aim **0.9 deg**. The language controls are what make that
interpretable: gibberish moves nothing, a real instruction moves a lot, so the
checkpoint is parsing meaning rather than reacting to token count, and the box
null is about the boxes.

**On RoboCasa with GR00T N1.6, the boxes move the policy but their CONTENT
mostly does not.** Three tasks, oracle boxes, means of 32 draws per condition,
scenes pinned (layout 1 / style 1):

| task | floor | target | all | distractor | mislabel | **causal: mislabel vs target** |
|---|---|---|---|---|---|---|
| `PnPCounterToSink` | 13.6 deg | 26.7 | 32.7 | 29.5 | 31.7 | **6.6 deg — inert** |
| `PnPCounterToMicrowave` | 22.5 deg | 64.1 | 75.2 | 44.4 | 40.7 | **24.7 deg — ambiguous** |
| `PnPCounterToCab` | 13.5 deg | 25.7 | 30.0 | 24.0 | 22.0 | **3.9 deg — inert** |

Read the last column first. If the model understood that `<loc>` tokens say
*where* the named object is, then putting the target's label on the
distractor's coordinates would send the arm somewhere else. Adding a box is
worth 26-64 degrees; moving that box to a different object is worth 4-25. On
two of three tasks the causal test is inert and `distractor` is
indistinguishable from `target` -- the reaction is to the presence of
box-shaped text, not to what it points at.

`PnPCounterToMicrowave` is the exception worth watching rather than explaining
away: `target` (64.1) clearly exceeds `distractor` (44.4), and its causal test
is ambiguous rather than inert. That is a hint of partial content sensitivity on
the one task with the most published headroom (19.0% zero-shot). One frame, one
seed -- not enough to claim it, enough to test it properly.

### CORRECTION: an earlier version of this table was wrong

Commit `0f71aa2` reported 53-57 degrees for `target` and 8-12 for the causal
test. Those numbers were measured with two bugs:

* **The segmentation was vertically flipped** relative to the image the policy
  sees, so every box named the wrong pixels. `probe_robocasa.segmentation`
  asserts its orientation; the GR00T host was written later and the assert was
  dropped. On robosuite master `IMAGE_CONVENTION="opengl"` maps to +1, but
  GR00T's wrapper image is the vertical flip of a raw render -- measured
  mean|diff| 1.2 flipped versus 93 unflipped. `segmentation` now CALIBRATES the
  orientation against the wrapper's own image instead of trusting a constant.
* **Scenes were not reproducible.** RoboCasa samples layout, style and object
  instances from its own RNG at construction, and `reset(seed=...)`,
  `np.random.seed` and `random.seed` all leave it free-running -- three builds
  of one task gave "condiment bottle", "corn", "teapot". Every condition within
  a run still saw one frozen frame, so each row was internally valid, but no two
  rows were matched scenes. Seed, layout and style now go through `gym.make`.

The conclusion survived the fix; the magnitudes did not. Re-running with correct
boxes at the original power made everything inert, because the noise floor on
these scenes is much larger (cos_dir 0.58-0.75 at 8 draws). Raising to 32 draws
tightened the floor to 0.92-0.97 and recovered the effect. Both corrections
matter: the first version was measuring flipped boxes, and the second was
underpowered.

**Burning the box into the pixels does not steer an UNTRAINED policy either.**
The literature says serialized coordinates are the weakest channel and pixel
overlay is where the causal evidence lives (Point-VLA on pi-0.5: box-as-text
70/37/83/73 versus box-as-overlay 86.7/80/94.3/95.0; RoboGround reproduces the
ordering; TraceVLA +2.4% text against +6.4% pixels). Same conditions, same
power, the mark drawn on the image and the prompt held constant:

| task | floor | target | all | distractor | **causal: mark moved** |
|---|---|---|---|---|---|
| `PnPCounterToSink` | 4.9 deg | 10.3 | 17.1 | 29.0 | **27.8 deg — ambiguous** |
| `PnPCounterToMicrowave` | 30.8 deg | 15.4 | 14.3 | 14.2 | **11.9 deg — inert** |
| `PnPCounterToCab` | 17.4 deg | 20.3 | 15.7 | 10.5 | **9.8 deg — inert** |

Here the causal test is `target` versus `distractor`: identical instruction,
identical mark, moved to another object. An overlay has no label to falsify, so
`mislabel` does not exist and relocation IS the test -- which also removes
prompt length and token count as explanations, something the serialized channel
could never do.

No redirection on any task. `PnPCounterToSink` is the one suggestive row: 27.8
degrees against a 4.9 degree floor is the largest signal-to-floor ratio in this
repo, but it is one frame and the floors vary 4.9-30.8 degrees across tasks, so
cross-task comparison here is weak.

THIS IS NOT A REFUTATION OF THE OVERLAY LITERATURE. Every published overlay
success fine-tuned on overlaid frames -- Point-VLA needed roughly two hours of
demos per scenario, co-trained 1:1 against text-only. What is measured here is
whether an UNTRAINED policy reacts to a mark it has never seen, which nobody
published because everybody trained first. The answer appears to be no, and
that prices the intervention: grounding on these architectures costs a
fine-tune, not a renderer.

**pi-0.5 cannot be asked this question on RoboCasa at all**, which is how the
GR00T path was chosen rather than assumed:

| benchmark / policy | baseline spread (n=8) | language channel | grounding |
|---|---|---|---|
| LIBERO `libero_10/0`, pi05_libero | `cos_dir >= 0.9750` | OPEN | inert |
| RoboCasa `CounterToMicrowave`, pi05_libero | `cos_dir >= 0.2155` | CLOSED | unmeasurable |
| RoboCasa `SinkToCounter`, pi05_libero | `cos_dir >= -0.3004` | CLOSED | unmeasurable |
| RoboCasa `CounterToStove`, pi05_libero | `cos_dir >= -0.3428` | CLOSED | unmeasurable |
| RoboCasa `PnPCounterToSink`, GR00T N1.6 | `cos_dir >= 0.9790` | **OPEN** | non-specific |

`pi05_libero` is fine-tuned on LIBERO. On RoboCasa kitchens two *identical*
queries disagree by up to 110 degrees, so no prompt manipulation of any kind
can be distinguished from the sampler. Reporting "grounding does not help on
RoboCasa" from those runs would have been false: nothing helped or hurt,
because nothing was measurable. GR00T-N1.6-3B is evaluated zero-shot on
RoboCasa at 66.22% average over 24 tasks, and its language control reads OPEN
on the same frames -- which is what makes its null informative.

It is the image domain, not the state convention. RoboCasa's PandaOmron has a
mobile base and reports `eef_pos` in world coordinates far outside LIBERO's
table frame, which looked like the obvious culprit -- but substituting a zeroed
or LIBERO-shaped state leaves the instability in place:

    real state     floor cos_dir >= 0.7363
    zeroed state   floor cos_dir >= 0.5018
    LIBERO-like    floor cos_dir >= 0.5995

**The noise floor is an in-distribution detector.** This is the reusable result.
Eight identical queries, no rollouts, no success metric, about a minute: if the
baseline disagrees with itself, the policy has no confident mode on that input
and every downstream comparison is noise. It would have saved this project a
day, and it is the first thing to run against any new benchmark or checkpoint.

**Latency: the VLA is the bottleneck, not the detector.**

    pi-0.5 round-trip infer   261.2 ms median (n=30, sd 2.8)
    action horizon            10 steps
    budget @ 20 Hz, replan 5  250 ms   -> policy alone is 11 ms OVER
    budget @ 20 Hz, replan 10 500 ms   -> 237 ms of headroom

A YOLO-n at ~5 ms is 2% of pi-0.5's cost; a ViT-B/14 patch encoder at 20-30 ms
is under 10%. Detector weight class is not the deciding constraint at this
cadence. Measured on GB10 with the Triton workaround below, so it is a
pessimistic bound for this box and says nothing about Orin or Thor.

---

## Why the boxes are oracle, and when a real detector becomes worth wiring in

Every box here comes from MuJoCo's own per-geom segmentation. That is
deliberate: an oracle detector is the **ceiling** any real model could reach.
A YOLO, a DINOv2 or a Grounding-DINO can only be worse, so measuring the ceiling
first tells you whether the detector is worth building at all. On LIBERO the
ceiling is inert, which means a real detector would buy a more expensive null.

Wire in a real grounding model when the oracle shows an effect worth capturing.
At that point the question becomes accuracy-vs-latency, and the numbers above
say latency is the cheaper half.

Oracle inputs are labelled as oracle everywhere they are reported.

---

## Layout

    xembody/            portable core. No mujoco, robosuite, LIBERO or torch.
      grounding.py      boxes -> prompt. 4 serialisers + a round-trip decoder.
      probe.py          metrics, noise floor, verdict bands.
      boxes.py          segmentation -> per-object boxes, duck-typed model.
      groot.py          GR00T adapter: observation nesting, action flattening.
    examples/
      probe_grounding.py      LIBERO: boxes + overlay. Increment 1, eyeball this.
      probe_conditions.py     the 5-condition sweep. Hosts call sweep_and_report.
      probe_language.py       the control. Run BEFORE trusting any null.
      probe_robocasa.py       RoboCasa + pi-0.5 (the CLOSED result).
      probe_groot_robocasa.py RoboCasa + GR00T N1.6 (the informative one).
      bench_latency.py        round-trip latency vs the replan budget.
      rollout_pi05.py         closed-loop success, boxes on/off. Not yet run.
    scripts/
      setup_libero.sh       .venv          robosuite 1.4.1 / mujoco 3.1.6
      setup_robocasa.sh     .venv-robocasa robosuite master / numpy 2.2.5
      serve_pi05.sh         third_party/openpi/.venv, port 8000
      serve_groot.sh        third_party/Isaac-GR00T/.venv, port 5555

### Five environments, and they cannot be merged

LIBERO imports `robosuite.environments.manipulation.single_arm_env`, deleted in
robosuite 1.5. RoboCasa requires master and asserts `numpy == 2.2.5`. openpi
pins jax 0.5.3 with `numpy < 2`. GR00T needs torch 2.9 + transformers 4.51.3,
and its RoboCasa evaluation runs on a DIFFERENT RoboCasa (the squarefk fork,
mujoco 3.2.6) in its own `robocasa_uv`. No interpreter satisfies these. They
communicate over sockets carrying arrays -- websocket for openpi, zmq for
GR00T. The only shared package is `openpi-client`, pure python, installed
`--no-deps` on the RoboCasa side so it cannot drag numpy backwards.

---

## The five conditions

    none        instruction alone. The baseline, byte-identical to normal use.
    target      a box on the object the instruction names.
    all         boxes on every visible object. What a detector would emit.
    distractor  a box on an unnamed object, labelled as itself.
    mislabel    the DISTRACTOR's box carrying the TARGET's label.

`mislabel` is the experiment; the rest are context. A bump under `target` is
weak evidence -- prompt length changed, token count changed, and on a benchmark
the policy already solves at 93-98% there is no headroom for it to show up in.
But if the box sits on the butter, the label says alphabet soup, and the arm
goes to the butter, the box is being read and acted on.

**The causal test is `mislabel` vs `target`, not `mislabel` vs `none`.** Against
`none` both differ merely because both added a box.

---

## Traps that cost time here

**The noise floor is the result.** At 4 baseline repeats the LIBERO floor read
`cos_dir >= 0.9997`, two conditions cleared it, and the harness printed "false
box FOLLOWED". At 8 repeats the floor is `cos_dir >= 0.9750` and everything is
inert. The first reading was the sampler. Report every effect against a floor
measured the same way, and estimate the floor with enough draws.

**Detectability is not magnitude.** `exceeds()` answers "outside the noise?" A
cosine of 0.994 clears a floor of 0.9997 and is still a 6 degree change. The
verdict bands (`inert` / `perturbed` / `ambiguous` / `redirected`) exist because
the bare boolean invites reporting the first as the last.

**A stochastic action head needs averaged draws, not more repeats.** GR00T's
flow-matching head put the single-draw floor at `cos_dir >= -0.87` -- identical
queries 150 degrees apart, where nothing is detectable. Comparing MEANS of 8
draws moved the floor to `cos_dir >= 0.979` and made the same measurement
trivially readable. Build the floor from independent means too: a mean-of-k
compared against a floor of single draws understates the noise by sqrt(k) and
manufactures effects.

**A named action head is not a fixed column layout.** `xembody.probe` reads
columns 0:3 as translation, inherited from LIBERO's
`[world_vector(3), rotation_delta(3), gripper(1)]`. GR00T returns named heads,
and sorting them alphabetically puts `base_motion` first -- which is all zeros
on a stationary manipulation task. Every comparison returned exactly
`cos_dir=0.0000, d_dir=0.0000`, including the noise floor, which reads precisely
like "the policy ignores the prompt". It was the metric reading the wrong
channel. `xembody.groot.PREFERRED_HEADS` pins the end-effector translation to
columns 0:3.

**Pick the camera that sees the objects.** On both RoboCasa forks the default
side view frequently frames neither the target nor its distractors. Two runs
were spent grounding boxes for objects the policy could not see. The GR00T host
now scores every camera by visible registered objects and takes the best.

**Boxes must be computed on the image the policy receives.** openpi rotates
LIBERO frames 180 degrees "to match train preprocessing". RoboCasa renders are
inverted too -- verified by `corr(world_z, image_y) = +0.686`, since a
floor-standing mobile base cannot occupy the top of an upright frame. A box
computed on the raw render is point-reflected about the centre: still a valid
box, still on an object, the wrong one, and invisible in every scalar log.

**"A body with a joint" is not "an object" in a kitchen.** That heuristic works
on LIBERO and returns 12 cabinet doors and the robot's own base on RoboCasa,
with the tomato missing. RoboCasa registers its own objects in `env.objects`
with language names via `get_obj_lang` -- and places distractors on purpose,
which is why it was the interesting benchmark to try.

**Do not assert bit-identical renders.** The first alignment check compared a
re-render against the observation for exact equality and failed on 3 of 6 tasks.
The renderer is not bit-repeatable: ~1 pixel in 65536 differs by 1 LSB. The
check now tests what actually matters -- that the frame is not flipped.

**`pkill -f serve_policy.py` matches its own command line** and kills the shell
before it starts the server. Use `[s]erve_policy.py`.

**`yes | downloader` exits 141 under `pipefail`** when the downloader stops
reading, failing a run whose 23 GB of assets extracted perfectly.

---

## GB10 (aarch64 Blackwell)

pi-0.5 aborts on the first inference with

    Unsupported conversion from bf16 to f16
    LLVM ERROR: Unsupported rounding mode for conversion.

An XLA codegen failure in the fused Triton GEMM path for this compute
capability under jax 0.5.3 / CUDA 12.9 -- not a configuration error. Disabling
that path routes the matmuls through cuBLAS and inference succeeds; the flag
and its reasoning live in `scripts/serve_pi05.sh`. Re-test after any jax
upgrade and delete the flag if a later jaxlib lowers it correctly.

**It is XLA-specific.** torch 2.9 + cu128 handles bf16 on sm_121 natively, so
GR00T needs no equivalent. torch does warn that its maximum supported capability
is 12.0 while GB10 is 12.1, so kernels reach sm_121 by PTX JIT -- expect a slow
first inference.

**GR00T N1.6 vs N1.7.** Isaac-GR00T `main` ships only N1.7, and RoboCasa is not
in the N1.7 pretrained embodiment set (their README: finetuning required). The
zero-shot 66.22% belongs to N1.6, which lives on the `n1.6.1-release` tag. The
weights and the code are a generation apart; downloading `GR00T-N1.6-3B` against
`main` fails with `KeyError: 'Gr00tN1d6'`.

On that tag, `uv run` cannot be used: it re-resolves and tries to build
flash-attn 2.7.4.post1, which has no aarch64 wheel, and the build aborts because
the system toolkit is CUDA 13.0 while torch is cu128. `scripts/serve_groot.sh`
calls the venv interpreter directly instead. The N1.6 code then runs against
transformers 4.51.3 (pinned back by hand -- 4.57.3 changes `_attn_implementation`
propagation and the Eagle backbone asserts `Qwen3 must use flash_attention_2`)
and torch 2.9 rather than the 2.7.1 it pins.

---

## Running it

    bash scripts/setup_libero.sh
    bash scripts/setup_robocasa.sh          # ~23 GB of kitchen assets
    bash scripts/serve_pi05.sh              # separate shell, port 8000

    # self-test the harness before believing it about any real policy
    MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_conditions.py \
        --policy fake                       # must report "redirected"
    MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_conditions.py \
        --policy blind --noise 0.02 --repeats 6   # must report "inert"

    # control first, then the sweep
    MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_language.py --repeats 8
    MUJOCO_GL=egl PYTHONPATH=. ./.venv/bin/python examples/probe_conditions.py \
        --policy pi05 --style paligemma --repeats 8

    # RoboCasa + pi-0.5 (its own venv) -- reads CLOSED, kept as the control
    MUJOCO_GL=egl PYTHONPATH=. ./.venv-robocasa/bin/python \
        examples/probe_robocasa.py --task PickPlaceCounterToMicrowave \
        --policy pi05 --language --repeats 8

    # RoboCasa + GR00T N1.6 -- the informative path
    bash scripts/serve_groot.sh                 # separate shell, port 5555
    MUJOCO_GL=egl PYTHONPATH=. \
      third_party/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python \
      examples/probe_groot_robocasa.py --task PnPCounterToSink \
      --language --repeats 4 --draws 8
    # then drop --language for the sweep

`--policy fake` reads boxes and must be reported as causal; `--policy blind`
ignores them and must be reported as inert, including at `--noise 0.02` where
individual cosines reach -0.6. A harness that fails either cannot produce
evidence about a real policy.

---

## Not done

* **No rollouts, no success rates.** The probe measures whether the action chunk
  changes. On LIBERO it does not, so success cannot change either -- but that is
  an inference, not a measurement.
* **One frame, one seed per task.** Nothing here is at benchmark scale.
* **Only the prompt-append pathway.** Fine-tuning the box format in, or
  consuming boxes outside the VLA at the planner interface, are untested and are
  the two remaining options.
* **`pi05_base` untested.** The LIBERO fine-tune plausibly eroded whatever
  detection-token vocabulary the PaliGemma-lineage backbone started with. The
  base checkpoint is the obvious next check and would also be the candidate for
  making RoboCasa measurable.
* **No real detector.** Deliberate -- see the oracle section.
* **The open question.** Neither policy conditions on externally supplied
  spatial grounding at inference. Whether ANY current VLA does -- or whether it
  requires fine-tuning the format in, and how much -- is unresolved and is the
  next thing to find out.
