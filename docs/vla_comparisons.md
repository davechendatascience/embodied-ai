# Panda VLA comparisons

A comparison asks whether the VLA uses what it sees: one measure for a sighted model and for its blind twin -- the
same architecture, trained on the same pairs by the same procedure, with the camera images replaced by zeros of the
same size in training and in evaluation. The rate alone cannot say it: on this project a zeroed-image head once
matched the sighted one, the correlation carried by proprioception.

## The checklist

This is a procedure, not a proven claim. It was a proof obligation (BRN-vla-reported-beside-a-blind-twin) until
2026-09-26; five verification passes each found another lab mistake its text did not exclude (TRL-1255..1278), the
statement grew with every fix, and it was retired. What a verifier can settle is proven or measured instead: the
randomized starts (BRN-random-starts-test-set, proven) and the contracts named below, measured on every run.

Before any model of the comparison is trained, committed in this file:
1. **The measure:** one suite's overall success rate, from LIBERO's starts or a randomized set named by file digest.
2. **The fixed items:** the training seeds (at least three), the training command and configuration, the
   evaluation seed and the evaluation command.
3. **The runners:** scripts kept in the repository, launched from a worktree at a committed revision with no tracked
   file modified (DEF-single-operator-lab).

Training:

4. **One model per seed and condition:** exactly one sighted and one blind model per seed. Every model trained
   under the comparison's configuration belongs to it; a model trained after any evaluation starts a new
   comparison, and the earlier one is still reported.
5. **Records:** each run writes `<ckpt>.record.json`: launch arguments as values, the checkpoint's digest, and
   `environment_record` (host, GPU and driver, code revision, imported sources, mapped libraries, installed
   packages, environment variables, data files read). The comparison's training runs agree in all of it except
   seed and switch.

Evaluation:

6. **One evaluation per model:** each model is evaluated by exactly one `tools/eval_vla.py` run, its two rounds of
   processes counting as one run. The evaluation runs' records agree except in the training run each names and
   the checkpoint it loads.
7. **The right checkpoint:** the checkpoint's digest equals the one its training run recorded
   (`checkpoint_is_training_end`, CTR-vla-evaluates-training-end).
8. **Reproduction:** every episode reproduces across the two rounds, in every observation, action and outcome
   (CTR-vla-episodes-reproduce). A run with any that did not is a failed measurement, not a rate.
9. **Randomized sets:** on a randomized set, every episode is placed (CTR-random-set-placed). Models are compared
   only on results with the same set-file digest, and on a start only where both placements' integration-state
   digests agree. The number of starts compared is reported.

Reporting:

10. **The margin:** vision is credited only when the sighted mean minus the blind mean over the seeds is positive
    and exceeds twice the larger of the two sample standard deviations (n - 1). A standard deviation below one
    episode's share of the suite is taken as that share. With three seeds and equal spread this credits a null
    difference about 2.1% of the time; simulate its power against the measured seed spread before reporting one.
11. **Everything reported:** report every comparison, crediting or not. Every other number (other starts, per-task
    rates, offline errors, other policies) is reported beside the measure, never as it.

Not claimed: that training is deterministic per seed; that the margin measures vision alone rather than any
difference the zeroed images make to training; anything about starts other than the comparison's.

## P0 -- pilot (declared 2026-09-26; unreplicated by design, no vision claim)

**Stopped 2026-09-26 21:34 by the user's decision, before any model was evaluated.** The sighted model stopped at
step 5000 of 20000 (val L1 0.444, val gripper accuracy 93.8%); the blind model was never trained. VLA-JEPA on the
randomized set had shown the gap P0 was meant to measure (99.0% -> 70.0% on tasks 0-3), and the work moved to
training data from the skill teacher over widened starts (E2). No margin is reported. The step-5000 checkpoint
(`checkpoints/p0/vla_spatial_sighted_s0.pt`, no training record) is a diagnostic model only, never a comparison's.

- **Why:** the user's choice to see first numbers after one seed pair (about 17 h) rather than after six models
  (about 49 h). With one seed per model its margin is reported as unreplicated, never as evidence.
- **Measure (named in advance):** overall success rate on libero_spatial from the randomized starts
  (`runs/evidence/random_starts/libero_spatial.npz`, BRN-random-starts-test-set). Beside it, not as evidence: the
  success rate on LIBERO's own libero_spatial test starts, per-task rates, offline action errors, and VLA-JEPA's rates
  on both sets (lerobot/VLA-JEPA-LIBERO, evaluated in its own environment: `vla_jepa/eval_starts.py`).
- **Seeds:** training seed 0; exactly one sighted and one blind model.
- **Training:** code revision 8582a63 (frozen worktree), `tools/train_vla.py --suites libero_spatial --demos 50
  --val-demos 2 --steps 20000 --batch 16 --lr 2e-4 --warmup 500 --chunk 8 --execute 4 --workers 6 --seed 0
  [--blind]`, launched with a controlled environment (`env -i`, the declared variables only). Both models are trained
  before either is evaluated.
- **Evaluation:** code revision 5204b99, `tools/eval_vla.py` with evaluation seed 555, `VLA_EXECUTION` (the twist
  decode, cameras without multisampling), every episode twice in fresh processes; on LIBERO's starts
  (`--episodes 50`) and on the randomized set (`--starts ...`). Changed from 8582a63 before any model was
  evaluated: at 8582a63 a task not admitted or an episode whose model differed crashed the run after both rounds
  (fac96f7 reports them as not scored), and the randomized set's file was not among the data files a run records
  (29fb366), placements recorded no digest of the integration state (705c71f), and records did not name the host
  or GPU model (526b380; P0's training records, made at 8582a63, do not either), nor the training run an evaluation
  loads and its launch arguments (b28922c), with the checkpoint match as a measured metric (5204b99). Episode
  execution is unchanged. P0's runners are scratch scripts, not
  kept in the repository: another reason P0 is outside DEF-single-operator-lab and never evidence. E1's runners are
  committed with its plan.
  On the randomized set, models are compared only on results recording the same set-file digest, and on a start only
  where both placements record the same integration-state digest; other starts are reported as differing, and the
  number of starts compared is reported beside every comparison.
  VLA-JEPA's runner records both digests too, but its project is not under version control, so its runs are outside
  DEF-single-operator-lab: beside, never evidence.

## E1 -- evidence (to be fixed here before its first model is trained)

Seeds 1, 2 and 3, one sighted and one blind model each; procedure, configuration, measure and runners written here
before any of its models is trained (items 1-3). A change made after P0's results is allowed (E1 is a new comparison) and is
written here with its reason.
