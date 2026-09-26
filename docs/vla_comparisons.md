# Panda VLA comparisons, fixed before evaluation

BRN-vla-reported-beside-a-blind-twin requires every comparison's measure, training procedure and configuration,
seeds, evaluation seed and execution configuration to be fixed before any of its models is evaluated, one sighted
and one blind model per seed, and every comparison reported whether or not it credits vision. They are fixed here,
committed before evaluation.

## P0 -- pilot (declared 2026-09-26; unreplicated by design, no vision claim)

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
- **Evaluation:** code revision b28922c, `tools/eval_vla.py` with evaluation seed 555, `VLA_EXECUTION` (the twist
  decode, cameras without multisampling), every episode twice in fresh processes; on LIBERO's starts
  (`--episodes 50`) and on the randomized set (`--starts ...`). Changed from 8582a63 before any model was
  evaluated: at 8582a63 a task not admitted or an episode whose model differed crashed the run after both rounds
  (fac96f7 reports them as not scored), and the randomized set's file was not among the data files a run records
  (29fb366), placements recorded no digest of the integration state (705c71f), and records did not name the host
  or GPU model (526b380; P0's training records, made at 8582a63, do not either), nor the training run an evaluation
  loads and its launch arguments (b28922c). Episode execution is unchanged. P0's runners are scratch scripts, not
  kept in the repository: another reason P0 is outside DEF-single-operator-lab and never evidence. E1's runners are
  committed with its plan.
  On the randomized set, models are compared only on results recording the same set-file digest, and on a start only
  where both placements record the same integration-state digest; other starts are reported as differing, and the
  number of starts compared is reported beside every comparison.
  VLA-JEPA's runner records both digests too, but its project is not under version control, so its runs are outside
  DEF-single-operator-lab: beside, never evidence.

## E1 -- evidence (to be fixed here before its first evaluation)

Seeds 1, 2 and 3, one sighted and one blind model each; procedure, configuration and measure written here before
any of its models is evaluated. A change made after P0's results is allowed (E1 is a new comparison) and is
written here with its reason.
