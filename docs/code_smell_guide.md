# Code smells: what to avoid in this repository, and how it is checked

This code grows by measurement: a run fails, a trace says why, a fix lands. That loop is
the project's strength and also how smells get in — each fix is local and correct, and
after thirty of them the same contact test exists in five places, a threshold is typed
in three, and a function has twelve branches nobody can hold in their head. This guide
names the smells a scan of the repository actually found, the rule that prevents each,
and the gate that enforces them.

**The gate:** `scripts/smell_check.sh` must exit 0 before a commit. It runs ruff
(`ruff.toml`), vulture, pylint's duplicate-code check, and reports radon complexity — all
through `uvx` with pinned versions, so neither virtualenv is touched. `third_party/` is not
ours and is not checked.

The scan that motivated this (2026-09-21, 44 files, 9.2k lines): ruff 201 findings,
vulture 47 unused names, 7 blocks duplicated across files, 12 functions of complexity
> 20 (one of 89). After the pass the gate is clean on every file: no findings, no unused
names, no duplicated blocks, the most complex function at 19 (radon) and the average 3.95.

---

## S1. The same logic written twice

**Found:** the "is the robot touching something, and what" loop over `data.contact` in 14
places across 5 files; the IK-feasibility block (solve, smallest singular value, joint
margin, collision) three times in `skills.py` alone; `task_env.py` repeating
`teacher_env.py`'s settling, start randomisation and contact test almost line for line.

**Why it matters here:** the copies drift. One contact loop skipped `dist >= 0` contacts
and another did not; one treated the table as a fixture and another did not. A fix to one
copy (the fingertip bodies `finger_joint{1,2}_tip`, found the hard way) does not reach the
others.

**Rule:** a computation written a second time gets a name and one home. Contact queries
live in `screwhead/sim/contacts.py`; reachability scoring in one method; environment plumbing
shared by both environments in one module. The gate fails on any block of six or more
similar lines in two files.

## S2. Dead code

**Found:** 47 names nothing uses — superseded helpers (`_grasp_width`, `bowl_distance`),
config fields no code reads (`SkillConfig.at_pos`, `release_steps`), attributes set and
never read.

**Rule:** delete it; git remembers. A name that *looks* unused but is read by name by
something outside the call graph (robosuite builds robots from class attributes such as
`default_mount`) goes in the `IGNORE` list in `scripts/smell_check.sh`, with the reason.

## S3. Functions too long to hold in your head

**Found:** `distill.collect` cyclomatic complexity 89, `token_data.train` 49,
`skill_eval.main` 31; twelve functions above 20.

**Rule:** complexity ≤ 15, ≤ 15 branches, ≤ 60 statements per function (ruff C901,
PLR0912, PLR0915). Split along the seams the function already has — one phase of an
episode, one stage of a pipeline, one section of a report — not into arbitrary halves.

## S4. Errors swallowed

**Found:** 17 `except Exception:` blocks, some returning a default that then looks like
data (a diagnostic printed `(KeyError)` where a number should have been, and the real bug
went unnoticed for a run).

**Rule:** catch the exception you expect, by type. A blind `except Exception` is allowed
only at a process boundary that must not die (a worker reporting a failed episode), must
record the exception's type and message, and carries `# noqa: BLE001` with the reason.

## S5. Unnamed thresholds

**Found:** 136 literal comparisons; the same 0.05 (smallest singular value), 0.15 (joint
margin) and 0.02 (alignment) typed in several places.

**Rule:** a tuning constant lives in a config dataclass (`SkillConfig`) next to the
measurement that set it; a physical constant is a named module constant; a literal used
twice must be named. The gate (PLR2004) rejects float literals in comparisons; integers
and strings are allowed.

## S6. Closures over loop variables

**Found:** 14 lambdas and inner functions reading a loop variable (bugbear B023) — they
see the variable's *last* value, not the one from their iteration.

**Rule:** bind explicitly (`lambda x, i=i: ...`) or hoist the function out of the loop.

## S7. Reaching into other code's internals

**Found:** `env.env.env.robots[0]._ref_joint_pos_indexes`, `env.env.env._get_observations`,
`sim.model._model` scattered through the teacher and the tools.

**Rule:** a private attribute of a third-party object is touched in exactly one place —
an accessor on our environment class — so that when robosuite changes, one line changes.

## S8. Hidden state in a feedback law

**Found (as a bug):** the teacher cached its grasp choice across episodes, keyed only on
the object moving more than a centimetre; a bad choice in episode 0 failed 17 more.

**Rule:** the teacher's action is a function of the current state (DAgger depends on it).
Any cache must be (a) cleared at every episode boundary and (b) re-derivable from the
state. Say in the cache's docstring what invalidates it.

## S9. Parameter creep

**Found:** 24 functions with more than 8 parameters.

**Rule:** at most 8 (PLR0913). Past that, the parameters are a concept — group them in a
dataclass (`SkillConfig`, `ProgramConfig`) and pass that.

## S10. Two implementations of one truth

**Found:** kinematics exist in torch (differentiable, for training) and NumPy (fast, for
the control loop).

**Rule:** allowed only when both are needed *and* a test holds them equal
(`tests/test_kin_np.py`). Otherwise delete one.

## S11. Comments that narrate history

**Rule:** a measurement that justifies a constant or a design choice stays next to it
("the palm bottoms out 40 mm above the tool point"). How the code came to be — what was
tried first, what got reverted — belongs in the commit message, where it is dated and
attached to the diff.

## S12. The small ones

Unused imports, variables and arguments; `zip` without `strict=` (silent truncation);
function calls as default arguments (evaluated once); assign-then-return. All caught by
the gate.

---

## Refactoring without changing behaviour

A refactor that is supposed to change nothing must be shown to change nothing. For the
teacher: record fixed-seed episodes before (`tools/skill_eval.py --trials` over tasks that
exercise every path — face grasps, rim grasps, the drawer, articulation, preconditions),
refactor, run again, and compare the per-episode phase timelines. They must be identical,
not merely equally successful. For kinematics: `tests/test_kin_np.py`. For the training
pipeline: the same command on the same cache must produce the same losses.

## Justifying an exception

`# noqa: CODE` on the line, followed by why the rule does not apply there. A `noqa`
without a reason is itself a finding in review.
