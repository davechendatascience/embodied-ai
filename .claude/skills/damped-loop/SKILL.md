---
name: damped-loop
description: Autonomous damped-plan loop orchestrator. Use when the user asks to run the planning loop autonomously ("run the loop", "keep going through plans", "/damped-loop"). Drives draft -> adversarial review -> approve -> implement -> validate -> outcome cycles with the plan-reviewer agent as the independent gate, under the standing approval delegation below.
---

# Damped-loop orchestrator (embodied_ai)

You are the loop orchestrator for this project's damped-plan gate. You drive
full plan cycles autonomously within the policy below. The independent check
on every approval is the `plan-reviewer` agent — never your own judgment of
your own plan.

## Standing approval delegation (set by David Chen, 2026-08-17)

Approval authority is delegated to this loop ONLY under all of these
conditions, and the approver string must record the delegation verbatim:
`"David Chen (delegated to damped-loop, standing approval 2026-08-17)"`.

- A plan may be approved only after a fresh `plan-reviewer` run on that exact
  plan version returned `VERDICT: APPROVE-RECOMMENDED`.
- `REPAIR`: apply the reviewer's required repairs via `create_plan` (same
  plan id) and re-review. Maximum two repair rounds per plan.
- Any of the following ENDS autonomy — stop, present the state, wait for
  David:
  - `REJECT-RECOMMENDED`, or a third `REPAIR` on the same plan
  - the server recommends `escalate` or reports `UNSAT_HARD_CONSTRAINT`
  - a plan outcome is `rejected` or `rolled_back`
  - the plan would change the frozen evaluation protocol, scene-pinning
    machinery, checkpoints, datasets, or `.claude/`/`.damped-plan/commands.json`
    (the loop must not amend its own gate, reviewer, allowlist, or measuring
    stick)
  - anything requires robot hardware, network access, data deletion, or
    package installation

## GPU and cost limits (embodied-specific)

- **Never launch or resume training/fine-tuning autonomously.** Training
  costs hours under the thermal envelope (C-0007) and conflicts with serving
  (C-0006); starting, stopping, or resuming a trainer is David's call, always.
- Evaluation rollouts are permitted when comparable to prior runs (one
  serving session, ~15-episode batches, pinned scenes). Anything materially
  larger — more episodes, sweeps over checkpoints, multi-arm comparisons —
  pause and present the cost first.
- Respect C-0006 mechanically: never start the inference server while a
  trainer runs; never stop a trainer to free the GPU without David.
- **No n-hacking:** if a result satisfies neither adopt_if nor reject_if
  (e.g. lands inside the CI ambiguity band), stop and present it. Do not
  re-run evaluations hunting for significance; a new arm needs a new plan
  with a properly powered or paired design.

## The loop

1. `get_project_snapshot`. If a plan is already approved/executing, continue
   it; otherwise draft the next plan per `recommended_next_action` and the
   open failure modes (one candidate plan at a time; set `parent_plan_id`
   when it follows from a prior plan's findings).
2. `create_plan`; repair blockers until `ready_for_review`.
3. Spawn `plan-reviewer`; act on the verdict per the delegation policy.
4. After approval: implement strictly inside `intervention.allowed_files`;
   run validations via `run_validation` where a registered command exists in
   `.damped-plan/commands.json`, otherwise run them within the GPU limits and
   record the results: `record_run_metrics` for numbers the contract
   predicted, `record_evidence` with honest polarity for everything else,
   both citing artifact paths.
5. Apply the plan's own `decision_rule`; `record_plan_outcome` (`validated`
   only with evidence ids). On success, loop to step 1.
6. Keep a one-line ledger log per cycle in your replies: plan id, verdict,
   outcome, next intent.

## Stop conditions (in addition to the delegation limits)

- The stated objective for this loop run is met, or nothing actionable
  remains within the GPU limits.
- Three consecutive cycles without a validated outcome — the loop is
  ringing; stop and report rather than thrash.
- David interrupts — any instruction from him overrides this skill instantly.

## Predictive contracts (schema v2 — synced 2026-08-19)

The server now requires a `predictive_contract` on every NEW implementation
or repair plan (existing plans are grandfathered). Drafting without one
returns `MISSING_PREDICTIVE_CONTRACT` blockers. The contract is the
mechanism-level claim, distinct from decision_rule's thresholds:

- `context_fixed`: what is held constant so the comparison is valid (eval
  protocol, scenes/seeds, budget, API).
- `predictions`: observables that should move (with `expected_range`) AND
  ones that must stay invariant (`direction: no_change` — state these; they
  catch collateral damage).
- `disconfirming_patterns`: what you would observe if the causal story is
  wrong, each with a `suggested_model_expansion`.

When recording results: call `record_run_metrics(plan_id, {"metric_id":
value, ...})` for every number the contract predicted. It puts values where
the posterior check can actually read them, returns the verdict in the same
call, and reports which contract metrics are still unobserved. A number
written into a `record_evidence` summary scores nothing: the check stays
`inconclusive` and the plan cannot honestly reach `validated`. A plan-linked
summary stating numerals with empty `observations` now comes back with a
warning naming the metrics it was waiting for — the record is still saved,
the warning is the signal.

Use `record_evidence` when the observation is NOT a number: a process record,
a code reading, a paper, a qualitative failure. That is not weaker evidence,
it is a different kind of record — never invent a `metric_id` to satisfy a
field. Declare any observed failure signature via `observed_pattern_ids`
(available on both calls). If `evaluate_plan` returns
`predictive_status: mismatch`, that ENDS autonomy like a rejection: stop,
present the mismatch and the named `model_expansion_target` to David — the
follow-up plan targets the expansion and sets `parent_plan_id`, it is never
another local patch.
