# Ledger reviews

## remit_2026-09-24.json

stamp-monitor's workflow check found the consistency ledger's reclassifications one-sided: on
2026-09-23 the agent marked 27 trials invalid as "outside the verifier's remit" (they read the
implementation), and all 27 were adverse (18 falsified, 9 gaps) while sound trials of the same kind
were left standing.

On the user's instruction the rule was applied evenly. One reviewer classified every trial that
mentions code in any form (168 of 414), whatever its outcome and whether or not already amended,
against the ledger's own rule -- a verifier does not read the implementation and does not run it:

- OUT_OF_REMIT: the verdict rests on the implementation (file:line, what a function does) or on
  numbers measured by running the code or the simulator.
- IN_REMIT: the verdict follows from the declared statements (a file named only because a premise
  names it, or a gap that names the missing premise, is in remit).

Result: 80 more trials marked invalid (31 sound, 28 falsified, 21 gaps) and 5 of the 27 restored
(TRL-0067, 0068, 0069, 0073, 0075: declaration-level arguments). Each amendment in
.consistency/evidence.jsonl cites this file.
