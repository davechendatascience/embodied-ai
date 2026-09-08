#!/usr/bin/env python
"""Turn runs/*.json into ingest records for the component-belief ledger.

The sweep produces rollouts outside run_test, so they reach the ledger through
`ingest`, which is what that channel is for: belief-eligible evidence produced
elsewhere, carrying its source and the artifact it came from. It is not the
`note` channel -- these are measurements, not testimony.

Which contract a cell feeds is decided here, once, rather than at the call site:

  screwhead / source          CTR-libero-source-arm       success == true
  screwhead / <swapped>       CTR-heldout-embodiment      success == true
  baseline  / source          CTR-baseline-works-on-source success == true
  baseline  / <swapped>       CTR-baseline-fails-swap     success == FALSE

That last one is deliberately inverted: the premise is that a padded-vector
policy does NOT survive the swap, so the contract is supported when it fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONTRACT = {
    ("screwhead", False): ("CTR-libero-source-arm", "TST-libero-source"),
    ("screwhead", True): ("CTR-heldout-embodiment", "TST-libero-heldout"),
    ("baseline", False): ("CTR-baseline-works-on-source", "TST-baseline-source"),
    ("baseline", True): ("CTR-baseline-fails-swap", "TST-baseline-heldout"),
}


def records_for(path: Path) -> list[dict]:
    trials = json.loads(path.read_text())["trials"]
    out = []
    for t in trials:
        c = t["conditions"]
        swapped = bool(c["arm_swapped"] or c["gripper_swapped"])
        contract, test = CONTRACT[(c["policy_revision"], swapped)]
        out.append({
            "contract_id": contract,
            "test_id": test,
            "outcome": "pass",                 # the trial ran; the metric carries the result
            "metrics": t["metrics"],
            "conditions": c,
            "repro": t["repro"],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="runs/evidence")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    for r in args.runs:
        p = Path(r)
        recs = records_for(p)
        dest = outdir / f"{p.stem}.records.json"
        dest.write_text(json.dumps({"source": f"rollout:{p.stem}",
                                    "artifact_uri": str(p),
                                    "records": recs}, indent=2))
        by = {}
        for x in recs:
            by[x["contract_id"]] = by.get(x["contract_id"], 0) + 1
        print(f"{p.name:32s} -> {dest}  {by}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
