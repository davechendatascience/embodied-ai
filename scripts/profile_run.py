"""Where a ReKep run actually spends its time.

Wall-clock alone cannot tell you whether this approach is viable, because the
costs are structurally different and only some of them scale with task length:

  * CONSTRAINT GENERATION is a one-off VLM call per task. It looks expensive
    (seconds) and is actually free, because it is amortised over the whole
    episode -- and could be cached to disk entirely.
  * GRASP PROPOSAL is once per grasp stage. Contact-GraspNet inference is
    ~0.6 s, but aiming the wrist first MOVES THE ARM, which costs sim steps and
    is easy to miss.
  * SOLVING is per control cycle and is the only cost that scales. It has two
    wildly different paths: a cold `dual_annealing` solve and a warm local
    SLSQP one. Measured on this port: 0.75 s vs 0.029 s -- a 26x gap. Upstream's
    "real-time 10 Hz" claim is about the warm path only.
  * SIM STEPPING is the rest, and on a real robot it is replaced by real time.

The number that matters for viability is therefore the WARM solve rate and how
often the loop is forced back onto the cold path -- not the total.

    .venv/bin/python scripts/profile_run.py <run.log>
"""

import re
import sys
from collections import defaultdict


def parse(path):
    solves = []          # (type, from_scratch, seconds)
    pending = {}
    wall = steps = None
    stages = grasps = backtracks = 0

    for line in open(path, errors="replace"):
        line = line.strip()
        if m := re.search(r"# solve_time\s*:\s*([\d.]+)", line):
            pending["t"] = float(m.group(1))
        elif m := re.search(r"# from_scratch\s*:\s*([\d.]+)", line):
            pending["cold"] = float(m.group(1)) > 0.5
        elif m := re.search(r"# type\s*:\s*(\w+)", line):
            if "t" in pending:
                solves.append((m.group(1), pending.get("cold", False), pending["t"]))
            pending = {}
        elif m := re.search(r"wall\s*:\s*([\d.]+)s", line):
            wall = float(m.group(1))
        elif m := re.search(r"steps\s*:\s*(\d+)", line):
            steps = int(m.group(1))
        elif "via contact-graspnet" in line:
            grasps += 1
        elif "backtrack to stage" in line:
            backtracks += 1
        elif re.search(r"\[stage=\d+\]", line):
            stages += 1
    return solves, wall, steps, grasps, backtracks


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    solves, wall, steps, grasps, backtracks = parse(sys.argv[1])
    if not solves:
        print("no solver telemetry found — was the full log captured?")
        return 1

    buckets = defaultdict(list)
    for kind, cold, t in solves:
        buckets[(kind, cold)].append(t)

    print(f"{'solver':16s} {'path':6s} {'n':>4s} {'total':>8s} {'mean':>8s} {'rate':>9s}")
    total = 0.0
    for (kind, cold), ts in sorted(buckets.items()):
        mean = sum(ts) / len(ts)
        total += sum(ts)
        print(f"{kind:16s} {'cold' if cold else 'warm':6s} {len(ts):4d} "
              f"{sum(ts):7.2f}s {mean:7.3f}s {1 / mean:8.1f}Hz")

    print(f"\nsolver total     : {total:.1f}s")
    if wall:
        print(f"wall             : {wall:.1f}s  ({total / wall * 100:.0f}% in the solver)")
        print(f"unaccounted      : {wall - total:.1f}s  (sim stepping, rendering, "
              f"grasp inference, VLM)")
    if steps:
        print(f"sim steps        : {steps}")
    print(f"grasp proposals  : {grasps}")
    print(f"backtracks       : {backtracks}")

    cold = sum(len(v) for (k, c), v in buckets.items() if c)
    warm = sum(len(v) for (k, c), v in buckets.items() if not c)
    if cold + warm:
        print(f"cold solves      : {cold}/{cold + warm} ({cold / (cold + warm) * 100:.0f}%)")
        print("\nEvery backtrack sets first_iter=True, forcing a COLD solve. That is the")
        print("single biggest lever on wall-clock: the cold path is ~26x the warm one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
