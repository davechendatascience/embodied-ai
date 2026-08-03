"""Does the cross-venv bridge work, and does it give the SAME answer?

Two questions, and the second is the one that matters. A bridge that responds
but disagrees with the in-process path is worse than no bridge: it would put a
silent divergence directly underneath the planner, where every wrong action
would look like a modelling problem.

So this runs the recorded episode's own action plus counterfactuals through
the socket, and checks the resulting cost ordering against what
`rank_actions_pointworld.py` measures in-process. Same control discipline that
caught the double-centring bug (`NOTES.md` section 4).

Runs in `.venv` -- numpy 1.26, mujoco. The service runs in `.venv-pw` --
numpy 2.5, torch 2.11 -- and this file does NOT start it: the two stacks must
not know about each other, so lifecycle belongs to
`scripts/run_bridge_test.sh`. That numpy gap is also the point of the wire
format, which is `.npy` inside a zip rather than pickle.

    scripts/run_bridge_test.sh
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pointworld_bridge.client import PointWorldClient  # noqa: E402
from pointworld_bridge.protocol import DEFAULT_SOCKET  # noqa: E402

EPISODE = ROOT / "data" / "pw_episodes" / "libero_goal_0_ep0.npz"
MOVED_THRESHOLD = 0.002


def rigid(robot0, direction, per_step_mm, T):
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else d
    steps = np.arange(T, dtype=np.float32)[:, None, None]
    return robot0[None] + steps * (d * per_step_mm / 1000.0)[None, None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    cli = ap.parse_args()

    ep = {k: v for k, v in np.load(EPISODE, allow_pickle=True).items()}
    scene = ep["scene_flows"][0]                      # (T,Ns,3) world
    T = scene.shape[0]
    moved = np.linalg.norm(scene[-1] - scene[0], axis=1) > MOVED_THRESHOLD
    goal_idx = np.flatnonzero(moved)
    goal_pos = scene[-1][moved]                       # where they really ended up
    robot0 = ep["robot_flows"][0, 0]

    print(f"client  : numpy {np.__version__}")
    print(f"episode : {T} steps, {scene.shape[1]} scene pts, "
          f"{len(goal_idx)} goal pts\n")

    if True:
        with PointWorldClient(cli.socket) as pw:
            info = pw.ping()
            print(f"          torch {info['torch']}, numpy {info['numpy']}, "
                  f"{info['device']}")
            if info["numpy"].split(".")[0] == np.__version__.split(".")[0]:
                print("          NOTE: both sides share a numpy major version; the "
                      "portability claim is untested here")
            else:
                print(f"          numpy {np.__version__} <-> {info['numpy']} across "
                      f"the socket, as intended")

            # The first call pays cuDNN algorithm selection and kernel autotune.
            # A control loop pays it once, so quoting it as the per-observation
            # cost would overstate the loop's budget by ~20x -- report both.
            for label in ("cold", "warm"):
                t0 = time.perf_counter()
                head = pw.observe(ep)
                rt = time.perf_counter() - t0
                print(f"observe : {rt*1e3:6.1f} ms round trip ({label:4s}) "
                      f"-- assemble {head['assemble_ms']:5.1f} + encode "
                      f"{head['encode_ms']:6.1f} ms")

            # ---- the control: the recorded action, through the socket -------
            t0 = time.perf_counter()
            h, out = pw.rollout(ep["robot_flows"][0][None], goal_idx, goal_pos)
            rt_one = time.perf_counter() - t0
            recorded_cost = float(out["cost"][0])
            print(f"rollout : {rt_one*1e3:6.1f} ms round trip for K=1 "
                  f"(model {h['model_ms']:.1f}, assemble {h['assemble_ms']:.1f})")

            # ---- is batching SAFE? ------------------------------------------
            # PTv3 serializes the whole batch into one space-filling-curve
            # sequence and then cuts it into attention patches. If a patch can
            # straddle a batch boundary, candidates contaminate each other and
            # every planner score is subtly wrong -- the kind of defect that
            # looks like a bad model forever. Identical inputs in one batch
            # must produce identical outputs.
            same = np.repeat(ep["robot_flows"][0][None], 8, axis=0)
            _, out_same = pw.rollout(same, goal_idx, goal_pos)
            spread_batched = float(out_same["cost"].max() - out_same["cost"].min())

            # The control. Contamination is not the only way identical inputs
            # produce different answers: spconv and scatter reduce with atomics,
            # whose summation order is not fixed, and PTv3 redraws its
            # serialization-order permutation every forward. Both make a SINGLE
            # candidate irreproducible on its own. So run the same candidate
            # eight times, each in its own batch, and compare the spreads. Only
            # an excess over that baseline is contamination.
            serial = [float(pw.rollout(ep["robot_flows"][0][None],
                                       goal_idx, goal_pos)[1]["cost"][0])
                      for _ in range(8)]
            spread_serial = max(serial) - min(serial)

            print(f"\nbatch   : 8 identical candidates, one batch  -> spread "
                  f"{spread_batched*1000:6.3f} mm")
            print(f"          8 identical candidates, run apart -> spread "
                  f"{spread_serial*1000:6.3f} mm  (control)")
            # Batching is safe if it adds no scatter beyond what one candidate
            # already has by itself.
            batch_clean = spread_batched <= max(spread_serial * 2.0, 1e-6)
            if batch_clean:
                print("          batching adds no scatter beyond the model's own "
                      "non-determinism -- safe to batch")
            else:
                print("          batched spread EXCEEDS the serial baseline -- "
                      "candidates are contaminating each other")

            # ---- the candidates, batched ------------------------------------
            names, cands = ["RECORDED"], [ep["robot_flows"][0]]
            for label, d in [("+Y", (0, 1, 0)), ("-Y", (0, -1, 0)),
                             ("+X", (1, 0, 0)), ("-X", (-1, 0, 0)),
                             ("+Z", (0, 0, 1)), ("-Z", (0, 0, -1))]:
                for rate in (6.5, 13.0, 26.0):
                    names.append(f"{label} @ {rate:4.1f} mm/step")
                    cands.append(rigid(robot0, d, rate, T))
            names.append("still")
            cands.append(rigid(robot0, (0, 0, 0), 0.0, T))
            batch = np.stack(cands)

            t0 = time.perf_counter()
            h, out = pw.rollout(batch, goal_idx, goal_pos)
            rt_k = time.perf_counter() - t0
            cost = out["cost"]
            K = len(batch)
            print(f"        : {rt_k*1e3:6.1f} ms round trip for K={K} "
                  f"({h['per_candidate_ms']:.1f} ms/candidate in the model, "
                  f"{rt_k*1e3/K:.1f} ms/candidate end to end)")
            print(f"        : batching speedup vs K=1 serial: "
                  f"{(rt_one * K) / rt_k:.1f}x")

            order = np.argsort(cost)
            print(f"\n{'rank':>4s}  {'candidate':22s} {'cost (mm)':>10s}")
            for r, i in enumerate(order[:6], 1):
                print(f"{r:4d}  {names[i]:22s} {cost[i]*1000:10.1f}")
            print(f"{'':4s}  {'...':22s}")
            for r, i in enumerate(order[-2:], len(order) - 1):
                print(f"{r:4d}  {names[i]:22s} {cost[i]*1000:10.1f}")

            # ---- equivalence with the in-process result ---------------------
            # rank_actions_pointworld.py measured, in-process: the recorded
            # action ~96 mm, rigid +Y at the true 13 mm/step within a few mm of
            # it, all three +Y candidates ahead of everything else, -Y @ 26
            # worst at ~299 mm. If the bridge agrees on all of that, it is the
            # same computation and not merely a working socket.
            y13 = names.index("+Y @ 13.0 mm/step")
            gap = abs(cost[y13] - recorded_cost) * 1000
            # RECORDED *is* a +Y pull at the true rate, so it belongs with the
            # +Y family here rather than counting as a fourth direction.
            top3 = {("+Y" if names[i] == "RECORDED" else names[i].split(" @")[0])
                    for i in order[:3]}
            worst = names[order[-1]]
            print(f"\ncontrol : recorded {recorded_cost*1000:.1f} mm vs rigid +Y at the "
                  f"true rate {cost[y13]*1000:.1f} mm ({gap:.1f} mm apart)")
            print(f"          top 3 are all {sorted(top3)}, worst = {worst!r}")

            ok = gap < 25 and top3 == {"+Y"} and worst.startswith("-Y") and batch_clean
            print(f"\nverdict : {'BRIDGE AGREES with the in-process path' if ok else 'DIVERGENCE - do not build on this'}")
            return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
