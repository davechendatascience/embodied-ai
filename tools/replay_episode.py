#!/usr/bin/env python
"""Play a recorded episode again, and optionally film it.

  replay_episode.py runs/teacher/goal7_r0/*.episode.json --check
  replay_episode.py runs/teacher/goal7_r0/*.episode.json --video videos/teacher

A record holds the task, the reset and the actions and nothing else (screwhead/sim/episode_record.py),
so replaying it is the only way to know it is a record of anything: --check replays and compares
the outcome with what was written. --video replays with rendering on and writes an mp4 per record,
agent view beside the wrist camera, with the step, the language and the recorded verdict burned in.

Rendering is the only reason this costs more than the original episode: the search is not run
again -- the actions are already known.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screwhead.sim.episode_record import Episode  # noqa: E402

RES = 256


def _frame(env, text: str, note: str) -> np.ndarray:
    import cv2
    sim = env.env.env.sim if hasattr(env.env, "env") else env.env.sim
    img = np.ascontiguousarray(np.concatenate(
        [sim.render(camera_name="agentview", width=RES, height=RES)[::-1],
         sim.render(camera_name="robot0_eye_in_hand", width=RES, height=RES)[::-1]], 1))
    cv2.putText(img, text[:78], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, note[:78], (6, RES - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def film(record: Episode, out_dir: Path) -> Path:
    import imageio.v2 as imageio

    frames: list[np.ndarray] = []
    language = record.task.get("language", "")
    verdict = record.outcome.get("outcome", "?")

    def on_step(t, env, success):
        frames.append(_frame(env, f"{record.task['suite']} {record.task['index']}: {language}",
                             f"step {t + 1}/{record.steps}  {verdict}"
                             f"{'  success' if success else ''}"))

    env, got = record.replay(render=True, on_step=on_step)
    tag = "settled" if record.outcome.get("settled") else record.outcome.get("outcome", "end")[:12]
    stem = (f"{record.task['suite']}_t{record.task['index']}_i{record.task['init_index']}"
            f"_{tag.replace(':', '').replace(' ', '_')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.mp4"
    imageio.mimsave(path, frames, fps=20, macro_block_size=1)
    print(json.dumps({"video": str(path), "frames": len(frames), **got}), flush=True)
    env.close() if hasattr(env, "close") else None
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="episode record files")
    ap.add_argument("--check", action="store_true", help="replay and compare with what was recorded")
    ap.add_argument("--video", default="", help="directory to write one mp4 per record into")
    args = ap.parse_args()

    agreed = 0
    for path in args.records:
        record = Episode.load(path)
        if args.video:
            film(record, Path(args.video))
        elif args.check:
            report = record.check()
            agreed += bool(report["agrees"])
            print(json.dumps({"record": path, **report}), flush=True)
        else:
            print(json.dumps({"record": path, "steps": record.steps, **record.outcome}))
    if args.check:
        print(json.dumps({"records": len(args.records), "replayed_as_recorded": agreed}))
        return 0 if agreed == len(args.records) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
