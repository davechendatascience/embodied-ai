"""An episode as the few numbers needed to play it again: what was reset, and what was executed.

A demonstration is worth keeping for two reasons -- to train on and to look at -- and neither needs
the frames or the states. An episode replays exactly from the task, the reset it started from, the
execution options, and the actions, because everything else is a function of those: the simulator
is deterministic under the anchored reset (BRN-reset-anchors-all-execution), and every policy's
action goes through one execution path (AXM-one-execution-path). Recording the states or the video
instead would store megabytes of derived data that goes stale the moment the decode or the
simulator changes, while the actions stay exactly what was commanded -- and a replay checks that
claim rather than assuming it.

The record names its own action space rather than leaving it implied, because the actions are
normalized body twists and a gripper level, not joint angles: what they mean on an arm depends on
that arm's normalization and tool frame. A record made here can therefore be read somewhere else,
and a backend this module does not know is an error rather than a silent mis-replay.

  record = Episode.of(env, actions, task=..., outcome=...)   # written by the collector
  env, replayed = record.replay()                            # exact, or it says where it diverged
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "screwhead.episode/1"


@dataclass
class Episode:
    task: dict[str, Any]              # backend, and what identifies the task and its initial state
    reset: dict[str, Any]             # seed, start-pose noise, horizon: what reset was called with
    embodiment: dict[str, Any]        # the arm and the frame its twists are expressed in
    action_space: dict[str, Any]      # how a row of `actions` is to be read
    execution: dict[str, Any]         # the execution options the actions were executed under
    actions: list[list[float]]        # one row per control step, exactly as executed
    outcome: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    @property
    def steps(self) -> int:
        return len(self.actions)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=1))
        return p

    @classmethod
    def load(cls, path: str | Path) -> Episode:
        blob = json.loads(Path(path).read_text())
        if blob.get("schema") != SCHEMA:
            raise SystemExit(f"{path}: schema {blob.get('schema')!r}, this build reads {SCHEMA!r}")
        return cls(**blob)

    @classmethod
    def of(cls, env, actions, *, suite: str, task: int, init: int, seed: int, noise,
           outcome: dict[str, Any], provenance: dict[str, Any]) -> Episode:
        """The record of an episode just executed in `env`, from the environment's own settings."""
        ex, spec = env.execution, env.spec
        return cls(
            task={"backend": "libero", "suite": suite, "index": task, "init_index": init,
                  "language": getattr(env, "language", "")},
            reset={"seed": seed, "start_noise": noise.as_dict(), "horizon": env.horizon},
            embodiment={"arm": env.chain.name, "joints": env.chain.n,
                        "tool_frame": env.chain.tool_frame, "source": env.chain.source},
            action_space={"kind": "normalized_body_twist+gripper", "moment_first": True,
                          "twist_bounds": [-1.0, 1.0], "control_hz": spec.control_hz,
                          "pos_scale": spec.pos_scale, "rot_scale": spec.rot_scale,
                          "gripper": "snap level, decoded by the execution's gripper mode"},
            execution=asdict(ex),          # whole, not the interesting three: a replay under
            #                                  another gripper mode or ramp is a different episode
            actions=[[round(float(x), 9) for x in a] for a in actions],
            outcome=dict(outcome), provenance=dict(provenance),
        )

    def replay(self, *, render: bool = False, on_step=None):
        """Execute the recorded actions from the recorded reset and report what happened.

        Returns (env, result). The result carries the outcome measured this time, so a caller can
        compare it with `self.outcome`: a record that does not replay is a record of nothing.
        """
        if self.task.get("backend") != "libero":
            raise SystemExit(f"backend {self.task.get('backend')!r} has no replay here; "
                             "this module knows libero, and refuses to guess at another")
        from .sim_arm import Execution
        from .task_env import StartNoise, TaskEnv

        noise = StartNoise(**self.reset["start_noise"])
        env = TaskEnv(self.task["suite"], self.task["index"], horizon=self.reset["horizon"],
                      seed=self.reset["seed"], render=render, start=noise,
                      execution=Execution(**self.execution))
        env.reset(self.task["init_index"])
        success = False
        for t, action in enumerate(self.actions):
            env.execute(np.asarray(action, float))
            success = env.success()
            if on_step is not None:
                on_step(t, env, success)
        return env, {"steps": len(self.actions), "success": bool(success), "init": env.init_index}

    def check(self) -> dict[str, Any]:
        """Replay and compare with what was recorded, so the record is evidence and not a claim."""
        _env, got = self.replay()
        want = self.outcome
        return {"replayed": got, "recorded": {k: want.get(k) for k in ("steps", "settled", "success")},
                "agrees": got["steps"] == want.get("steps")
                and got["success"] == bool(want.get("success", want.get("settled")))}
