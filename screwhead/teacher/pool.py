"""The search's rollouts, evaluated in parallel processes.

Ninety-six plans are simulated for every step executed, and each is independent of the others: it
restores the same saved state, forks the same watch, and reports a key. That is the whole of the
search's cost, and it parallelises exactly, because BRN-execution-state-restore's measured
property includes restores across processes -- a worker that restores the state computes the same
rollout the parent would have.

Determinism is kept by leaving the parent in charge of everything that is not a rollout: the plans
are sampled there from the seeded generator, the ranking and the distribution update happen there,
and a worker only answers "this plan, from this state, gives this key". A pool of one and a pool
of eight therefore choose the same action, which is what the smoke check asserts.

What crosses between processes is small: the integration state and the model's body poses once per
control step, the plans, and a key and six numbers back. The environments themselves are built
once, when the pool starts, because building one costs tens of seconds and a rollout costs
milliseconds.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..sim import exec_state
from .search import Plan, Rollout, Search, Settings


@dataclass
class Job:
    """One control step's work for one worker: where to start, and which plans to try."""

    state: Any                      # exec_state.ExecState, the state every plan starts from
    watch: dict                     # the carried watch, as _watch_fields sees it
    start_reference: dict
    previous_reference: dict
    plans: list[tuple[int, np.ndarray, np.ndarray]]      # (index, twist, level)


def _watch_fields(watch) -> dict:
    """The watch's carried state, as plain containers a process boundary can take."""
    return {"touching": dict(watch.touching), "open": dict(watch.open), "count": watch.count,
            "armed": watch.armed, "supported": set(watch.supported),
            "next": watch._next, "read_at_end": watch._read_at_end}


def _watch_from(verdicts, fields: dict):
    """Rebuild a watch in the worker, carrying exactly what the parent's watch carried."""
    w = verdicts.watch()
    w.touching, w.open, w.count = dict(fields["touching"]), dict(fields["open"]), fields["count"]
    w.armed, w.supported = fields["armed"], set(fields["supported"])
    w._next, w._read_at_end, w.violation = fields["next"], fields["read_at_end"], ""
    return w


def _serve(suite: str, task: int, execution, settings: Settings, start, init, inbox, outbox) -> None:
    """One worker: build the environment once, then answer jobs until the pool closes."""
    from ..sim.task_env import TaskEnv
    from .task_loss import TaskLoss
    from .verdicts import Verdicts

    env = TaskEnv(suite, task, seed=0, render=False, start=start, execution=execution)
    env.reset(init)
    loss = TaskLoss(env)
    verdicts = Verdicts(env, loss)
    search = Search(env, verdicts, loss, settings)
    periods = settings.periods()
    outbox.put("ready")

    while True:
        job = inbox.get()
        if job is None:
            return
        out = []
        for index, twist, level in job.plans:
            watch = _watch_from(verdicts, job.watch)
            roll = search._rollout(job.state, watch, Plan(twist, level), periods,
                                   job.start_reference, job.previous_reference)
            out.append((index, search._key(roll), roll))
        outbox.put(out)


class RolloutPool:
    """Workers that evaluate a control step's plans, built once and reused for the episode."""

    #: A worker builds one environment before it can answer anything, and that is tens of seconds.
    #: Past it, a worker that has not spoken is a worker that died.
    STARTUP_S = 300.0

    #: A share of one step's plans, on a loaded machine. Past it, the worker is gone.
    ANSWER_S = 900.0

    def __init__(self, suite: str, task: int, execution, settings: Settings, start, init,
                 workers: int = 6):
        ctx = mp.get_context("spawn")           # a forked MuJoCo context is not safe to reuse
        self.workers = []
        self.settings = settings
        for _ in range(workers):
            inbox, outbox = ctx.Queue(), ctx.Queue()
            p = ctx.Process(target=_serve, daemon=True,
                            args=(suite, task, execution, settings, start, init, inbox, outbox))
            p.start()
            self.workers.append((p, inbox, outbox))
        self._await_ready()

    def _await_ready(self) -> None:
        """Wait for every worker to have built its environment, or say which one did not.

        Spawn re-imports the caller's main module, so a caller without an `if __name__` guard
        starts its whole program again in each worker; multiprocessing kills those children, and a
        bare `get()` then waits on them for as long as the machine is up. Reporting the dead worker
        is the difference between a bug and an afternoon.
        """
        for p, _inbox, outbox in self.workers:
            try:
                reply = outbox.get(timeout=self.STARTUP_S)
            except queue.Empty:
                reply = None
            if reply != "ready":
                self.close()
                raise RuntimeError(
                    f"rollout worker {p.pid} did not start (exit {p.exitcode}, said {reply!r}). "
                    "A caller that builds a pool at module scope is the usual cause: spawn "
                    "re-imports the main module, so the pool must be built under `if __name__`.")

    def evaluate(self, saved, watch, plans: list[Plan], start_reference: dict,
                 previous_reference: dict) -> list[tuple[tuple, Rollout]]:
        """Every plan's key and rollout, in the order the plans were given."""
        fields = _watch_fields(watch)
        shares: list[list[tuple[int, np.ndarray, np.ndarray]]] = [[] for _ in self.workers]
        for i, plan in enumerate(plans):
            shares[i % len(self.workers)].append((i, plan.twist, plan.level))

        for (_p, inbox, _outbox), share in zip(self.workers, shares, strict=True):
            inbox.put(Job(saved, fields, start_reference, previous_reference, share))
        results: list[tuple[tuple, Rollout] | None] = [None] * len(plans)
        for p, _inbox, outbox in self.workers:
            try:
                answer = outbox.get(timeout=self.ANSWER_S)
            except queue.Empty:
                self.close()
                raise RuntimeError(f"rollout worker {p.pid} stopped answering "
                                   f"(exit {p.exitcode}); the step has no ranking") from None
            for index, key, roll in answer:
                results[index] = (key, roll)
        if any(r is None for r in results):
            raise RuntimeError("a plan came back unevaluated; the ranking would not be the "
                               "search's own")
        return [r for r in results if r is not None]

    def close(self) -> None:
        for _p, inbox, _outbox in self.workers:
            inbox.put(None)
        for p, _inbox, _outbox in self.workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()

    def __enter__(self) -> RolloutPool:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def saved_state(env):
    """The state a control step's plans all start from."""
    return exec_state.save(env)
