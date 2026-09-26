"""Writing and reading the poses of the fixtures a LIBERO reset draws -- the scene a stored start needs besides its
state (AXM-libero-resamples-fixtures)."""
from __future__ import annotations

import numpy as np


def drawn_fixtures(m) -> list[int]:
    """The bodies a reset draws poses for: scene objects' root bodies (<object>_main) fixed to the world."""
    return [b for b in range(1, m.nbody) if int(m.body_parentid[b]) == 0 and int(m.body_jntnum[b]) == 0
            and m.body(b).name.endswith("_main")]


def read_fixtures(m) -> dict:
    """{body name: (position, quaternion)} of the drawn fixtures, as they stand."""
    return {m.body(b).name: (m.body_pos[b].copy(), m.body_quat[b].copy()) for b in drawn_fixtures(m)}


def write_fixtures(m, fixtures: dict) -> None:
    for name, (pos, quat) in fixtures.items():
        b = m.body(name).id
        m.body_pos[b] = np.asarray(pos, float)
        m.body_quat[b] = np.asarray(quat, float)


STEP_CHECK = (100, 500)           # physics steps after which a stepping reference is digested


def stepping_digests(env, state: np.ndarray, fixtures: dict) -> list[str]:
    """From a placed start, fixed actuator inputs (seeded) for the longest of STEP_CHECK physics steps; the digest of
    positions and velocities after each of STEP_CHECK (BRN-random-starts-test-set's admission check)."""
    import hashlib

    import mujoco
    env.place_stored(state, fixtures)
    m, d = env.env.sim.model._model, env.env.sim.data._data
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    u = np.random.default_rng(0).uniform(-1, 1, (max(STEP_CHECK), m.nu)) * 0.5 * (hi - lo)
    out = []
    for i in range(max(STEP_CHECK)):
        d.ctrl[:] = u[i]
        mujoco.mj_step(m, d)
        if i + 1 in STEP_CHECK:
            out.append(hashlib.sha1(np.concatenate([d.qpos, d.qvel]).tobytes()).hexdigest()[:16])
    return out


def load_starts(path) -> dict:
    """{task: {"states": [...], "fixtures": [...], "references": {...}}} from a randomized set's file."""
    import json
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    by_task = {t["task"]: t for t in meta["tasks"]}
    out = {}
    for task, t in by_task.items():
        states = z["states"][z["tasks"] == task]
        fixtures = [{n: (np.asarray(p, float), np.asarray(q, float)) for n, (p, q) in f.items()} for f in t["fixtures"]]
        out[task] = {"states": states, "fixtures": fixtures, "references": t.get("references")}
    return out
