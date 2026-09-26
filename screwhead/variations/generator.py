"""LIBERO-Variations' generator: a benchmark's metadata in, kept tasks out.

A task is drafted from the metadata and its seed alone (template, objects, layout, instruction, task
file), built by LIBERO from its task file, held until at rest, and kept only if the checks of
BRN-lv-scenes-valid, the preconditions of BRN-lv-tasks-doable and the naming of
BRN-lv-instructions-unambiguous all hold at rest -- otherwise the next attempt is drafted. Nothing
generated is stored: the same metadata and seed give the same task, recorded by digest.

Layout planning uses the catalog (each category's shape measured once at rest); every decision that a
claim rests on is re-measured in the built scene.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import yaml

from . import shapes, task_file
from .reach_map import ReachMap, matches

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_MOD = 2**31 - 1


# -- the benchmark ------------------------------------------------------------------------------
@dataclass
class Benchmark:
    meta: dict
    reach: ReachMap
    catalog: dict[str, shapes.Shape]
    affordances: dict

    @classmethod
    def load(cls, path: str) -> Benchmark:
        with open(path) as f:
            meta = yaml.safe_load(f)
        root = os.path.dirname(os.path.abspath(path))
        reach = ReachMap.load(os.path.join(root, meta["action_space"]["map"]), meta["action_space"]["digest"])
        cat_path = os.path.join(root, meta["catalog"]["file"])
        with open(cat_path) as f:
            raw = json.load(f)
        if catalog_digest(raw) != meta["catalog"]["digest"]:
            raise ValueError(f"{cat_path}: digest {catalog_digest(raw)}, the benchmark declares {meta['catalog']['digest']}")
        with open(os.path.join(HERE, "..", "teacher", "affordances.yaml")) as f:
            aff = yaml.safe_load(f)["categories"]
        return cls(meta=meta, reach=reach, catalog={k: shapes.Shape.from_dict(v) for k, v in raw.items()}, affordances=aff)

    @property
    def scene(self) -> dict:
        return self.meta["scene"]

    def noun(self, category: str) -> str:
        return self.meta["nouns"][category]

    def task_seed(self, split_seed: int, index: int) -> int:
        """The seed of a split's task: its draws and LIBERO's sampler draws come from it alone."""
        return int(np.random.default_rng([int(self.meta["version"]), int(split_seed), int(index)]).integers(SEED_MOD))


def generator_revision(bench_path: str) -> str:
    """The generator that runs: its code (this package) and the benchmark's metadata with the files it names."""
    with open(bench_path) as f:
        meta = yaml.safe_load(f)
    root = os.path.dirname(os.path.abspath(bench_path))
    files = sorted(os.path.join(HERE, n) for n in os.listdir(HERE) if n.endswith(".py"))
    files += [os.path.abspath(bench_path), os.path.join(root, meta["action_space"]["map"]),
              os.path.join(root, meta["catalog"]["file"])]
    h = hashlib.sha1()
    for p in files:
        with open(p, "rb") as f:
            h.update(os.path.basename(p).encode())
            h.update(f.read())
    return f"lv{meta['version']}:{h.hexdigest()[:10]}"


def catalog_digest(raw: dict) -> str:
    return hashlib.sha1(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:16]


# -- drafting -----------------------------------------------------------------------------------
@dataclass
class Draft:
    template: str
    a: str                                  # the object moved (instance)
    target: str                             # the support or container (instance)
    placements: list[task_file.Placement]
    goal: tuple
    language: str
    text: str                               # the task file

    @property
    def categories(self) -> dict[str, str]:
        return {p.name: p.category for p in self.placements}


def _radius(shape: shapes.Shape) -> float:
    """A plan radius that holds the object's collision box at any yaw."""
    return float(np.hypot(*shape.half))


def _fits(bench: Benchmark, template: str, a: shapes.Shape, t: shapes.Shape) -> bool:
    c = bench.scene["fit_clearance"]
    return shapes.fits_in(a, t, c) if bench.meta["templates"][template]["supports"] == "container" else shapes.fits_on(a, t, c)


def _candidates(bench: Benchmark, template: str) -> list[tuple[str, str]]:
    """(A, target) category pairs the template allows, by the affordance table and the catalog's fit."""
    out = []
    for t in bench.meta["templates"][template]["targets"]:
        for a in bench.meta["pool"]["pick"]:
            if a != t and _fits(bench, template, bench.catalog[a], bench.catalog[t]):
                out.append((a, t))
    return out


def _layout(bench: Benchmark, rng, cats: list[str]) -> list[tuple[np.ndarray, float]] | None:
    """A centre and yaw per object: its plan circle, grown by the region, the rest shift and the scene's
    clearance, inside the action space and clear of the others' (largest first)."""
    sc = bench.scene
    pad = sc["region_half"] + sc["rest_shift"]
    pts = bench.reach.points()
    order = sorted(range(len(cats)), key=lambda i: -_radius(bench.catalog[cats[i]]))
    placed: dict[int, tuple[np.ndarray, float]] = {}
    for i in order:
        r = _radius(bench.catalog[cats[i]]) + pad
        for _ in range(sc["layout_tries"]):
            xy = pts[rng.integers(len(pts))]
            if not bench.reach.contains(xy - r - sc["clearance"], xy + r + sc["clearance"]):
                continue
            if all(np.linalg.norm(xy - q) >= r + _radius(bench.catalog[cats[j]]) + pad + 2 * sc["clearance"]
                   for j, (q, _y) in placed.items()):
                placed[i] = (xy, float(rng.uniform(-np.pi, np.pi)))
                break
        else:
            return None
    return [placed[i] for i in range(len(cats))]


def order_words(names: list[str], xy: dict[str, np.ndarray], base: np.ndarray, gap: float) -> dict[str, str] | None:
    """BRN-lv-instructions-unambiguous's order words for the objects of one category: none for one; front/back
    along the base's x axis (nearer the base first) or else left/right along its y axis for two; front, middle,
    back along x for three -- None when the needed axis does not separate every two by the gap."""
    if len(names) == 1:
        return {names[0]: ""}
    if len(names) > 3:
        return None
    dx = {n: float(xy[n][0] - base[0]) for n in names}
    dy = {n: float(xy[n][1] - base[1]) for n in names}

    def separated(d):
        v = sorted(d.values())
        return all(b - a >= gap for a, b in zip(v, v[1:], strict=False))

    if separated(dx):
        words = ["front", "back"] if len(names) == 2 else ["front", "middle", "back"]
        return dict(zip(sorted(names, key=dx.get), words, strict=True))
    if len(names) == 2 and separated(dy):
        return dict(zip(sorted(names, key=dy.get), ["right", "left"], strict=True))
    return None


def instruction(bench: Benchmark, template: str, a: str, target: str, cats: dict[str, str],
                xy: dict[str, np.ndarray], base: np.ndarray, gap: float) -> str | None:
    """The instruction from the goal and where the objects are, or None when a named object cannot be singled
    out by order words `gap` apart."""
    def phrase(inst: str) -> str | None:
        same = [n for n, c in cats.items() if c == cats[inst]]
        words = order_words(same, xy, base, gap)
        if words is None:
            return None
        return f"{words[inst]} {bench.noun(cats[inst])}".strip()
    pa, pt = phrase(a), phrase(target)
    if pa is None or pt is None:
        return None
    return bench.meta["templates"][template]["wording"].format(a=pa, b=pt)


def goal_of(bench: Benchmark, template: str, a: str, target: str) -> tuple:
    tpl = bench.meta["templates"][template]
    return (tpl["goal"], a, f"{target}_{tpl['region']}" if tpl.get("region") else target)


def draft(bench: Benchmark, seed: int, attempt: int) -> Draft | None:
    """A task drafted from the metadata, the task's seed and the attempt alone; None when this attempt's draws
    leave no layout or no unambiguous instruction."""
    rng = np.random.default_rng([int(seed), int(attempt)])
    names = [t for t, v in bench.meta["templates"].items() if v["weight"] > 0]
    w = np.array([bench.meta["templates"][t]["weight"] for t in names], float)
    template = names[rng.choice(len(names), p=w / w.sum())]
    pairs = _candidates(bench, template)
    a_cat, t_cat = pairs[rng.integers(len(pairs))]
    sc = bench.scene
    pool = list(dict.fromkeys(bench.meta["pool"]["pick"] + bench.meta["pool"]["targets"]))
    cats = [a_cat, t_cat]
    for _ in range(int(rng.integers(sc["others"][0], sc["others"][1] + 1))):
        c = pool[rng.integers(len(pool))]
        if cats.count(c) < sc["same_category_max"]:
            cats.append(c)
    counts: dict[str, int] = {}
    insts = []
    for c in cats:
        counts[c] = counts.get(c, 0) + 1
        insts.append(task_file.instance_name(c, counts[c]))
    layout = _layout(bench, rng, cats)
    if layout is None:
        return None
    placements = [task_file.Placement(n, c, (float(xy[0]), float(xy[1])), sc["region_half"], yaw)
                  for n, c, (xy, yaw) in zip(insts, cats, layout, strict=True)]
    base = np.asarray(bench.reach.reference["base_world"][:2])
    # planned centres: the words must survive the sampler's and the settling's shifts, so plan with them added
    planned_gap = sc["order_gap"] + 2 * (sc["region_half"] + sc["rest_shift"])
    cat_of = {p.name: p.category for p in placements}
    lang = instruction(bench, template, insts[0], insts[1], cat_of, {p.name: np.asarray(p.xy) for p in placements},
                       base, planned_gap)
    if lang is None:
        return None
    goal = goal_of(bench, template, insts[0], insts[1])
    return Draft(template=template, a=insts[0], target=insts[1], placements=placements, goal=goal, language=lang,
                 text=task_file.write(lang, placements, [goal], interest=[insts[0], insts[1]]))


# -- building and keeping -------------------------------------------------------------------------
@dataclass
class Kept:
    draft: Draft
    seed: int
    attempt: int
    state: np.ndarray
    fixtures: dict
    fingerprint: str
    placed_digest: str

    @property
    def digest(self) -> str:
        """The task, for comparing two generations of it: its task file, model and placed state."""
        h = hashlib.sha1(self.draft.text.encode())
        h.update(self.fingerprint.encode())
        h.update(self.placed_digest.encode())
        return h.hexdigest()[:16]


def _touches(m, d, body: int, other: int) -> bool:
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if {b1, b2} == {body, other}:
            return True
    return False


def check_scene(bench: Benchmark, env, dr: Draft, sampled: dict[str, np.ndarray]) -> dict[str, str]:
    """BRN-lv-scenes-valid's checks at rest: {failed check: detail}, empty when the scene is kept."""
    sc = bench.scene
    m, d = env.env.sim.model._model, env.env.sim.data._data
    fails: dict[str, str] = {}
    if not matches(m, d, bench.reach.reference):
        fails["reference"] = "robot, mount, table or fixtures differ from the action space's reference"
    table = m.body("table").id
    boxes = {}
    for p in dr.placements:
        body = m.body(f"{p.name}_main").id
        pos = d.xpos[body]
        lo, hi = np.asarray(p.ranges()[:2]), np.asarray(p.ranges()[2:])
        if not ((pos[:2] >= lo).all() and (pos[:2] <= hi).all()):
            fails[f"region:{p.name}"] = f"origin {pos[:2].round(4).tolist()} outside {p.ranges()}"
        shift = float(np.linalg.norm(pos[:2] - sampled[p.name][:2]))
        if shift > sc["rest_shift"]:
            fails[f"shift:{p.name}"] = f"{shift * 1000:.1f} mm from where sampled"
        tilt = shapes.upright_angle(env.scene, p.name, bench.catalog[p.category])
        if tilt > sc["upright_deg"]:
            fails[f"upright:{p.name}"] = f"{tilt:.1f} deg"
        scene_body = env.scene.body_id(p.name)
        if not (_touches(m, d, scene_body, table) or _touches(m, d, body, table)):
            fails[f"table:{p.name}"] = "not touching the table"
        glo, ghi = shapes.plan_bounds(env.scene, p.name, grow=sc["clearance"])
        if not bench.reach.contains(glo, ghi):
            fails[f"reach:{p.name}"] = "footprint and clearance leave the action space"
        boxes[p.name] = shapes.plan_bounds(env.scene, p.name)
    names = list(boxes)
    for i, u in enumerate(names):
        for v in names[i + 1:]:
            (ul, uh), (vl, vh) = boxes[u], boxes[v]
            if (ul < vh).all() and (vl < uh).all():
                fails[f"overlap:{u}:{v}"] = "plan footprints overlap"
    return fails


def check_task(bench: Benchmark, env, dr: Draft) -> dict[str, str]:
    """BRN-lv-tasks-doable's preconditions and BRN-lv-instructions-unambiguous's naming, at rest."""
    fails: dict[str, str] = {}
    cats = dr.categories
    aff = bench.affordances
    a, t = cats[dr.a], cats[dr.target]
    if aff.get(a, {}).get("moved_by") != "pick" or aff.get(a, {}).get("held_by") in (None, "open", "none"):
        fails["affordance:a"] = f"{a}: moved_by {aff.get(a, {}).get('moved_by')}, held_by {aff.get(a, {}).get('held_by')}"
    kind = bench.meta["templates"][dr.template]["supports"]
    if aff.get(t, {}).get("supports") != kind:
        fails["affordance:target"] = f"{t}: supports {aff.get(t, {}).get('supports')}, the template needs {kind}"
    sa = shapes.measure(env.scene, dr.a)
    st = shapes.measure(env.scene, dr.target, container=kind == "container")
    if not _fits(bench, dr.template, sa, st):
        fails["fit"] = f"{a} {np.round(sa.half, 3).tolist()} in/on {t} {np.round(st.half, 3).tolist()}"
    if env.success():
        fails["goal"] = "the goal already holds"
    xy = {p.name: env.env.sim.data.body_xpos[env.env.sim.model.body_name2id(f"{p.name}_main")][:2].copy()
          for p in dr.placements}
    base = np.asarray(bench.reach.reference["base_world"][:2])
    lang = instruction(bench, dr.template, dr.a, dr.target, cats, xy, base, bench.scene["order_gap"])
    if lang != dr.language:
        fails["instruction"] = f"at rest it reads {lang!r}, the task file says {dr.language!r}"
    return fails


def build(bench: Benchmark, dr: Draft, seed: int, attempt: int, folder: str, render: bool | int = False,
          horizon: int | None = None, robot: str = "Panda"):
    """The drafted task built, drawn and checked: (env, Kept or None, failed checks)."""
    from ..sim.sim_arm import Execution
    from ..sim.task_env import TaskEnv
    from ..sim.task_env_place import read_fixtures
    path = os.path.join(folder, f"lv{bench.meta['version']}_{seed}_{attempt}.bddl")
    with open(path, "w") as f:
        f.write(dr.text)
    env = TaskEnv("variations", path, horizon=horizon or bench.meta["horizon"], seed=seed, render=render,
                  execution=Execution(robot=robot, hard_reset=True))
    sampled = env.draw_scene(attempt)
    fails = check_scene(bench, env, dr, sampled)
    fails.update(check_task(bench, env, dr))
    if fails:
        return env, None, fails
    m = env.env.sim.model._model
    state = np.asarray(env.env.sim.get_state().flatten()).copy()
    fixtures = read_fixtures(m)
    fingerprint = env.model_fingerprint()
    env.keep_model()
    if not env.place_stored(state, fixtures, fingerprint):
        return env, None, {"placement": "the placed model or fixtures differ from the kept scene's"}
    return env, Kept(draft=dr, seed=seed, attempt=attempt, state=state, fixtures=fixtures, fingerprint=fingerprint,
                     placed_digest=env.placed_digest), {}


def generate(bench: Benchmark, split_seed: int, index: int, folder: str, render: bool | int = False,
             horizon: int | None = None, robot: str = "Panda", log=None):
    """The split's task `index`: (env placed at its start, Kept, the attempts' failed checks). Raises when every
    attempt the metadata allows fails."""
    seed = bench.task_seed(split_seed, index)
    history: list[dict] = []
    for attempt in range(bench.scene["attempts"]):
        dr = draft(bench, seed, attempt)
        if dr is None:
            history.append({"attempt": attempt, "draft": "no layout or no unambiguous instruction"})
            continue
        env, kept, fails = build(bench, dr, seed, attempt, folder, render, horizon, robot)
        if kept is not None:
            return env, kept, history
        env.close()
        history.append({"attempt": attempt, **fails})
        if log:
            log(f"task {index} attempt {attempt}: {sorted(fails)}")
    raise RuntimeError(f"task {index} (seed {seed}): no attempt kept: {history}")


# -- the catalog ----------------------------------------------------------------------------------
def measure_catalog(categories: list[str], containers: set[str], folder: str, per_scene: int = 6,
                    robot: str = "Panda") -> dict[str, dict]:
    """Each category's shape at rest upright, as LIBERO stands it at yaw 0, measured in probe scenes of a few
    objects spaced along the table's far half (layout planning only: every check re-measures)."""
    from ..sim.sim_arm import Execution
    from ..sim.task_env import TaskEnv
    out: dict[str, dict] = {}
    for k in range(0, len(categories), per_scene):
        group = categories[k:k + per_scene]
        places = [task_file.Placement(task_file.instance_name(c, 1), c, (0.25 * (i % 3) - 0.25, 0.35 * (i // 3) - 0.2),
                                      0.001, 0.0) for i, c in enumerate(group)]
        text = task_file.write("probe", places, [("On", places[0].name, f"{task_file.WORKSPACE}_{places[0].region}")])
        path = os.path.join(folder, f"catalog_{k}.bddl")
        with open(path, "w") as f:
            f.write(text)
        env = TaskEnv("variations", path, render=False, seed=0, execution=Execution(robot=robot, hard_reset=True))
        env.draw_scene(0)
        for p in places:
            out[p.category] = shapes.measure(env.scene, p.name, container=p.category in containers).as_dict()
        env.close()
    return out
