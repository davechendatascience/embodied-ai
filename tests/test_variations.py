"""LIBERO-Variations' pure parts: the reach map's containment and border, the fit rules, the order words and
the benchmark's own declarations (CMP-libero-variations)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

from screwhead.variations import shapes
from screwhead.variations.reach_map import MapSpec, ReachMap, erode

ROOT = Path(__file__).resolve().parents[1]
V0 = ROOT / "screwhead/variations/benchmarks/v0.yaml"


def _map(inside: np.ndarray, spacing: float = 0.02) -> ReachMap:
    return ReachMap(spec=MapSpec(spacing=spacing), origin=(0.0, 0.0), inside=inside)


def test_contains_is_the_union_of_cells():
    inside = np.zeros((5, 5), bool)
    inside[1:4, 1:4] = True                     # points 0.02..0.06, cells 0.01..0.07
    rm = _map(inside)
    assert rm.contains(np.array([0.011, 0.011]), np.array([0.069, 0.069]))
    assert not rm.contains(np.array([0.009, 0.02]), np.array([0.03, 0.03]))    # reaches cell 0
    assert not rm.contains(np.array([0.02, 0.02]), np.array([0.071, 0.03]))    # reaches cell 4
    assert not rm.contains(np.array([-0.05, 0.02]), np.array([0.03, 0.03]))    # off the grid


def test_erode_needs_every_neighbour_within_the_border():
    passed = np.ones((7, 7), bool)
    passed[3, 3] = False
    inside = erode(passed, 1)
    assert not inside[3, 3] and not inside[2, 3] and not inside[3, 4]
    assert inside[2, 2]                          # diagonal neighbour is sqrt(2) cells away: outside radius 1
    assert inside[0, 0]                          # a neighbour off the grid does not count against a point


def test_reach_map_round_trips_with_its_digest():
    inside = np.zeros((3, 4), bool)
    inside[1, 2] = True
    rm = _map(inside)
    back = ReachMap.from_dict(rm.as_dict())
    assert back.digest() == rm.digest()
    assert np.array_equal(back.inside, inside)
    assert np.allclose(back.points(), [[0.02, 0.04]])


def _shape(ha: float, hb: float, walls=()) -> shapes.Shape:
    return shapes.Shape(up=(0.0, 0.0, 1.0), axes=(0, 1), half=(ha, hb), height=0.05, walls=tuple(walls))


def test_fits_on_either_way_round():
    plate = _shape(0.06, 0.03)
    assert shapes.fits_on(_shape(0.024, 0.05), plate, 0.005)      # turned a quarter turn
    assert not shapes.fits_on(_shape(0.03, 0.03), plate, 0.005)    # 70 mm grown across 60 mm


def test_fits_in_avoids_the_walls():
    box = _shape(0.06, 0.06, walls=[(-0.06, -0.06, -0.05, 0.06), (0.05, -0.06, 0.06, 0.06),
                                    (-0.06, -0.06, 0.06, -0.05), (-0.06, 0.05, 0.06, 0.06)])
    assert shapes.fits_in(_shape(0.04, 0.04), box, 0.005)
    assert not shapes.fits_in(_shape(0.047, 0.02), box, 0.005)     # 52 mm half grown past the 50 mm wall
    assert not shapes.fits_in(_shape(0.01, 0.01), _shape(0.06, 0.06), 0.005)   # no walls: not a container


def test_order_words():
    from screwhead.variations.generator import order_words
    base = np.array([-0.66, 0.0])
    xy = {"a": np.array([-0.3, 0.0]), "b": np.array([-0.1, 0.01]), "c": np.array([0.05, 0.0])}
    assert order_words(["a"], xy, base, 0.04) == {"a": ""}
    assert order_words(["a", "b"], xy, base, 0.04) == {"a": "front", "b": "back"}
    assert order_words(["a", "b", "c"], xy, base, 0.04) == {"a": "front", "b": "middle", "c": "back"}
    side = {"a": np.array([-0.2, 0.2]), "b": np.array([-0.2, -0.2])}
    assert order_words(["a", "b"], side, base, 0.04) == {"a": "left", "b": "right"}
    assert order_words(["a", "b"], {"a": np.array([-0.2, 0.0]), "b": np.array([-0.19, 0.01])}, base, 0.04) is None


def test_v0_nouns_single_out_categories():
    """BRN-lv-instructions-unambiguous: distinct noun phrases, none inside another as whole words -- over the
    scene's objects and its fixtures, the table among them."""
    meta = yaml.safe_load(V0.read_text())
    pool = set(meta["pool"]["pick"]) | set(meta["pool"]["targets"]) | {meta["family"]["workspace"]}
    pool |= set(meta["family"]["fixtures"])
    nouns = meta["nouns"]
    assert pool <= set(nouns)
    phrases = [nouns[c] for c in pool]
    assert len(set(phrases)) == len(phrases)
    for p in phrases:
        for q in phrases:
            if p != q:
                assert not re.search(rf"\b{re.escape(p)}\b", q), (p, q)


def test_v0_templates_agree_with_the_affordance_table():
    """BRN-lv-tasks-doable: moved objects are picked with a declared grasp; targets support as the template needs."""
    meta = yaml.safe_load(V0.read_text())
    aff = yaml.safe_load((ROOT / "screwhead/teacher/affordances.yaml").read_text())["categories"]
    for c in meta["pool"]["pick"]:
        assert aff[c]["moved_by"] == "pick" and aff[c]["held_by"] not in (None, "none", "open"), c
        assert aff[c]["rests"] == "upright" or aff[c]["rests"].get("default") == "upright", c
    for name, t in meta["templates"].items():
        assert isinstance(t["goal"], str), f"{name}: goal must be a string (YAML reads a bare On as true)"
        for c in t["targets"]:
            assert aff[c]["supports"] == t["supports"], (name, c)


def test_v0_declared_digests_match_the_stored_files():
    """The benchmark is its metadata: the files it names are the ones it declares by digest."""
    from screwhead.variations.generator import Benchmark
    bench = Benchmark.load(str(V0))              # raises on either mismatch
    assert bench.reach.digest() == bench.meta["action_space"]["digest"]
    assert set(bench.catalog) == set(bench.meta["pool"]["pick"]) | set(bench.meta["pool"]["targets"])
