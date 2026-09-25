"""screwhead/teacher/affordances.yaml covers every category the three suites name, and each
entry is well formed.

  PYTHONPATH=third_party/LIBERO:. .venv-libero/bin/python -m pytest tests/test_affordances.py -q
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLE = yaml.safe_load((ROOT / "screwhead/teacher/affordances.yaml").read_text())["categories"]
SUITES = ("libero_object", "libero_spatial", "libero_goal")
MOVED_BY = {"pick", "push", "none"}
HELD_BY = {"faces", "sides", "rim", "body", "handle", "none"}
RESTS = {"upright", "lying", "any"}
SUPPORTS = {"surface", "container", "none"}


def suite_categories() -> tuple[set[str], set[str]]:
    benchmark = pytest.importorskip("libero.libero.benchmark")
    from libero.libero import get_libero_path

    from screwhead.teacher.task_spec import parse
    objects, fixtures = set(), set()
    for suite in SUITES:
        bm = benchmark.get_benchmark_dict()[suite]()
        for i in range(bm.n_tasks):
            t = bm.get_task(i)
            spec = parse(os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file), suite)
            objects |= set(spec.objects.values())
            fixtures |= set(spec.fixtures.values())
    return objects, fixtures


def test_every_suite_category_has_an_entry():
    objects, fixtures = suite_categories()
    assert not objects - TABLE.keys(), f"objects without affordances: {sorted(objects - TABLE.keys())}"
    assert not fixtures - TABLE.keys(), f"fixtures without affordances: {sorted(fixtures - TABLE.keys())}"
    for cat in objects:
        assert TABLE[cat]["kind"] == "object", cat
    for cat in fixtures:
        assert TABLE[cat]["kind"] == "fixture", cat


@pytest.mark.parametrize("cat", sorted(TABLE))
def test_entry_is_well_formed(cat):
    e = TABLE[cat]
    assert e["kind"] in ("object", "fixture")
    assert e["supports"] in SUPPORTS
    assert e.get("evidence"), f"{cat}: every entry says what backs it"
    assert set(e["evidence"]) <= {"measured", "declared"}
    if e["kind"] == "object":
        assert e["moved_by"] in MOVED_BY and e["held_by"] in HELD_BY
        rests = e["rests"] if isinstance(e["rests"], dict) else {"default": e["rests"]}
        assert "default" in rests and set(rests.values()) <= RESTS
        if e["moved_by"] == "push":
            assert e["held_by"] == "none", f"{cat}: a pushed object is not held"
        # a second hold is read only as the handle, and only beside a different first hold
        assert e.get("also_held_by", "handle") == "handle", f"{cat}: also_held_by is read only as handle"
        assert not ("also_held_by" in e and e["held_by"] == "handle"), f"{cat}: held by its handle already"
