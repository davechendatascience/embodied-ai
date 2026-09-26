"""tools/eval_vla.reproduce: the two rounds of an evaluation compared episode by episode."""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("eval_vla", Path(__file__).resolve().parents[1] / "tools" / "eval_vla.py")
eval_vla = importlib.util.module_from_spec(spec)
sys.modules["eval_vla"] = eval_vla          # its dataclasses look their module up
spec.loader.exec_module(eval_vla)


def _row(task, e, **metrics):
    return {"metrics": dict(metrics), "repro": {"task": task, "init_index": e}}


def test_agreeing_rounds_reproduce():
    a = [_row(0, 0, success=True, digest="x"), _row(0, 1, success=False, digest="y")]
    b = [_row(0, 1, success=False, digest="y"), _row(0, 0, success=True, digest="x")]
    out = eval_vla.reproduce(a, b)
    assert [r["metrics"]["reproduced"] for r in out] == [True, True]
    assert all("digest" not in r["metrics"] for r in out)


def test_a_differing_digest_or_outcome_does_not_reproduce():
    a = [_row(0, 0, success=True, digest="x"), _row(0, 1, success=True, digest="y")]
    b = [_row(0, 0, success=True, digest="z"), _row(0, 1, success=False, digest="y")]
    assert [r["metrics"]["reproduced"] for r in eval_vla.reproduce(a, b)] == [False, False]


def test_unscored_episodes():
    # a task not admitted has no digest and nothing to compare; a model that differed in both rounds reproduces
    a = [_row(1, 0, admitted=False, placed=False), _row(2, 0, placed=False, digest="model differs")]
    b = [_row(1, 0, admitted=False, placed=False), _row(2, 0, placed=False, digest="model differs")]
    out = eval_vla.reproduce(a, b)
    assert "reproduced" not in out[0]["metrics"]
    assert out[1]["metrics"]["reproduced"] is True and "success" not in out[1]["metrics"]


def test_an_episode_missing_from_the_second_round_does_not_reproduce():
    out = eval_vla.reproduce([_row(0, 0, success=True, digest="x")], [])
    assert out[0]["metrics"]["reproduced"] is False
