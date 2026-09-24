"""screwhead/teacher/door_swing.json, the table BRN-swing-door-as-the-humans reads: every table the law can
choose is non-empty, its entries sorted by angle, each a point and a rotation in the door's frame."""
import json
from pathlib import Path

import numpy as np

TABLE = Path(__file__).resolve().parents[1] / "screwhead/teacher/door_swing.json"


def test_every_table_the_law_chooses_has_entries():
    modes = json.loads(TABLE.read_text())["modes"]
    for mode, side in (("open", "front"), ("open", "behind"), ("close", "front")):
        rows = modes[mode][side]
        assert rows, (mode, side)
        qs = [r["q"] for r in rows]
        assert qs == sorted(qs), (mode, side)


def test_entries_are_poses():
    modes = json.loads(TABLE.read_text())["modes"]
    for sides in modes.values():
        for rows in sides.values():
            for r in rows:
                assert len(r["p"]) == 3
                R = np.asarray(r["R"], float)
                assert np.allclose(R.T @ R, np.eye(3), atol=2e-3)
                assert abs(np.linalg.det(R) - 1.0) < 2e-3
