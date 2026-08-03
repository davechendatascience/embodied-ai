"""`urdfpy` under Python 3.12, via the maintained fork.

PointWorld's `robot_sampler.py` imports `urdfpy`, which is dead on 3.10+
(`from collections import Mapping`) and, worse, pins `networkx<3` — installing
it downgraded networkx to 2.2 and broke torch 2.11 in this venv. `urchin` is
the maintained fork with the same public API, so this shim re-exports it under
the old name and leaves upstream untouched.

Reached via PYTHONPATH, not site-packages, so it is visible only to runs that
opt in and cannot silently shadow a real urdfpy elsewhere.
"""
from urchin import *  # noqa: F401,F403
from urchin import URDF, Link, Joint  # noqa: F401
