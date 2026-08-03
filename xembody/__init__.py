"""VLA intent -> planner feasibility, in four small modules.

    from xembody import keypose, world, frames
    from xembody.planner import Planner     # imports cuRobo; the others do not

Read README.md first: six failure modes are documented there, each of which
cost days to find and none of which is visible from the code.
"""

from . import frames, keypose, world

__all__ = ["frames", "keypose", "world"]
