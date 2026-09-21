"""Contact queries against a MuJoCo model, written once.

Every question the teacher, the environments and the assessment ask about contact --
is the robot touching something, what, is it bolted down, which finger is on the object,
is the object held in the air -- used to be its own loop over `data.contact`, fourteen of
them in five files, and they drifted: one skipped non-penetrating contacts and another
did not, one missed the fingertip bodies. Each query is defined here and nowhere else.

The robot is every body whose name starts with "robot" or "gripper" (robosuite names
them robot0_* and gripper0_*). The Panda's pads are gripper0_{left,right}finger and
their tips gripper0_finger_joint{1,2}_tip; contact usually lands on the tips.
"""
from __future__ import annotations

from collections.abc import Iterator

ROBOT_PREFIXES = ("robot", "gripper")
LEFT_FINGER = ("leftfinger", "finger_joint1")
RIGHT_FINGER = ("rightfinger", "finger_joint2")


def body_name(m, body: int) -> str:
    return m.body_id2name(int(body)) or ""


def is_robot(name: str) -> bool:
    return name.startswith(ROBOT_PREFIXES)


def is_movable(m, body: int) -> bool:
    """A body the arm can push out of the way: one that hangs on a free joint."""
    n, adr = int(m.body_jntnum[body]), int(m.body_jntadr[body])
    return any(int(m.jnt_type[adr + i]) == 0 for i in range(n))


def _pairs(m, d, penetrating: bool) -> Iterator[tuple[int, int]]:
    for i in range(d.ncon):
        c = d.contact[i]
        if penetrating and c.dist >= 0:
            continue
        yield int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])


def touching(m, d, body: int, penetrating: bool = False) -> Iterator[int]:
    """Bodies in contact with `body`."""
    for b1, b2 in _pairs(m, d, penetrating):
        if body == b1:
            yield b2
        elif body == b2:
            yield b1


def robot_contacts(m, d, penetrating: bool = True) -> Iterator[int]:
    """Non-robot bodies the robot is in contact with (penetrating by default)."""
    for b1, b2 in _pairs(m, d, penetrating):
        r1, r2 = is_robot(body_name(m, b1)), is_robot(body_name(m, b2))
        if r1 != r2:
            yield b2 if r1 else b1


def robot_in_contact(m, d) -> bool:
    """True if any robot geom penetrates something that is not the robot."""
    return next(robot_contacts(m, d), None) is not None


def finger_sides(m, d, body: int) -> set[int]:
    """Which finger groups touch `body`: 0 left, 1 right."""
    sides = set()
    for other in touching(m, d, body):
        name = body_name(m, other)
        if any(k in name for k in LEFT_FINGER):
            sides.add(0)
        elif any(k in name for k in RIGHT_FINGER):
            sides.add(1)
    return sides


def finger_sides_on_geom(m, d, geom: int) -> set[int]:
    """Which finger groups touch this one geom: 0 left, 1 right."""
    sides = set()
    for i in range(d.ncon):
        c = d.contact[i]
        if geom not in (c.geom1, c.geom2):
            continue
        name = body_name(m, int(m.geom_bodyid[c.geom2 if c.geom1 == geom else c.geom1]))
        if any(k in name for k in LEFT_FINGER):
            sides.add(0)
        elif any(k in name for k in RIGHT_FINGER):
            sides.add(1)
    return sides


def touch_summary(m, d, body: int) -> tuple[set[int], bool, bool]:
    """(finger sides touching `body`, the robot touches it, something else touches it)."""
    others = [body_name(m, b) for b in touching(m, d, body)]
    return finger_sides(m, d, body), any(is_robot(n) for n in others), any(not is_robot(n) for n in others)


def only_gripper(m, d, body: int) -> bool:
    """`body` touches the gripper and nothing else -- it is being carried."""
    seen = False
    for other in touching(m, d, body):
        if not body_name(m, other).startswith("gripper"):
            return False
        seen = True
    return seen


def contact_grade(m, d, allow: int = -1) -> int:
    """0 clear, 1 the robot touches something loose, 2 something bolted down.

    `allow` (and bodies whose parent is `allow`) do not count: touching the target is
    the point of a grasp.
    """
    worst = 0
    for other in robot_contacts(m, d, penetrating=True):
        if other == allow or int(m.body_parentid[other]) == allow:
            continue
        worst = max(worst, 1 if is_movable(m, other) else 2)
    return worst
