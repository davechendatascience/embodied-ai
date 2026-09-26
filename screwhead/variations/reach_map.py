"""The robot's action space over a table: a top-down reach map (BRN-lv-action-space).

A square grid over the table's top, computed once in a reference scene (the robot on its mount, the
table, the family's fixtures, nothing else fixed) and stored with the benchmark by digest. A grid point
is in when, for every declared jaw line, one of the line's two directions makes the tool, pointing
straight down over the point, a reachable pose (DEF-reachable-pose, the teacher's own screen: reach.Reach)
at every declared height, from the start joints -- the arm written at the solution, the fingers at the
declared aperture, no contact with the table or a fixture -- and every grid point within the border
passes too. Placement then only checks containment: no inverse
kinematics per task.

Grid coordinates are the table's (world x, y), the coordinates of the task file's regions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

REFERENCE_PARTS = ("robot0_", "gripper0_", "mount0_")   # the robot, its gripper and its mount, by body name
MESH = 7                  # mjGEOM_MESH


@dataclass(frozen=True)
class MapSpec:
    spacing: float = 0.02                       # m between grid points
    border: float = 0.05                        # m: every grid point this near must pass as well
    # jaw lines, degrees in the base frame, each served by either of its two directions as the grasp planner
    # offers both (a fixed direction per line confined the map to y < 0.2: the wrist ran out of travel)
    lines_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    heights: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25)   # m, tool point above the top
    aperture: float = 0.08                      # m, fingers fully open: the widest they are on a descent


@dataclass
class ReachMap:
    spec: MapSpec
    origin: tuple[float, float]                 # table coordinates of grid point (0, 0)
    inside: np.ndarray                          # bool [nx, ny]: in the action space
    reference: dict = field(default_factory=dict)   # what a scene must match (match_digest) and how it was made

    # -- containment ------------------------------------------------------------------
    def _cells(self, lo: float, hi: float, axis: int) -> tuple[int, int]:
        """Index range of the cells [c - s/2, c + s/2] that the interval [lo, hi] meets."""
        s, o = self.spec.spacing, self.origin[axis]
        return int(np.floor((lo - o) / s + 0.5)), int(np.ceil((hi - o) / s - 0.5))

    def contains(self, lo: np.ndarray, hi: np.ndarray) -> bool:
        """The plan rectangle [lo, hi] (table coordinates) lies within the cells of points in the set."""
        i0, i1 = self._cells(float(lo[0]), float(hi[0]), 0)
        j0, j1 = self._cells(float(lo[1]), float(hi[1]), 1)
        nx, ny = self.inside.shape
        if i0 < 0 or j0 < 0 or i1 >= nx or j1 >= ny:
            return False
        return bool(self.inside[i0:i1 + 1, j0:j1 + 1].all())

    def points(self) -> np.ndarray:
        """Table coordinates of the grid points in the set, [n, 2]."""
        i, j = np.nonzero(self.inside)
        return np.stack([self.origin[0] + i * self.spec.spacing, self.origin[1] + j * self.spec.spacing], 1)

    # -- storage ----------------------------------------------------------------------
    def as_dict(self) -> dict:
        return {"spec": asdict(self.spec), "origin": list(self.origin),
                "rows": ["".join("#" if v else "." for v in row) for row in self.inside],
                "reference": self.reference}

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> ReachMap:
        spec = MapSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d["spec"].items()})
        inside = np.array([[c == "#" for c in row] for row in d["rows"]], bool)
        return cls(spec=spec, origin=tuple(d["origin"]), inside=inside, reference=d["reference"])

    @classmethod
    def load(cls, path: str, digest: str | None = None) -> ReachMap:
        with open(path) as f:
            rm = cls.from_dict(json.load(f))
        if digest is not None and rm.digest() != digest:
            raise ValueError(f"{path}: digest {rm.digest()}, the benchmark declares {digest}")
        return rm


# -- the reference match ------------------------------------------------------------------
def _reference_bodies(m) -> list[int]:
    """The robot's, gripper's and mount's bodies, the table and the fixtures a reset draws, with every body
    attached below them."""
    from ..sim.task_env_place import drawn_fixtures
    roots = [b for b in range(1, m.nbody) if m.body(b).name.startswith(REFERENCE_PARTS) or b == table_body(m)]
    roots += drawn_fixtures(m)
    return [b for b in range(1, m.nbody) if b in roots or any(_descends(m, b, r) for r in roots)]


def match_digest(m) -> str:
    """A digest of what BRN-lv-action-space's match compares: the world body's own geoms (a floor, or a table or
    fixture modelled on the world), the robot's, its gripper's and its mount's bodies, the table and the fixtures a
    reset draws, with every body below them -- each with its parent's name, its pose
    in its parent (so, the chain rooted at the world, its world pose at given joint positions), its inertia -- their
    geoms, each with its pose in its body, its shape (a mesh's vertices and faces too), contact filters, margins
    and gaps, and their joints; and the model's contact settings: its options, and its contact exclusions and
    explicit pairs by name. The fixtures' joint positions are compared separately (fixtures_of). The arm's joint
    positions are not in it."""
    h = hashlib.sha1()
    _digest_contact_settings(h, m)
    table = table_body(m)
    for b in [0] + sorted(_reference_bodies(m), key=lambda b: m.body(b).name):   # the world's own geoms too
        _digest_body(h, m, b, sites=b not in (0, table))
    return h.hexdigest()[:16]


def _put(h, *arrays) -> None:
    for a in arrays:
        h.update(np.ascontiguousarray(a, float).tobytes())


def _digest_contact_settings(h, m) -> None:
    """The model's options, and its contact exclusions and explicit pairs by name."""
    for name in sorted(n for n in dir(m.opt) if not n.startswith("_")):
        h.update(name.encode())
        _put(h, getattr(m.opt, name))
    for sig in sorted(int(v) for v in m.exclude_signature):
        h.update(f"exclude {m.body(sig >> 16).name} {m.body(sig & 0xFFFF).name}".encode())
    for i in range(m.npair):
        h.update(f"pair {m.geom(int(m.pair_geom1[i])).name} {m.geom(int(m.pair_geom2[i])).name}".encode())
        _put(h, m.pair_margin[i:i + 1], m.pair_gap[i:i + 1])


def _digest_body(h, m, b: int, sites: bool) -> None:
    """A body's name, parent, pose in its parent and inertia; its geoms (meshes too) and joints; and, with `sites`,
    its sites with their poses and sizes -- the robot's and gripper's (the tool frame) and the fixtures' (their
    regions: a drawer's interior, the heating region). Not the table's: they carry each task's own regions."""
    h.update(m.body(b).name.encode())
    h.update(m.body(int(m.body_parentid[b])).name.encode())
    _put(h, m.body_pos[b], m.body_quat[b], m.body_ipos[b], m.body_iquat[b], m.body_mass[b:b + 1], m.body_inertia[b])
    for g in np.nonzero(m.geom_bodyid == b)[0]:
        _put(h, m.geom_type[g:g + 1], m.geom_size[g], m.geom_pos[g], m.geom_quat[g], m.geom_contype[g:g + 1],
             m.geom_conaffinity[g:g + 1], m.geom_margin[g:g + 1], m.geom_gap[g:g + 1])
        if int(m.geom_type[g]) == MESH and int(m.geom_dataid[g]) >= 0:
            k = int(m.geom_dataid[g])
            v0, nv = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
            f0, nf = int(m.mesh_faceadr[k]), int(m.mesh_facenum[k])
            _put(h, m.mesh_vert[v0:v0 + nv], m.mesh_face[f0:f0 + nf])
    h.update(f"mocap {int(m.body_mocapid[b])}".encode())     # -1: posed by its chain, not by the state
    for j in np.nonzero(m.jnt_bodyid == b)[0]:
        a = int(m.jnt_qposadr[j])
        width = 7 if int(m.jnt_type[j]) == 0 else 4 if int(m.jnt_type[j]) == 1 else 1
        # the reference position (qpos0) too: a hinge or slide poses its child by its value less the reference
        _put(h, m.jnt_type[j:j + 1], m.jnt_axis[j], m.jnt_pos[j], m.jnt_range[j], m.jnt_limited[j:j + 1],
             m.qpos0[a:a + width])
    if sites:
        for k in np.nonzero(m.site_bodyid == b)[0]:
            h.update(m.site(int(k)).name.encode())
            _put(h, m.site_pos[k], m.site_quat[k], m.site_size[k], m.site_type[k:k + 1])


def simulator() -> str:
    """The simulator release the screen ran under: collision and kinematics are the same function of a model only
    within one release."""
    import mujoco
    return f"mujoco {mujoco.__version__}"


def held_joints(m, d, moving: set[str]) -> dict[str, float]:
    """{joint: position} of every joint of the robot's, gripper's and mount's bodies and the table other than the
    arm's and gripper's own (`moving`) -- BRN-lv-action-space pins them all; the Panda on its mount has none."""
    from ..sim.task_env_place import drawn_fixtures
    fixtures = set(drawn_fixtures(m))
    bodies = {b for b in _reference_bodies(m)
              if b not in fixtures and not any(_descends(m, b, f) for f in fixtures)}
    return {m.joint(j).name: round(float(d.qpos[m.jnt_qposadr[j]]), 9) for j in range(m.njnt)
            if int(m.jnt_bodyid[j]) in bodies and m.joint(j).name not in moving}


def fixtures_of(m, d) -> dict:
    """{fixture: (world position, world quaternion, joint positions)} of the fixtures a reset draws."""
    from ..sim.task_env_place import drawn_fixtures
    out = {}
    for b in drawn_fixtures(m):
        joints = [float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)
                  if int(m.jnt_bodyid[j]) != b and _descends(m, int(m.jnt_bodyid[j]), b)]
        out[m.body(b).name] = [d.xpos[b].round(9).tolist(), d.xquat[b].round(9).tolist(), joints]
    return out


def _descends(m, body: int, root: int) -> bool:
    while body > 0:
        body = int(m.body_parentid[body])
        if body == root:
            return True
    return False


def matches(m, d, reference: dict) -> bool:
    """The scene matches the reference: the same simulator release, no flex (deformable collision geometry, which
    match_digest does not see), the same robot, mount and table (match_digest), rooted at the world, their joints
    other than the arm's and gripper's at the reference's positions, and the same fixtures at the same poses and
    joint positions."""
    return (simulator() == reference["simulator"] and int(m.nflex) == 0 and match_digest(m) == reference["match_digest"]
            and fixtures_of(m, d) == reference["fixtures"]
            and held_joints(m, d, set(reference["moving_joints"])) == reference["held_joints"])


# -- computing it ---------------------------------------------------------------------------
def _overflows(d) -> tuple[int, int]:
    """MuJoCo's counts of full contact and constraint buffers (mjWARN_CONTACTFULL, mjWARN_CNSTRFULL): it drops what
    does not fit and warns, so a screen during which either rose may have missed a contact."""
    import mujoco
    return (int(d.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number),
            int(d.warning[mujoco.mjtWarning.mjWARN_CNSTRFULL].number))


def erode(passed: np.ndarray, r: int) -> np.ndarray:
    """Points whose every grid neighbour within r cells (Euclidean) passed; a neighbour off the grid is no grid
    point and does not count against it."""
    nx, ny = passed.shape
    inside = passed.copy()
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj > r * r:
                continue
            here = (slice(max(0, -di), nx - max(0, di)), slice(max(0, -dj), ny - max(0, dj)))
            there = (slice(max(0, di), nx - max(0, -di)), slice(max(0, dj), ny - max(0, -dj)))
            inside[here] &= passed[there]
    return inside


def table_body(m) -> int:
    """The table: the body hanging from the world named table or <kind>_table (kitchen, study, living room)."""
    found = [b for b in range(1, m.nbody) if int(m.body_parentid[b]) == 0
             and (m.body(b).name == "table" or m.body(b).name.endswith("_table"))]
    if len(found) != 1:
        raise ValueError(f"expected one table, found {[m.body(b).name for b in found]}")
    return found[0]


def table_origin(scene, region: str, centre) -> np.ndarray:
    """The table coordinates' origin, world xy: where a task-file region's site stands, less the centre the task file
    gave it. Table coordinates are the task file's region coordinates; the kitchen table's origin is the world's, the
    study table's is not."""
    # rounded to a micrometre: read through float arithmetic the kitchen table's came out -1e-17, which moved the
    # table's edge off a grid line and dropped a row of the grid (26 reachable points of K1)
    origin = scene.region(region)[1][:2] + scene.base[:2] - np.asarray(centre, float)
    return np.round(origin, 6) + 0.0


def table_top(scene, origin) -> tuple[float, np.ndarray, np.ndarray]:
    """(height of the top, base frame; plan lo, hi in table coordinates) from the table's collision geometry."""
    m = scene.m
    box = scene.object_box(m.body_id2name(table_body(getattr(m, "_model", m))))
    centre = box.world_centre
    half = np.abs(box.R @ np.diag(box.half)).sum(1)
    world = centre[:2] + scene.base[:2]
    return float(centre[2] + half[2]), world - half[:2] - origin, world + half[:2] - origin


def compute(env, origin, spec: MapSpec | None = None, log=print) -> ReachMap:
    """The map, in env's scene (the reference scene: its free objects play no part, only contacts with the
    table and fixtures fail a pose). env: a TaskEnv at its start joints; origin: the table coordinates' origin."""
    from ..geometry.frames import pose, top_down
    from ..teacher.reach import MIN_MARGIN, MIN_SIGMA, Reach
    spec = spec or MapSpec()
    m, d = env.env.sim.model._model, env.env.sim.data._data
    env.raw = env.observe()
    reach = Reach(env, None)
    table_origin = np.asarray(origin, float)
    top, lo, hi = table_top(env.scene, table_origin)
    base = env.scene.base
    to_base = table_origin - base[:2]                    # table coordinates to the base frame, in plan
    s = spec.spacing
    origin = np.ceil(lo / s) * s                         # grid points on multiples of the spacing
    nx, ny = (np.floor((hi - origin) / s) + 1).astype(int)
    gx, gy = np.meshgrid(origin[0] + s * np.arange(nx), origin[1] + s * np.arange(ny), indexing="ij")
    xy = np.stack([gx.ravel(), gy.ravel()], 1)
    ok = np.ones(len(xy), bool)
    overflowed = 0
    for line in spec.lines_deg:
        served = np.zeros(len(xy), bool)
        for yaw in (line, line + 180.0):
            a = np.deg2rad(yaw)
            R = top_down(np.array([np.cos(a), np.sin(a), 0.0]))
            here = ok & ~served                          # this direction must serve the point at every height
            for h in spec.heights:
                live = np.nonzero(here)[0]
                if not len(live):
                    break
                Ts = [pose(R, np.array([xy[i, 0] + to_base[0], xy[i, 1] + to_base[1], top + h])) for i in live]
                th, conv, sig, margin = reach.solve(Ts)
                good = conv & (sig > MIN_SIGMA) & (margin > MIN_MARGIN)
                for k in np.nonzero(good)[0]:
                    full = _overflows(d)
                    good[k] = reach.collides(th[k], aperture=spec.aperture) < 2
                    if _overflows(d) != full:          # a contact may have been dropped: the screen is not a pass
                        good[k] = False
                        overflowed += 1
                here[live[~good]] = False
            served |= here
        ok &= served
        log(f"jaw line {line:.0f} deg: {int(ok.sum())} of {len(xy)} points pass")
    passed = ok.reshape(nx, ny)
    inside = erode(passed, int(np.floor(spec.border / s + 1e-9)))
    log(f"border {spec.border} m: {int(inside.sum())} of {nx * ny} points in the action space")
    reference = {
        "robot": env.execution.robot, "gripper": env.execution.gripper,
        "start_joints": np.asarray(env.raw["robot0_joint_pos"], float).round(9).tolist(),
        "base_world": base.round(9).tolist(), "table_top": round(top, 9),
        "table_origin": np.asarray(table_origin, float).round(9).tolist(),
        "match_digest": match_digest(m), "fixtures": fixtures_of(m, d), "simulator": simulator(),
    }
    moving = [m.joint(int(j)).name for j in range(m.njnt)
              if int(m.jnt_qposadr[j]) in set(env.joint_indexes) | set(env.gripper_indexes)]
    reference.update(moving_joints=sorted(moving), held_joints=held_joints(m, d, set(moving)),
                     screens_overflowed=overflowed)
    roots = {m.body(int(m.body_parentid[b])).name for b in _reference_bodies(m)} - {m.body(b).name for b in _reference_bodies(m)}
    if roots != {"world"}:
        raise ValueError(f"the matched bodies hang from {sorted(roots)}, not the world alone")
    if int(m.nflex):
        raise ValueError("the reference holds a flex: its collision geometry is not geoms, which the match compares")
    if any(int(m.body_mocapid[b]) >= 0 for b in _reference_bodies(m)):
        raise ValueError("a matched body is a mocap body: its pose is the state's, not its chain's")
    return ReachMap(spec=spec, origin=(float(origin[0]), float(origin[1])), inside=inside, reference=reference)
