"""Load a MuJoCo-exported world into cuRobo — the .venv-curobo side of the bridge.

`rekep_libero/world_export.py` runs in `.venv` beside MuJoCo and writes JSON.
This runs in `.venv-curobo` and reads it. Neither imports the other; the file on
disk is the whole interface. Same rule as `pointworld_bridge/`.

This module therefore imports **cuRobo, numpy and stdlib only**. If it ever
grows an import of `rekep_libero`, the separation is gone and rebuilding one
venv will start breaking the other for no visible reason.

API NOTE, because the cuRobo in `third_party/curobo` is NOT the one most
documentation describes. The widely-quoted form:

    from curobo.geom.types import WorldConfig, Mesh          # OLD
    world = WorldConfig(mesh=[...])

does not exist here. This checkout (2026, commit 8e734f3) flattened the public
API and moved internals under `curobo/_src/`:

    from curobo.scene import Scene, Cuboid, Sphere, Cylinder, Mesh, VoxelGrid

and `Scene` takes LISTS of typed obstacles rather than name-keyed dicts.
`VoxelGrid` is the hook E3b needs for the depth-derived world.
"""

import json

from curobo.scene import Cuboid, Cylinder, Mesh, Scene, Sphere


def scene_from_dict(world):
    """Build a cuRobo Scene from world_export.py's schema."""
    cuboids = [
        Cuboid(name=name, pose=list(spec["pose"]), dims=list(spec["dims"]))
        for name, spec in (world.get("cuboid") or {}).items()
    ]
    spheres = [
        Sphere(name=name, pose=[*spec["position"], 1.0, 0.0, 0.0, 0.0],
               radius=float(spec["radius"]))
        for name, spec in (world.get("sphere") or {}).items()
    ]
    cylinders = [
        Cylinder(name=name, pose=list(spec["pose"]),
                 radius=float(spec["radius"]), height=float(spec["height"]))
        for name, spec in (world.get("cylinder") or {}).items()
    ]
    meshes = [
        Mesh(name=name, pose=list(spec["pose"]),
             vertices=spec["vertices"], faces=spec["faces"])
        for name, spec in (world.get("mesh") or {}).items()
    ]
    return Scene(cuboid=cuboids or None, sphere=spheres or None,
                 cylinder=cylinders or None, mesh=meshes or None)


def scene_from_json(path):
    with open(path) as f:
        return scene_from_dict(json.load(f))


def describe(scene):
    return {
        "cuboid": len(scene.cuboid or []),
        "sphere": len(scene.sphere or []),
        "cylinder": len(scene.cylinder or []),
        "mesh": len(scene.mesh or []),
        "objects": len(scene.objects or []),
    }


if __name__ == "__main__":
    import argparse

    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("world_json")
    args = ap.parse_args()

    scene = scene_from_json(args.world_json)
    print(f"loaded: {describe(scene)}")

    # Extent check. The oracle world must enclose LIBERO's table scene; a world
    # that silently came back empty, or with everything at the origin, loads
    # without complaint and then reports every trajectory collision-free.
    pts = np.array([o.pose[:3] for o in scene.objects])
    print(f"obstacle centres: x {pts[:, 0].min():+.3f}..{pts[:, 0].max():+.3f}  "
          f"y {pts[:, 1].min():+.3f}..{pts[:, 1].max():+.3f}  "
          f"z {pts[:, 2].min():+.3f}..{pts[:, 2].max():+.3f}")
    at_origin = int((np.linalg.norm(pts, axis=1) < 1e-9).sum())
    print(f"obstacles at exactly the origin: {at_origin}"
          + ("  <-- suspicious" if at_origin > 1 else ""))
