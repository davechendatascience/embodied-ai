"""Object proposals from the point cloud alone, and the scene graph over them.

THE GUIDE'S LAYER, not KUDA's. `docs/pointworld_pipeline_implementation_prereqs.md`
asks for a coarse grounding stack that produces an OBJECT-LEVEL scene graph --
`object_id`, `label`, `centroid_xyz`, `bbox_extent_xyz`, `relations` -- which an
LLM goal formalizer then consumes. That is a different shape from the
keypoint-offset prompt in `keypoint_grounding.py`, and the difference is why
this exists.

Why proposals come from GEOMETRY first and features second, measured:

  * farthest-point keypoints do not reliably land on the target at all. On
    `libero_goal/1` NONE of 24 keypoints fell on the bowl, so the VLM could not
    have named it however good it was.
  * a GLOBAL DINOv3 feature-similarity mask tops out at 0.29-0.67 IoU even when
    the query prototype is computed FROM the oracle answer -- an upper bound no
    real query beats. Appearance cannot separate three identical drawer faces;
    that is a spatial question.

Objects on a table are spatially separated, so connectivity does the work that
appearance cannot, and DINOv3 is left for what it is good at: telling two
touching things apart, and labelling.

Nothing here reads the simulator.
"""

import numpy as np
from scipy.spatial import cKDTree


def table_height(points, bins=200):
    """The z of the dominant horizontal surface.

    A tabletop is the single most populated z-slice in a tabletop scene, which
    is cheaper and steadier than plane RANSAC and needs no normals. Used only
    to CUT, so being a few millimetres off costs a few millimetres of skirt.
    """
    z = points[:, 2]
    hist, edges = np.histogram(z, bins=bins)
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) * 0.5)


def cluster(points, radius=0.02, min_points=25, z_margin=0.008):
    """Connected components over a radius graph, above the table.

    Returns an int array of labels, -1 for table/unclustered.

    `radius` is the only real parameter and it trades merging against
    splitting: too large fuses a bowl into the table skirt, too small shatters
    a thin rim. 20 mm against a 15 mm voxel grid means neighbours in the
    downsampled cloud connect and separate objects do not.
    """
    z0 = table_height(points)
    above = points[:, 2] > z0 + z_margin
    labels = np.full(len(points), -1, dtype=np.int64)
    idx = np.flatnonzero(above)
    if len(idx) == 0:
        return labels

    pts = points[idx]
    tree = cKDTree(pts)
    pairs = tree.query_pairs(radius, output_type="ndarray")

    # Union-find. `scipy.sparse.csgraph` would also do it; this keeps the
    # dependency surface to what the project already imports.
    parent = np.arange(len(pts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    roots = np.array([find(i) for i in range(len(pts))])
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    keep = counts >= min_points
    remap = np.full(len(uniq), -1, dtype=np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    labels[idx] = remap[inv]
    return labels


def scene_graph(points, labels, features=None):
    """One record per proposal, in the guide's schema.

    `features` (Ns, C) is optional; when given, each object carries its mean
    DINOv3 feature so a text or image query can rank proposals WITHOUT having
    to segment anything itself -- the expensive part is already done.
    """
    out = []
    for k in range(int(labels.max()) + 1 if labels.max() >= 0 else 0):
        m = labels == k
        if not m.any():
            continue
        p = points[m]
        lo, hi = p.min(axis=0), p.max(axis=0)
        rec = {
            "object_id": int(k),
            "centroid_xyz": ((lo + hi) * 0.5).tolist(),
            "bbox_extent_xyz": (hi - lo).tolist(),
            "n_points": int(m.sum()),
            "top_z": float(hi[2]),
        }
        if features is not None:
            f = features[m].mean(axis=0)
            rec["feature"] = f / max(np.linalg.norm(f), 1e-9)
        out.append(rec)
    return out


def describe(graph, decimals=3):
    """The scene graph as compact text for a language model.

    Coordinates stay in the WORLD frame with axes named, because our agentview
    is oblique -- KUDA speaks in image coordinates only because their camera is
    top-down, and inheriting that would silently mean something else.
    """
    lines = ["# objects on the table (world frame, metres:"
             " +x away from the robot, +y to its left, +z up)"]
    for r in graph:
        c = np.round(r["centroid_xyz"], decimals)
        e = np.round(r["bbox_extent_xyz"], decimals)
        lines.append(f"object {r['object_id']}: centre=({c[0]}, {c[1]}, {c[2]}) "
                     f"size=({e[0]}, {e[1]}, {e[2]}) points={r['n_points']}")
    return "\n".join(lines)


def annotate(rgb, points, labels, intrinsic, cam2world, scale=3):
    """The image the VLM sees: each proposal numbered at its centroid."""
    from PIL import Image, ImageDraw

    from .keypoint_grounding import project

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for k in range(int(labels.max()) + 1 if labels.max() >= 0 else 0):
        m = labels == k
        if not m.any():
            continue
        c = points[m].mean(axis=0)[None]
        u, v, ok = project(c, intrinsic, cam2world)
        if not ok[0]:
            continue
        x, y = float(u[0]) * scale, float(v[0]) * scale
        if not (0 <= x < img.width and 0 <= y < img.height):
            continue
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 60, 60), outline=(0, 0, 0))
        d.text((x - 4, y - 6), str(k), fill=(255, 255, 0))
    return img


PROMPT = """\
A robot arm must perform one task on the table shown.

Each candidate object is marked with a NUMBER in the image, and listed below
with its position and size in the robot's world frame (metres): +x points away
from the robot base, +y to the robot's left, +z up.

{graph}

Task: "{instruction}"

Reply with ONLY a JSON object and nothing else:

{{"target": <number of the object that must MOVE>,
  "destination": <number of the object it should end up ON, or null>,
  "offset_cm": [dx, dy, dz]}}

Rules:
- "target" is the object the robot moves. It is never the robot or the table.
- If the task says to put something ON something else, give that as
  "destination"; the object will be placed on its top surface.
- If there is no destination object, leave it null and put the whole
  displacement in "offset_cm", in CENTIMETRES in the world frame above.
- "offset_cm" is an extra correction on top of the destination; [0, 0, 0] is
  fine and is the common case.
"""


def parse(reply):
    """The first JSON object in the reply -> (target, destination, offset_m).

    Tolerant on purpose: small models wrap JSON in prose or code fences, and a
    parse failure here is indistinguishable from a grounding failure unless it
    is reported separately.
    """
    import json
    import re

    m = re.search(r"\{.*?\}", reply, re.S)
    if not m:
        raise ValueError(f"no JSON object in reply: {reply.strip()[:200]!r}")
    obj = json.loads(m.group(0))
    off = np.asarray(obj.get("offset_cm") or [0, 0, 0], dtype=float) / 100.0
    dest = obj.get("destination")
    return int(obj["target"]), (None if dest is None else int(dest)), off


def to_goal(graph, points, labels, target, destination, offset_m, clearance=0.02):
    """(goal_idx, goal_pos) for the planner, from an object-level answer.

    The planner's contract is per-POINT: which scene points must move, and
    where each ends up. An object-level answer becomes that by translating the
    whole target cluster rigidly, which is also the only honest reading -- the
    grounder said nothing about deformation.
    """
    by_id = {r["object_id"]: r for r in graph}
    if target not in by_id:
        raise ValueError(f"target {target} is not a proposal (have {sorted(by_id)})")
    tgt = by_id[target]
    goal_idx = np.flatnonzero(labels == target)

    centre = np.asarray(tgt["centroid_xyz"], dtype=float)
    if destination is not None and destination in by_id:
        dst = by_id[destination]
        dc = np.asarray(dst["centroid_xyz"], dtype=float)
        new = np.array([dc[0], dc[1],
                        dst["top_z"] + tgt["bbox_extent_xyz"][2] * 0.5 + clearance])
    else:
        new = centre.copy()
    new = new + offset_m
    return goal_idx, (points[goal_idx] + (new - centre)).astype(np.float32)


def ground(instruction, rgb, points, intrinsic, cam2world, backend,
           radius=0.02, save_image=None):
    """instruction + scene -> (goal_idx, goal_pos, graph, labels, reply)."""
    import base64
    import io

    labels = cluster(points, radius=radius)
    graph = scene_graph(points, labels)
    img = annotate(rgb, points, labels, intrinsic, cam2world)
    if save_image:
        img.save(save_image)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    msg = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(graph=describe(graph),
                                               instruction=instruction)},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}]
    reply = backend(msg)
    t, d, off = parse(reply)
    gi, gp = to_goal(graph, points, labels, t, d, off)
    return gi, gp, graph, labels, reply
