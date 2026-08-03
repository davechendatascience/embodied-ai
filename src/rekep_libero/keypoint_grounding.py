"""Language instruction -> (which scene points, where they go), KUDA-style.

This is the layer PointWorld does not have and its authors did not build. The
paper's own task specification is a human clicking a SAM2 mask and typing
target positions into a GUI (Appendix A.6.2); everything downstream of that is
already working here. So this module replaces the human, and nothing else.

The recipe is KUDA's (`third_party/KUDA`, arXiv 2503.10546 -- 80% over 60
trials with free-form language):

  1. farthest-point-sample keypoints over the scene cloud
  2. project them into the RGB image and mark them `P[i]`
  3. ask a VLM to write the final state as offsets FROM OTHER KEYPOINTS
  4. parse `p_i = p_a + [dx, dy, dz]` back into world-frame targets

Step 3 is the whole trick, and it is why this can work at all. Our own
measurements put VLM spatial grounding at +/-150 mm and its constraint output
at +/-18 mm, against a task that turns on 7 mm. A VLM asked for an absolute
coordinate would be hopeless. Asked for "5 cm to the left of that other point"
it only has to be right about a RELATION, and the metric precision comes from
the perceived keypoint it references.

ONE DELIBERATE DEPARTURE FROM KUDA. Their camera is top-down, so image x/y are
table x/y and they can speak in image coordinates. Our agentview is oblique,
so the same wording would silently mean something else. The prompt below is
therefore written in the WORLD frame with the axes named explicitly, and
keypoints are projected for display only.

Nothing here is wired into the planner yet -- see `plan_pointworld.py`, which
still takes its target from `task_spec.py`. That swap is the next step, and
doing it in two stages keeps a grounding failure distinguishable from a
planner failure, which matters because our VLM grounding has failed once
already (`NOTES.md` section 4).
"""

import re

import numpy as np

PROMPT = """\
You are specifying the final state of a manipulation task for a robot.

The image shows the workspace with candidate keypoints marked P[i], each on a
real surface in the scene. Write a short Python function describing where the
keypoints you care about should END UP, as offsets from other keypoints.

    def keypoint_specification():
        p_3 = p_11 + [0, 5, 0]
        p_4 = p_12 + [0, 5, 0]
        return p_3, p_4

Rules:
- Coordinates are the ROBOT WORLD frame in centimetres, not image pixels:
  +x is away from the robot base, +y is to the robot's left, +z is up.
- Use the form `p_i = p_a + [dx, dy, dz]`, referencing another keypoint.
  If nothing suitable exists, `p_i = [dx, dy, dz]` is measured from keypoint 0.
- Do not invent keypoints. Only indices marked in the image exist.
- Specify only the few keypoints needed to pin down the final state, not every
  point on the object.
- A planner will drive the chosen keypoints to those targets under an MSE
  loss, so the offsets must describe the goal, not the path to it.
- If the task already looks complete, reply exactly: Done.

Task: {instruction}
"""


def propose_keypoints(points, n=24, seed=0):
    """Farthest-point sample `n` keypoint indices over the scene cloud.

    KUDA's proposal method. Deliberately geometric rather than semantic: the
    VLM is the thing that knows which keypoint matters, so the proposer only
    has to cover the scene evenly enough that a relevant one exists. It also
    means keypoint indices ARE scene-point indices, so the mask PointWorld
    wants needs no second lookup.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= n:
        return np.arange(len(pts))
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(pts - pts[nxt], axis=1))
    return np.array(sorted(chosen))


def project(points, intrinsic, cam2world):
    """(u, v, in_front) pixel coordinates of world points, OpenCV convention."""
    world2cam = np.linalg.inv(np.asarray(cam2world, dtype=np.float64))
    cam = np.asarray(points) @ world2cam[:3, :3].T + world2cam[:3, 3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = cam[:, 0] / safe * intrinsic[0, 0] + intrinsic[0, 2]
    v = cam[:, 1] / safe * intrinsic[1, 1] + intrinsic[1, 2]
    return u, v, z > 0


def annotate(rgb, points, idx, intrinsic, cam2world, scale=3):
    """The marked image the VLM sees: `P[i]` drawn at each keypoint."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    u, v, ok = project(points[idx], intrinsic, cam2world)
    for j, (x, y, good) in enumerate(zip(u * scale, v * scale, ok)):
        if not good or not (0 <= x < img.width and 0 <= y < img.height):
            continue
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 60, 60), outline=(0, 0, 0))
        d.text((x + 6, y - 6), f"P{j}", fill=(255, 255, 0))
    return img


def parse_targets(response):
    """`p_3 = p_11 + [0, 5, 0]` -> {3: (11, array([0, 5, 0]))}.

    Follows KUDA's parser, including its convention that a missing reference
    means "measured from the origin keypoint" rather than being an error.
    Returns an empty dict for "Done.", which is the VLM's own completion
    signal and doubles as a termination condition the planner currently takes
    from LIBERO's `_check_success()`.
    """
    if response.strip().lower().startswith("done"):
        return {}
    targets = {}
    for line in response.split("\n"):
        line = line.strip()
        if "=" not in line or not line.startswith("p"):
            continue
        arrays = re.findall(r"\[.*?\]", line)
        if not arrays:
            continue
        nums = re.findall(r"p_?(\d+)", line)
        if not nums:
            continue
        target = int(nums[0])
        reference = int(nums[1]) if len(nums) > 1 else -1
        try:
            offset = np.array(eval(arrays[0]), dtype=np.float64)  # noqa: S307
        except Exception:  # noqa: BLE001 - a malformed line is not fatal
            continue
        if offset.shape == (3,):
            targets[target] = (reference, offset)
    return targets


def to_goal(targets, points, idx):
    """(goal_idx, goal_pos) in the WORLD frame, ready for the bridge.

    `idx` maps the VLM's P[i] back to scene-point indices, so the returned
    `goal_idx` indexes the same cloud the planner masks over. Offsets arrive
    in centimetres, per the prompt, and leave in metres.
    """
    gi, gp = [], []
    for t, (ref, offset) in sorted(targets.items()):
        if t >= len(idx):
            continue
        base = points[idx[ref]] if 0 <= ref < len(idx) else points[idx[0]]
        gi.append(int(idx[t]))
        gp.append(base + offset / 100.0)
    if not gi:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float32)
    return np.array(gi, dtype=np.int64), np.array(gp, dtype=np.float32)


def ground(instruction, rgb, points, intrinsic, cam2world, backend,
           n_keypoints=24, save_image=None):
    """instruction -> (goal_idx, goal_pos, marked image, raw VLM reply).

    An empty `goal_idx` means the VLM said the task is already done.
    """
    idx = propose_keypoints(points, n_keypoints)
    img = annotate(rgb, points, idx, intrinsic, cam2world)
    if save_image:
        img.save(save_image)

    import base64
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT.format(instruction=instruction)},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}]
    reply = backend(messages)
    goal_idx, goal_pos = to_goal(parse_targets(reply), points, idx)
    return goal_idx, goal_pos, img, reply
