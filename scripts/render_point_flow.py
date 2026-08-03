"""Render predicted vs true point flow over a recorded episode.

The scoring harness prints two numbers per episode. Those numbers hide the one
thing that decides whether a dynamics model is useful for control: WHERE it
puts the motion. A model that moves the whole cloud a little scores similarly
to one that moves the right 73 points a lot, and only the second can be
planned against.

So: true positions in green, predicted in magenta, a line between them, over
the agentview image the points came from. The static majority of the scene is
drawn faintly, because spurious motion there is a real failure mode and hiding
it would flatter the model.

Runs in `.venv` (imageio, PIL), not `.venv-pw`. The prediction crosses the
venv boundary as an .npz, the same way the episode does:

    CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \\
        .venv-pw/bin/python tests/run_pointworld_on_episode.py \\
            --save-pred data/pw_episodes/pred_ep0.npz
    PYTHONPATH=src .venv/bin/python scripts/render_point_flow.py \\
        data/pw_episodes/pred_ep0.npz
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

SCALE = 3           # upscale, so 256 px of camera becomes something watchable
HOLD = 6            # frames per timestep
FPS = 12

GREY = (150, 150, 150)
GREEN = (60, 230, 120)
MAGENTA = (235, 70, 200)
LINE = (255, 200, 60)


def project(points_world, cam2world, K):
    """(u, v, z) of world points in an OpenCV-convention camera."""
    world2cam = np.linalg.inv(cam2world)
    cam = points_world @ world2cam[:3, :3].T + world2cam[:3, 3]
    z = np.clip(cam[:, 2], 1e-6, None)
    u = cam[:, 0] / z * K[0, 0] + K[0, 2]
    v = cam[:, 1] / z * K[1, 1] + K[1, 2]
    return u, v, cam[:, 2]


def dots(draw, u, v, colour, r):
    for x, y in zip(u, v):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help=".npz written by run_pointworld_on_episode.py --save-pred")
    ap.add_argument("--out", default=os.path.join("videos", "point_flow.mp4"))
    ap.add_argument("--cam", type=int, default=0, help="which recorded camera to draw in")
    args = ap.parse_args()

    import imageio

    p = np.load(args.pred, allow_pickle=True)
    ep = np.load(str(p["episode"]), allow_pickle=True)

    centre = p["centre"]
    pred = p["pred"] + centre          # back to world; the model works centred
    gt = p["gt"] + centre
    moved = p["moved"]
    T = gt.shape[0]

    rgb = ep["rgb"][0, args.cam]
    K = ep["intrinsic"][0, args.cam]
    cam2world = ep["extrinsic"][0, args.cam]
    H, W = rgb.shape[:2]

    base = Image.fromarray((rgb * 0.35).astype(np.uint8)).resize(
        (W * SCALE, H * SCALE), Image.NEAREST)

    frames = []
    for t in range(T):
        img = base.copy()
        d = ImageDraw.Draw(img)

        ug, vg, _ = project(gt[t], cam2world, K)
        up, vp, _ = project(pred[t], cam2world, K)
        ug, vg, up, vp = ug * SCALE, vg * SCALE, up * SCALE, vp * SCALE

        # The static majority first and faintly, so it never hides the drawer
        # but its spurious motion is still visible.
        dots(d, up[~moved], vp[~moved], GREY, 1)
        for x0, y0, x1, y1 in zip(ug[moved], vg[moved], up[moved], vp[moved]):
            d.line([x0, y0, x1, y1], fill=LINE, width=1)
        dots(d, ug[moved], vg[moved], GREEN, 2)
        dots(d, up[moved], vp[moved], MAGENTA, 2)

        err = np.linalg.norm(pred[t] - gt[t], axis=1)
        d.text((6, 6), f"step {t}/{T - 1}", fill=(255, 255, 255))
        d.text((6, 18), f"moved pts: {err[moved].mean() * 1000:5.1f} mm err", fill=(255, 255, 255))
        d.text((6, 30), f"true motion: "
                        f"{np.linalg.norm(gt[t] - gt[0], axis=1)[moved].mean() * 1000:5.1f} mm",
               fill=(255, 255, 255))
        d.text((6, H * SCALE - 34), "green = true", fill=GREEN)
        d.text((6, H * SCALE - 22), "magenta = PointWorld", fill=MAGENTA)
        d.text((6, H * SCALE - 10), "grey = predicted, static points", fill=GREY)

        frames.extend([np.asarray(img)] * (HOLD if t else HOLD * 2))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=FPS)
    print(f"{args.out}: {len(frames)} frames, {T} steps, "
          f"{int(moved.sum())}/{len(moved)} moved points")
    print(f"final step: PointWorld {p['err_moved']*1000:.1f} mm on moved points "
          f"(static baseline {p['static_moved']*1000:.1f} mm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
