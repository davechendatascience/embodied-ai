"""Run PointWorld on a recorded MuJoCo episode and score it against truth.

The first honest measurement of whether PointWorld's dynamics hold on OUR
scenes. Everything up to here was plumbing; this is the number.

The episode carries exact ground truth (see record_pointworld_episode.py): each
scene point is bound to a MuJoCo body and pushed through that body's real pose
at every step, so "where did point i actually go" is known rather than
annotated. The model is given step 0 plus the robot's point flow, and must
predict the rest.

Reported against two baselines, because a raw error is uninterpretable:
  STATIC    predict nothing moves. On a mostly-static tabletop this is a
            strong baseline and easy to beat by accident.
  ORACLE    0 by construction.
The number that matters is error on the points that ACTUALLY MOVED.

This drives `BaseModel` (see src/pointworld_bridge/model.py for why that is
not a detail). The one input the release does not pin down is the polarity of
the `gripper_open` channel, so `--sweep-gripper` runs it both ways and prints
the difference rather than quietly picking one.

THE MODEL IS NOT DETERMINISTIC IN EVAL MODE. `build_ptv3` hardcodes
`shuffle_orders=True` (`pointworld/base.py:198`), and that reaches
`torch.randperm` inside `Point.serialization` (`ptv3/structure.py:98`) on every
forward pass, permuting which of the four space-filling-curve orders each
attention block sees. `model.eval()` does not touch it. Two back-to-back runs
of this file differed by 5 mm on the moved points before that was found, which
is the same size as the effects being measured. So every forward here is
seeded, and the spread across seeds is REPORTED rather than hidden.

    CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \
        .venv-pw/bin/python tests/run_pointworld_on_episode.py \
            data/pw_episodes/libero_goal_0_ep0.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pointworld_bridge.episode import build_data_dict, load_episode  # noqa: E402
from pointworld_bridge.model import load_base_model  # noqa: E402

MOVED_THRESHOLD = 0.002    # 2 mm of true displacement over the window


def score(pred, gt, moved):
    """(T-1, Ns) L2 error, and its mean over all points / over the moved ones."""
    err = np.linalg.norm(pred[1:] - gt[1:], axis=-1)
    return err, err.mean(), err[:, moved].mean()


def ablated_features(model, data_dict, what):
    """Scene features with one half switched off, to see which half is load-bearing.

    `SceneFeatureEncoder.forward` fuses two things (`scene_featurizer.py:377`):
    the DINOv3 backbone projected onto the points, and the 31 raw geometry
    channels. Replacing one with zeros and re-running says how much each is
    actually contributing -- which is not knowable by inspection, and matters
    because "we use DINOv3" is only a meaningful claim if removing it changes
    the answer.
    """
    enc = model.scene_feature_encoder
    raw = enc.normalize_scene_features(data_dict["scene_features"][:, 0])
    coord = data_dict["scene_flows"][:, 0]
    exists = data_dict["scene_exists"][:, 0]

    backbone = enc.scene_encoder(coord, exists, enc.scene_encoder._extract_camera_data(data_dict))
    backbone = backbone.to(raw.dtype)
    if what == "dinov3":
        backbone = torch.zeros_like(backbone)
    raw_part = enc.scene_raw_norm(enc.scene_raw_feat_proj(raw))
    if what == "geometry":
        raw_part = torch.zeros_like(raw_part)
    return enc.scene_proj(torch.cat([enc.scene_encoder_norm(backbone), raw_part], dim=-1))


def run(model, data_dict, seed=0, ablate=None):
    """One seeded forward pass. See the module docstring for why the seed matters."""
    torch.manual_seed(seed)
    with torch.no_grad():
        if ablate:
            # forward() sets _current_domain_indices, which normalisation needs.
            model(data_dict, training=False)
            feat = ablated_features(model, data_dict, ablate)
            out = model(data_dict, training=False, encoded_scene_feat0=feat)
        else:
            out = model(data_dict, training=False)
    pred = out["scene_flows"][0].float().cpu().numpy()      # (T,Ns,3) absolute
    conf = out["confidence"][0].float().cpu().numpy()       # (T,Ns)
    return pred, conf


def run_many(model, data_dict, gt, moved, repeats, ablate=None):
    """Predictions from `repeats` seeds, plus the per-seed scores."""
    preds, alls, moveds = [], [], []
    for seed in range(repeats):
        pred, conf = run(model, data_dict, seed, ablate=ablate)
        _, a, m = score(pred, gt, moved)
        preds.append(pred)
        alls.append(a)
        moveds.append(m)
    return np.array(preds), np.array(alls), np.array(moveds), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?",
                    default=str(ROOT / "data" / "pw_episodes" / "libero_goal_0_ep0.npz"))
    ap.add_argument("--sweep-gripper", action="store_true",
                    help="also run with the gripper_open channel inverted")
    ap.add_argument("--save-pred", metavar="PATH",
                    help="write the prediction to .npz for scripts/render_point_flow.py, "
                         "which runs in the OTHER venv")
    ap.add_argument("--ablate", choices=["dinov3", "geometry"],
                    help="zero one half of the scene features and re-score, to "
                         "see which half the result actually rests on")
    ap.add_argument("--repeats", type=int, default=5,
                    help="seeded forward passes; PTv3 shuffles its serialization "
                         "orders even in eval, so one pass is not a measurement")
    args_cli = ap.parse_args()

    dev = torch.device("cuda")
    ep = load_episode(args_cli.episode)
    model, args, _ = load_base_model(device=dev)

    data_dict, meta = build_data_dict(ep, args, dev)
    gt = data_dict["scene_flows"][0].float().cpu().numpy()   # (T,Ns,3) centred frame
    Ns, T = meta["Ns"], meta["T"]
    print(f"episode : {Path(args_cli.episode).name}  scene ({T},{Ns},3) "
          f"robot ({T},{meta['Nr']},3)  centred by "
          f"[{', '.join(f'{v:+.3f}' for v in meta['centre'])}] m")
    print(f"          {meta['cameras']} camera(s), rgb {tuple(ep['rgb'].shape[2:])}, gripper_open="
          f"{meta['gripper_open']:.0f}, truncated_at={int(ep['truncated_at'][0])}")

    moved = np.linalg.norm(gt[-1] - gt[0], axis=1) > MOVED_THRESHOLD
    preds, alls, moveds, conf = run_many(model, data_dict, gt, moved,
                                         args_cli.repeats, ablate=args_cli.ablate)
    if args_cli.ablate:
        print(f"ABLATED : {args_cli.ablate} features zeroed\n")
    pred = preds.mean(0)

    # Sanity: the model is handed step 0, so it must reproduce it exactly.
    t0 = np.abs(preds[:, 0] - gt[0]).max()
    assert t0 < 1e-6, f"t=0 is an input, not a prediction, but differs by {t0:.2e} m"

    err, all_mm, moved_mm = score(pred, gt, moved)
    static = np.repeat(gt[0][None], T, axis=0)
    s_err, s_all, s_moved = score(static, gt, moved)

    print(f"\n{'':26s} {'all pts':>10s} {'moved pts':>12s}")
    print(f"{'PointWorld, seed mean (mm)':26s} {alls.mean() * 1000:9.2f} "
          f"{moveds.mean() * 1000:11.2f}")
    print(f"{'   spread over seeds (mm)':26s} {alls.std() * 1000:9.2f} "
          f"{moveds.std() * 1000:11.2f}   "
          f"[{moveds.min() * 1000:.2f}-{moveds.max() * 1000:.2f}] over "
          f"{args_cli.repeats} seeds")
    print(f"{'   seed-averaged pred (mm)':26s} {all_mm * 1000:9.2f} {moved_mm * 1000:11.2f}")
    print(f"{'static baseline (mm)':26s} {s_all * 1000:9.2f} {s_moved * 1000:11.2f}")
    print(f"\nmoved points: {int(moved.sum())}/{Ns}  "
          f"(max true motion {np.linalg.norm(gt - gt[0], axis=-1).max() * 1000:.1f} mm)")
    print(f"mean confidence: {conf[1:].mean():.3f}")

    # WHERE the predicted motion goes matters more than its size. A model that
    # has understood the interaction moves the drawer and leaves the table
    # alone; one that has not spreads motion over the whole cloud and can still
    # score well on a mean.
    disp = np.linalg.norm(pred[1:] - gt[0][None], axis=-1)
    true_disp = np.linalg.norm(gt[1:] - gt[0][None], axis=-1)
    print(f"\npredicted motion, final step:  moved pts {disp[-1, moved].mean() * 1000:6.1f} mm "
          f"(true {true_disp[-1, moved].mean():.3f} m)   "
          f"static pts {disp[-1, ~moved].mean() * 1000:6.1f} mm (true 0.0)")
    d_pred = (pred[-1] - gt[0])[moved]
    d_true = (gt[-1] - gt[0])[moved]
    cos = (d_pred * d_true).sum(-1) / (np.linalg.norm(d_pred, axis=-1)
                                       * np.linalg.norm(d_true, axis=-1) + 1e-12)
    print(f"direction on moved pts: cos(pred, true) = {cos.mean():+.3f} "
          f"(1.0 = right way, 0 = unrelated)")

    # Error against the horizon. A dynamics model that has learnt anything
    # should track the ramp of true motion rather than a flat offset.
    print(f"\n{'step':>4s} {'true motion':>12s} {'PointWorld':>11s} {'static':>9s}   (moved pts, mm)")
    for t in range(1, T):
        print(f"{t:4d} {true_disp[t - 1, moved].mean() * 1000:11.2f} "
              f"{err[t - 1, moved].mean() * 1000:11.2f} "
              f"{s_err[t - 1, moved].mean() * 1000:9.2f}")

    better = moveds.mean() < s_moved
    print(f"\nverdict : PointWorld {'BEATS' if better else 'does NOT beat'} "
          f"predict-no-motion on the points that moved "
          f"({moveds.mean() * 1000:.2f} vs {s_moved * 1000:.2f} mm), and is "
          f"{'BETTER' if alls.mean() < s_all else 'WORSE'} on the scene as a whole "
          f"({alls.mean() * 1000:.2f} vs {s_all * 1000:.2f} mm)")

    if args_cli.save_pred:
        Path(args_cli.save_pred).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args_cli.save_pred,
            pred=pred.astype(np.float32),        # (T,Ns,3) centred frame, seed-averaged
            gt=gt.astype(np.float32),
            moved=moved,
            centre=meta["centre"],               # world = centred + centre
            episode=str(args_cli.episode),
            err_all=np.float32(alls.mean()),
            err_moved=np.float32(moveds.mean()),
            err_moved_std=np.float32(moveds.std()),
            static_all=np.float32(s_all),
            static_moved=np.float32(s_moved),
        )
        print(f"\nwrote prediction to {args_cli.save_pred}")

    if args_cli.sweep_gripper:
        flipped = 1.0 - meta["gripper_open"]
        dd2, _ = build_data_dict(ep, args, dev, gripper_open=flipped)
        _, a2, m2, _ = run_many(model, dd2, gt, moved, args_cli.repeats)
        print(f"\ngripper_open={flipped:.0f} instead: {a2.mean() * 1000:.2f} mm all, "
              f"{m2.mean() * 1000:.2f} mm moved "
              f"(delta {(m2.mean() - moveds.mean()) * 1000:+.2f} mm on the moved "
              f"points, against a seed spread of {moveds.std() * 1000:.2f} mm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
