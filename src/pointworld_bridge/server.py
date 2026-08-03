"""PointWorld as a resident service, so a control loop can reach it from `.venv`.

The two stacks cannot share a process (mujoco 3.1.6 + numpy 1.26 against torch
2.11 + numpy 2.5), and disk is fine for scoring an episode and hopeless for a
loop that has to answer every tick. So the model lives here, on the GPU, and
LIBERO talks to it over a UNIX socket.

The split follows the measured cost, not tidiness:

    observe   41 ms   DINOv3 + projection. ONCE per camera observation.
    rollout   29 ms   PTv3 + heads. Once per candidate trajectory.

`BaseModel.forward` takes `encoded_scene_feat0` precisely so perception can be
reused, so `observe` caches it and `rollout` never recomputes it. A loop that
re-ran perception per candidate would pay 70 ms instead of 29 -- measured, see
tests/bench_pointworld_realtime.py.

Candidates are evaluated in BATCHES. They share a scene, so the only thing
that differs between them is the robot point flow, and the GPU would rather
see one batch than K sequential calls.

    CUMM_CUDA_VERSION=13.0 CUMM_CUDA_ARCH_LIST=12.0 PYTHONPATH=src \
        .venv-pw/bin/python -m pointworld_bridge.server
"""

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .episode import build_data_dict
from .fast_features import FastFeatures
from .model import load_base_model
from .protocol import DEFAULT_SOCKET, recv_frame, send_frame


class PointWorldService:
    """Holds the model and the cached perception for the current observation."""

    def __init__(self, device="cuda", max_batch=8):
        self.device = torch.device(device)
        self.model, self.args, _ = load_base_model(device=self.device, verbose=True)
        self.max_batch = max_batch
        self.obs = None            # the arrays from the last `observe`
        self.scene_feat = None     # (1, Ns, C) encoded, reused by every rollout
        self.centre = None
        self.T = None
        self._warm = False         # one graph-warming forward, not one per observe

    # ---------------------------------------------------------------- observe
    def observe(self, header, arrays):
        """Cache one camera observation and its DINOv3 features.

        `arrays` carries exactly what the recorder writes, so a live
        observation and a recorded episode take the same path into the model
        and cannot silently diverge.
        """
        t0 = time.perf_counter()
        data_dict, meta = build_data_dict(arrays, self.args, self.device)
        t_assemble = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            # ONCE, not per observation. This full forward was here because
            # `encode_scene_features` needs `_current_domain_indices` -- but it
            # sets that itself when the attribute is absent (`base.py:422`), so
            # the only thing the forward still bought was graph warmup, and that
            # is a one-time cost by definition. Running it every observation
            # paid for a whole PTv3 trunk and dynamics head plus a SECOND DINOv3
            # encode, on a call whose entire job is to encode the scene once.
            if not self._warm:
                self.model(data_dict, training=False)
                self._warm = True
            # `encode_scene_features` only derives the domain indices when the
            # attribute is ABSENT, and the previous `rollout` left them sized to
            # its K candidates. Dropping them here is what makes the cheap path
            # correct: observe is always B=1, so a stale K-length tensor from
            # the last tick asserts. `forward()` re-derives them per call, so
            # the rollout path is unaffected.
            if hasattr(self.model, "_current_domain_indices"):
                del self.model._current_domain_indices
            self.scene_feat = self.model.encode_scene_features(data_dict)
        torch.cuda.synchronize()
        t_encode = time.perf_counter() - t0

        # Candidate assembly moves to the GPU here. The reference pipeline runs
        # ONCE, to produce the invariant channels; per-candidate work is then
        # only the four action-dependent groups. Measured 10.2 -> 1.8 ms per
        # candidate, verified equal by tests/test_fast_features.py.
        n_grip = 2 if data_dict["robot_features"].shape[-1] == 17 else 1
        self.fast = FastFeatures(data_dict, meta, self.args, self.device, n_grip)
        self.obs = {k: v for k, v in arrays.items()}
        self.centre = meta["centre"]
        # Everything after this point must reuse THIS centre. The cached scene
        # features describe frame 0, so a chunk that re-derived its own centre
        # would silently pair features with coordinates from another frame.
        self.T = meta["T"]
        return ({"ok": True, "Ns": meta["Ns"], "Nr": meta["Nr"], "T": meta["T"],
                 "cameras": meta["cameras"],
                 "assemble_ms": t_assemble * 1e3, "encode_ms": t_encode * 1e3},
                {"centre": self.centre})

    # ---------------------------------------------------------------- rollout
    def rollout(self, header, arrays):
        """Score candidate gripper trajectories against a target specification.

        `robot_flows` is (K, T, Nr, 3) in WORLD coordinates -- the same frame
        the recorder writes. Centring happens inside `build_data_dict`; handing
        it already-centred points centres them twice and puts the gripper a
        metre from the scene, which reads exactly like "the model cannot rank
        actions" (NOTES.md section 4).

        The target is `goal_idx` (which scene points) and `goal_pos` (where
        they should end up, world frame). Scoring here rather than on the
        client keeps an 11 MB prediction tensor off the socket.
        """
        if self.obs is None:
            raise RuntimeError("rollout before observe")

        cand = arrays["robot_flows"]                     # (K, H+1, Nr, 3) world
        if cand.ndim == 3:
            cand = cand[None]
        K = len(cand)
        goal_idx = arrays.get("goal_idx")
        goal_pos = arrays.get("goal_pos")
        want_pred = bool(header.get("return_pred", False))

        # The paper's MPC rolls 30 steps as three chained 10-step passes, each
        # starting from the previous pass's predicted points. One chunk is a
        # third of the intended horizon, which makes the planner ask for far
        # more travel per step than the goal needs.
        span = self.T - 1
        n_chunks = max(1, (cand.shape[1] - 1) // span)

        t_assemble = t_model = 0.0
        scene = None                                     # (K, Ns, 3) WORLD frame
        pred_chunks = []
        for c in range(n_chunks):
            lo, hi = c * span, c * span + self.T
            t0 = time.perf_counter()
            # The frame stays pinned to the observation's centre for every
            # chunk; re-deriving it would drift as the points move while the
            # cached scene features still describe frame 0.
            batched = self.fast.batch(cand[:, lo:hi], scene_flows=scene)
            torch.cuda.synchronize()
            t_assemble += time.perf_counter() - t0

            t0 = time.perf_counter()
            out = []
            with torch.no_grad():
                for b in range(0, K, self.max_batch):
                    out.append(self._forward_slice(batched, b, min(b + self.max_batch, K)))
            torch.cuda.synchronize()
            t_model += time.perf_counter() - t0
            step = np.concatenate(out, axis=0)           # (K,T,Ns,3) centred
            pred_chunks.append(step if c == 0 else step[:, 1:])
            scene = step[:, -1] + self.centre            # hand the world frame back

        pred = np.concatenate(pred_chunks, axis=1)       # (K, H+1, Ns, 3) centred

        out = {}
        head = {"ok": True, "K": K, "chunks": n_chunks, "horizon": pred.shape[1] - 1,
                "assemble_ms": t_assemble * 1e3, "model_ms": t_model * 1e3,
                "per_candidate_ms": t_model * 1e3 / max(K, 1)}
        if goal_idx is not None and goal_pos is not None:
            # centred -> world before comparing, so the client never has to
            # know the frame the model works in.
            gi = goal_idx.astype(np.int64)
            # Cost at EVERY step, not only the last. A terminal-only cost cannot
            # express "arrive and stop": a candidate that reaches the goal at
            # step 3 and sails 40 mm past it by step 10 scores as badly as one
            # that never moves, so the planner picks "still" and never moves
            # again. Measured: both structured runs froze exactly there, with
            # margins of 3 mm against a 0.46 mm noise floor.
            # (K, T) is a few hundred floats -- free next to the prediction it
            # is derived from, which is why this is computed here rather than
            # shipping `pred` over the socket.
            steps = np.linalg.norm(
                pred[:, :, gi] + self.centre - goal_pos[None, None], axis=-1
            ).mean(axis=-1)                              # (K, T)
            # COLLISION, from the world model rather than from geometry.
            # The cost above sees ONLY the target's points, so a candidate that
            # sweeps the bottle off the table scores exactly as well as one
            # that does not -- there is no collision term anywhere in this
            # stack, and the guide flags that as a subsystem that must be added
            # externally. But it need not be geometric: the model already
            # predicts the WHOLE scene, so disturbing a bystander is simply
            # predicted motion where none was asked for.
            #
            # Referenced against `pred[:, 0]` rather than the observed cloud on
            # purpose: the model leaks spurious motion onto static points
            # (14.6 mm on small-droid, 60.6 mm on large), and differencing
            # within the same prediction cancels the part of that bias that is
            # common to every candidate, leaving what the ACTION did.
            avoid = arrays.get("avoid_idx")
            w = float(header.get("avoid_weight", 0.0) or 0.0)
            if w > 0.0 and avoid is not None and len(avoid):
                ai = avoid.astype(np.int64)
                disturb = np.linalg.norm(
                    pred[:, :, ai] - pred[:, :1, ai], axis=-1).mean(axis=-1)
                head["disturb_mm"] = float(disturb[:, -1].mean() * 1000)
                steps = steps + w * disturb

            cost = steps[:, -1]                          # terminal, unchanged
            out["cost"] = cost.astype(np.float32)
            out["cost_steps"] = steps.astype(np.float32)
            head["best"] = int(np.argmin(cost))
            head["best_cost"] = float(cost.min())
        if want_pred:
            out["pred"] = (pred + self.centre).astype(np.float32)
        return head, out

    def _forward_slice(self, batched, lo, hi):
        """Run candidates [lo, hi) of an already-batched dict."""
        n = hi - lo
        sl = {k: (v[lo:hi] if torch.is_tensor(v) else v[lo:hi] if isinstance(v, list) else v)
              for k, v in batched.items()}
        feat = self.scene_feat.expand(n, -1, -1)
        out = self.model(sl, training=False, encoded_scene_feat0=feat)
        return out["scene_flows"].float().cpu().numpy()

    # -------------------------------------------------------------- features
    def features(self, header, arrays):
        """The per-point DINOv3 features from the last `observe`, (Ns, C).

        THE MASK IS THE BIGGEST REMAINING CHEAT -- `task_spec.py` selects the
        target's points with MuJoCo collision geometry -- and these features
        are the substrate any replacement has to query. PointWorld already
        computes them for every scene point on every observation, so exposing
        them costs a memcpy rather than a second encoder.

        Deliberately NOT recomputed: handing back the cached `scene_feat` is
        what keeps the grounding layer and the dynamics layer looking at the
        same points, in the same order, in the same frame. A grounder running
        its own encoder could disagree with the planner about which point is
        which, and nothing would report it.
        """
        if self.scene_feat is None:
            raise RuntimeError("features before observe")
        f = self.scene_feat[0].float().cpu().numpy()
        return ({"ok": True, "Ns": int(f.shape[0]), "C": int(f.shape[1])},
                {"features": np.ascontiguousarray(f, dtype=np.float32)})

    # ------------------------------------------------------------------ ping
    def ping(self, header, arrays):
        return {"ok": True, "device": str(self.device),
                "torch": torch.__version__, "numpy": np.__version__}, {}


HANDLERS = ("ping", "observe", "rollout", "features")


def serve(path=DEFAULT_SOCKET, device="cuda", max_batch=8, seed=0):
    # PTv3 shuffles serialization orders under torch.randperm even in eval
    # (NOTES.md section 4). Seed once here so the service is reproducible;
    # a client that wants the spread can re-send and average.
    torch.manual_seed(seed)
    service = PointWorldService(device=device, max_batch=max_batch)

    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    print(f"listening on {path} (max_batch={max_batch})", flush=True)

    try:
        while True:
            conn, _ = srv.accept()
            print("client connected", flush=True)
            try:
                while True:
                    try:
                        header, arrays = recv_frame(conn)
                    except ConnectionError:
                        break
                    op = header.get("op")
                    try:
                        if op not in HANDLERS:
                            raise ValueError(f"unknown op {op!r}; expected {HANDLERS}")
                        rh, ra = getattr(service, op)(header, arrays)
                    except Exception as exc:  # noqa: BLE001 - report, do not die
                        rh, ra = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, {}
                    send_frame(conn, rh, ra)
            finally:
                conn.close()
                print("client disconnected", flush=True)
    finally:
        srv.close()
        if os.path.exists(path):
            os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batch", type=int, default=8)
    args = ap.parse_args()
    serve(args.socket, args.device, args.max_batch)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
