"""Grounded test-time action selection. No fine-tuning, no model changes.

Two grounding channels were measured dead on this hardware: serialised boxes in
the prompt (inert), and pixel overlays on an untrained policy (no redirection).
Both fail the same way -- they are INPUTS, and the policy is free to ignore an
input. This puts grounding where it cannot be ignored: the CHOICE.

    sample K action chunks -> score each against the grounded target -> execute
    the winner

Framework follows MG-Select (Verifier-free Test-Time Sampling for VLAs, ICLR
2026, arXiv:2510.05681): K candidates, pick one, no training and no extra
module. What differs is the selector -- theirs is the model's own confidence,
ours is grounding -- and that difference IS the experiment.

WHY THIS CAN WORK HERE, FROM OUR OWN MEASUREMENTS. GR00T's flow-matching head
is extremely stochastic: eight IDENTICAL queries produced chunks up to 150
degrees apart (noise floor cos_dir -0.87 on single draws). That variance ruined
the probe and is exactly the diversity selection needs. pi-0.5 was rejected for
this: its sampling is narrow, so there would be nothing to choose between.

THE SCORER IS DIRECTIONAL, AND THAT IS NOT A SIMPLIFICATION.
`default_pandaomron.json` sets the arm to OSC_POSE with
`input_type=delta, input_ref_frame=base`, so a candidate is 16 BASE-FRAME
DELTAS -- not absolute waypoints, whatever the checkpoint's `rep=ABSOLUTE`
normalisation flag suggests. Integrating them would walk into this repo's
documented trap: an OSC controller realises only ~25% of each commanded delta,
so the integral is a pose the arm never occupies. A cosine between the summed
intent and the direction to the target is scale-free, so the realisation ratio
cancels entirely.

FRAMES, VERIFIED RATHER THAN ASSUMED:
  * `state.end_effector_position_absolute` == the sim grip site  -> world
  * `base_rotation` is xyzw, and
    `R(base_rotation)^T @ (abs - base_position)` reproduces
    `end_effector_position_relative` to 0.00000
Every frame error in this repo's history came from skipping a check like that.

ARMS. `--mode` selects which:
  none      K=1, no selection. Reproduces the published baseline.
  mean      pick the candidate closest to the mean of K. Verifier-free control,
            MG-Select's spirit without token internals a flow-matching head
            does not expose. THIS IS THE ARM THAT MATTERS: it separates
            "selection helps because averaging suppresses sampler noise" from
            "grounding helps". Without it a win under `grounded` is unattributable.
  grounded  cosine against the grounded target.
"""
import numpy as np

#: Action head carrying end-effector translation. The others are left alone:
#: the gripper is a discrete decision the probes found transfers unchanged, and
#: re-ranking on it would add noise to a channel that is already correct.
#:
#: BOTH SPELLINGS OCCUR. The server returns modality keys unprefixed
#: (`end_effector_position`, per its own get_modality_config) while the gym
#: action space uses `action.end_effector_position`. Which one arrives depends
#: on where in the stack you are, so resolve it per call rather than assuming.
EE_KEYS = ("action.end_effector_position", "end_effector_position")


def ee_key(chunk):
    for k in EE_KEYS:
        if k in chunk:
            return k
    raise KeyError(f"no end-effector translation head in {sorted(chunk)}")


def quat_xyzw_to_R(q):
    """xyzw quaternion -> 3x3 rotation. Ordering verified against the sim."""
    x, y, z, w = (float(v) for v in np.asarray(q).ravel()[:4])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def world_to_base(p_world, base_pos, base_rot_xyzw):
    """World point -> the base frame the ACTIONS are expressed in."""
    return quat_xyzw_to_R(base_rot_xyzw).T @ (
        np.asarray(p_world, float).ravel() - np.asarray(base_pos, float).ravel())


def intent(chunk):
    """A candidate's summed translation: where this chunk is trying to go.

    NOT a position. Summing deltas gives a direction of intent and nothing
    more -- see the module docstring on OSC realisation.
    """
    return np.asarray(chunk[ee_key(chunk)], float).reshape(-1, 3).sum(axis=0)


def cos(a, b, eps=1e-9):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < eps or nb < eps else float(
        np.clip(a.dot(b) / (na * nb), -1.0, 1.0))


class GroundedSelector:
    """Wraps a PolicyClient: sample K, score, return the winner.

    Exposes `get_action(obs)` so it drops into GR00T's rollout harness in place
    of the bare client, with no change to the harness itself.
    """

    def __init__(self, client, mode="grounded", k=8, target_fn=None,
                 log=None):
        self.client = client
        self.mode = mode
        self.k = 1 if mode == "none" else k
        self.target_fn = target_fn      # () -> target position in WORLD frame
        self.log = log if log is not None else []

    # The harness may call either; keep both pointing at the same logic.
    def reset(self):
        pass

    def get_modality_config(self):
        return self.client.get_modality_config()

    @staticmethod
    def nest(obs):
        """Flat harness observation -> the nested dict the server validates.

        `rollout_policy` hands out FLAT dotted keys (`video.res256_image_side_0`)
        while `Gr00tPolicy._validate` requires three top-level modality dicts and
        rejects anything else with "Observation must contain a 'video' key".
        There is no adapter between them in this release, so the selector does
        it.

        This is a REGROUPING, not a reshape: MultiStepWrapper already stacks to
        (T, ...) and SyncVectorEnv adds the env axis, giving exactly the
        (B, T, ...) the server expects. Video and state keys drop their prefix;
        language keeps its full dotted name -- the asymmetry was read off the
        server's own `get_modality_config`, not guessed.
        """
        out = {"video": {}, "state": {}, "language": {}}
        for k, v in obs.items():
            if k.startswith("video."):
                out["video"][k[len("video."):]] = np.asarray(v)
            elif k.startswith("state."):
                out["state"][k[len("state."):]] = np.asarray(v, np.float32)
            elif k.startswith("annotation."):
                s = v
                # Arrive as arrays/lists of bytes or str depending on the
                # wrapper; the server wants [[str]] shaped (B, T).
                flat = np.asarray(s, dtype=object).ravel().tolist()
                flat = [x.decode() if isinstance(x, bytes) else str(x)
                        for x in flat]
                out["language"][k] = [flat] if flat else [[""]]
        return out

    def _sample(self, obs):
        nested = self.nest(obs)
        out = []
        for _ in range(self.k):
            a = self.client.get_action(nested)
            out.append(a[0] if isinstance(a, tuple) else a)
        return out

    @staticmethod
    def prefix(chunk):
        """Server action keys -> the gym action space's keys.

        The server answers with bare modality names (`gripper_close`) but the
        env's action space is keyed `action.gripper_close`, and
        SyncVectorEnv raises KeyError on the mismatch. Re-prefix on the way
        out; already-prefixed keys pass through so this is idempotent.
        """
        return {(k if k.startswith("action.") else f"action.{k}"): v
                for k, v in chunk.items()}

    def get_action(self, obs):
        # The harness does `actions, _ = policy.get_action(obs)`, so a
        # bare dict would unpack into its keys and corrupt the action.
        return self.prefix(self._choose(obs)), {}

    def _choose(self, obs):
        cands = self._sample(obs)
        if len(cands) == 1:
            return cands[0]

        vs = [intent(c) for c in cands]
        # Spread of the candidates. If this collapses, selection cannot help
        # and the premise of the whole approach is void -- so it is logged
        # every step rather than assumed from the probe's measurement.
        mean_v = np.mean(vs, axis=0)
        spread = float(np.mean([1.0 - cos(v, mean_v) for v in vs]))

        if self.mode == "mean":
            idx = int(np.argmax([cos(v, mean_v) for v in vs]))
            score = None
        else:
            tgt_w = self.target_fn() if self.target_fn else None
            if tgt_w is None:
                # Target not visible this step: fall back to the verifier-free
                # rule rather than scoring against nothing. Recorded, because a
                # run where this fires often is not testing grounding.
                idx = int(np.argmax([cos(v, mean_v) for v in vs]))
                score = None
                self.log.append({"spread": spread, "grounded": False})
                return cands[idx]
            # The harness hands over BATCHED, TIME-STACKED observations
            # (n_envs, T, D) via MultiStepWrapper. Take the LATEST timestep:
            # indexing the flattened array would silently score against the
            # oldest frame in the stack, which is a stale pose.
            def latest(key, d):
                return np.asarray(obs[key], float).reshape(-1, d)[-1]

            base_pos = latest("state.base_position", 3)
            base_rot = latest("state.base_rotation", 4)
            eef_b = latest("state.end_effector_position_relative", 3)
            want = world_to_base(tgt_w, base_pos, base_rot) - eef_b
            scores = [cos(v, want) for v in vs]
            idx = int(np.argmax(scores))
            score = float(scores[idx])

        self.log.append({"spread": spread, "grounded": score is not None,
                         "best": score})
        return cands[idx]
