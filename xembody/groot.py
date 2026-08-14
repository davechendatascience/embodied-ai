"""GR00T N1.6 behind its zmq policy server, shaped like the pi-0.5 adapter.

Why this exists: pi05_libero is fine-tuned on LIBERO and is not competent on
RoboCasa. Measured, on three RoboCasa tasks, eight identical queries each --
the baseline disagreed with itself by up to 110 degrees, so no prompt
manipulation of any kind could be distinguished from the sampler and the
grounding question could not be asked. GR00T-N1.6-3B is evaluated zero-shot on
RoboCasa at 66.22% average over 24 tasks, which is the competence the
measurement needs.

THE GROUNDING INJECTION POINT IS THE SAME AS pi-0.5's, which is why the whole
harness transfers. GR00T's RoboCasa observation carries the instruction as a
plain string under `annotation.human.action.task_description`, so appending a
serialised box segment to it is the identical intervention the LIBERO probe
made -- same conditions, same noise floor, same verdict bands, and therefore
comparable across the two model families.

TWO DIFFERENCES FROM pi-0.5 THAT MATTER:

  1. The observation is a DICT OF NAMED MODALITIES (`video.*`, `state.*`,
     `annotation.*`), not four fixed keys. It is built by the environment
     wrapper, so this adapter takes the wrapper's dict and edits one field
     rather than assembling one -- editing is the intervention, assembling
     would be a second uncontrolled change.
  2. The action comes back as a DICT of named action heads, not one array. The
     metrics need a single (T, D) matrix, so the heads are concatenated in
     SORTED KEY ORDER. That order is arbitrary but it must be STABLE: the
     metrics compare chunks elementwise, and a dict iteration order that
     changed between two conditions would register as a behavioural difference.
"""

import numpy as np

#: Where the instruction lives in a GR00T RoboCasa observation.
LANG_KEY = "annotation.human.action.task_description"


#: Heads that must come FIRST, in this order, when flattening an action.
#: THIS IS NOT COSMETIC. xembody.probe's headline metric is `chunk_dir`, the
#: summed TRANSLATION of the chunk, and it reads columns 0:3 -- a layout
#: inherited from LIBERO's [world_vector(3), rotation_delta(3), gripper(1)].
#: GR00T returns named heads, and sorting them alphabetically puts `base_motion`
#: first. On a stationary manipulation task the mobile base does not move, so
#: columns 0:3 were all zero and EVERY comparison returned cos_dir=0.0000,
#: d_dir=0.0000 -- including the noise floor. That reads exactly like "the
#: policy ignores the prompt" and is really "the metric is reading the wrong
#: channel". Pin the end-effector translation to columns 0:3 instead.
PREFERRED_HEADS = ("end_effector_position", "end_effector_rotation",
                   "gripper_close")


def flatten_action(action, order=None):
    """{head: (T, d)} -> (T, D), concatenated in a STABLE, MEANINGFUL order.

    Returns (matrix, order) so a caller can pin the order across queries. Heads
    that are 1-D are treated as (T, 1). See PREFERRED_HEADS for why the order
    is not simply sorted.
    """
    if order is None:
        first = [k for k in PREFERRED_HEADS if k in action]
        order = first + sorted(k for k in action if k not in first)
    keys = order
    cols = []
    for k in keys:
        v = np.asarray(action[k], dtype=float)
        cols.append(v.reshape(len(v), -1) if v.ndim > 1 else v.reshape(-1, 1))
    return np.concatenate(cols, axis=1), keys


def nest_observation(flat, prompt):
    """Flat gym observation -> the nested, batched dict the server validates.

    The RoboCasa gym wrapper emits FLAT dotted keys -- `video.res256_image_side_0`,
    `state.gripper_qpos` -- while `Gr00tPolicy._validate` requires three
    top-level modality dicts:

        video     {key: (B, T, H, W, C)}   key WITHOUT the "video." prefix
        state     {key: (B, T, D)}         key WITHOUT the "state." prefix
        language  {key: [[str]]}           key WITH its full dotted name

    The prefix asymmetry is not a guess: the server was asked. Its
    `get_modality_config` reports video keys as `res256_image_side_0` and
    language as `annotation.human.action.task_description`, so video and state
    are stripped and language is not.

    Batch and time axes are both 1 here BY DESIGN. This is a static probe: one
    frozen frame, no history, no rollout. Feeding a longer horizon would mean
    the conditions differed in temporal context as well as in prompt, which is
    two changes and one measurement.
    """
    out = {"video": {}, "state": {}, "language": {}}
    for k, v in flat.items():
        if k.startswith("video."):
            arr = np.asarray(v)
            if arr.ndim == 3:                      # (H, W, C) -> (T, H, W, C)
                arr = arr[None]
            out["video"][k[len("video."):]] = arr[None]
        elif k.startswith("state."):
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 1:                      # (D,) -> (T, D)
                arr = arr[None]
            out["state"][k[len("state."):]] = arr[None]
    # The instruction is the intervention, so it is set here rather than copied
    # from the observation -- that is the whole point of the probe.
    out["language"][LANG_KEY] = [[str(prompt)]]
    return out


class GrootPolicy:
    """One query per condition against a running `run_gr00t_server.py`.

    STATELESS PER CALL, like openpi's server: one observation in, one action
    chunk out. `reset()` is kept as a no-op so the sweep's contamination guards
    apply uniformly to every policy it can drive.
    """

    def __init__(self, host="localhost", port=5555, timeout_ms=60000):
        from gr00t.policy.server_client import PolicyClient

        self.client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms)
        if not self.client.ping():
            raise RuntimeError(f"no GR00T policy server at {host}:{port}")
        self._order = None

    def reset(self):
        pass

    def act_obs(self, obs, prompt):
        """Query with `obs`, overriding only the instruction. Returns (T, D).

        The observation is copied before editing. Mutating the caller's dict
        would leak one condition's prompt into the next, and the conditions are
        deliberately evaluated against a single frozen frame.
        """
        o = nest_observation(obs, prompt)
        out = self.client.get_action(o)
        # BasePolicy.get_action returns (action, info) on some paths and a bare
        # action dict on others; accept both rather than depend on which.
        action = out[0] if isinstance(out, tuple) else out
        mat, order = flatten_action(action, self._order)
        self._order = order
        return mat

    # The sweep calls `act(images, prompt, state)`. GR00T's observation is not
    # decomposable into that signature, so hosts drive it through `act_obs` and
    # this raises rather than silently inventing an observation.
    def act(self, images, prompt, state):
        raise NotImplementedError(
            "GrootPolicy takes a full observation dict; call act_obs(obs, "
            "prompt). The RoboCasa host builds it from the gym wrapper.")
