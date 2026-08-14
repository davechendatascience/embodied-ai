"""Did the prompt change the policy's behaviour, or is this the sampler?

numpy only. The metrics for a condition sweep, kept apart from the sweep itself
so they can be tested on synthetic actions with no simulator and no GPU.

THE MEASUREMENT THIS FILE EXISTS TO PREVENT.
"Condition B differs from condition A by cos 0.97" is not a finding until you
know what condition A differs from ITSELF by. A diffusion action head with
sampling on, a non-deterministic GPU reduction, or a server that batches
differently under load will all produce a spread with no input change at all.
So every sweep re-runs the baseline `repeats` times against byte-identical
input, and the NOISE FLOOR is the spread of that. A condition counts only if it
lands outside it. This is the same discipline as measuring what an unchanged
gripper does before calling a difference a defect.

WHY THE HEADLINE METRIC IS THE CHUNK, NOT THE FIRST ACTION.
These policies emit an action CHUNK and replay it open loop. The first delta of
a chunk is small and mostly says "start moving"; where the policy has decided to
go is in the sum. Two prompts that send the arm to two different objects can
agree closely on step 0 and disagree completely by step 7, so a first-action
metric is the least sensitive summary available.

  `chunk_dir` = sum of the translation channel over the chunk.

That sum is a DIRECTION OF INTENT, and nothing else. It is NOT a pose. An OSC
controller realises roughly a quarter of each commanded delta, so integrating
commands yields a point the arm never occupies -- fine as a summary of where a
chunk is aimed, wrong the moment it is treated as a position.

CHANNELS ARE NOT COMPARABLE AND ARE NOT MIXED. Translation and rotation live in
different units; a cosine over the concatenation is dominated by whichever the
normalisation happened to make larger. The gripper channel is a discrete
decision -- a cosine over it is meaningless -- so it is reported as an
agreement rate instead.
"""

import numpy as np

#: Channel layout of the 7-D policy action.
WV, ROT, GRIP = slice(0, 3), slice(3, 6), 6


def cos(a, b, eps=1e-9):
    """Cosine between two vectors. 0.0 when either is degenerate.

    Returns 0 rather than nan for a zero vector: a policy commanding nothing is
    a real thing that happens at the start of an episode, and a nan there
    poisons every aggregate downstream.
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.clip(a.dot(b) / (na * nb), -1.0, 1.0))


def chunk_dir(chunk):
    """(T, 7) action chunk -> (3,) summed translation. See module docstring."""
    return np.asarray(chunk, float)[:, WV].sum(axis=0)


def compare(ref, alt):
    """Two (T, 7) chunks -> the dict of differences we actually report.

    `ref` is the ungrounded baseline, `alt` the condition under test.
    """
    ref = np.asarray(ref, float)
    alt = np.asarray(alt, float)
    if ref.shape != alt.shape:
        raise ValueError(f"chunk shape {ref.shape} vs {alt.shape}")
    dr, da = chunk_dir(ref), chunk_dir(alt)
    return {
        # Headline: do the two chunks aim the same way?
        "cos_dir": cos(dr, da),
        # How far apart the aim points end up, in the policy's own action units.
        # NOT millimetres -- converting would need the controller's scaling, and
        # a number labelled mm that is not mm is worse than an unlabelled one.
        "d_dir": float(np.linalg.norm(da - dr)),
        # Least sensitive metric, kept because it is the one most papers report.
        "cos_first": cos(ref[0, WV], alt[0, WV]),
        # Discrete channel: fraction of steps where the open/close call agrees.
        "grip_agree": float(np.mean(np.sign(ref[:, GRIP])
                                    == np.sign(alt[:, GRIP]))),
    }


def noise_floor(chunks):
    """Repeats of ONE condition -> the spread attributable to nothing.

    Each repeat is compared against the first, so the floor is measured the same
    way every condition is. Returns the WORST (lowest cosine, largest distance)
    observed, not the mean: a floor is a bound, and averaging it away is how a
    sampler artefact gets published as an effect.
    """
    chunks = [np.asarray(c, float) for c in chunks]
    if len(chunks) < 2:
        return {"cos_dir": 1.0, "d_dir": 0.0, "n": len(chunks),
                "deterministic": None}
    cs = [compare(chunks[0], c) for c in chunks[1:]]
    return {
        "cos_dir": min(c["cos_dir"] for c in cs),
        "d_dir": max(c["d_dir"] for c in cs),
        "n": len(chunks),
        "deterministic": all(c["cos_dir"] >= 1.0 - 1e-9 for c in cs),
    }


#: Angular bands for describing a difference that has cleared the noise floor.
#: EXISTENCE AND SIZE ARE DIFFERENT QUESTIONS and the second needs a scale.
#: These cuts are a reporting convention, not a measurement: 60 degrees is well
#: past "same intent, slightly perturbed", and 15 degrees is comfortably inside
#: it. Anything between is called what it is -- ambiguous -- rather than rounded
#: to whichever verdict the writer preferred.
REDIRECTED_DEG = 60.0
PERTURBED_DEG = 15.0


def angle_deg(cos_value):
    """Cosine -> angle in degrees. The interpretable form of `cos_dir`."""
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def verdict(effect, floor, margin=1.0):
    """Classify a comparison as inert / perturbed / ambiguous / redirected.

    `exceeds` answers "is this outside the noise?" and nothing more. A cosine of
    0.994 clears a floor of 0.9997 and is still a six degree change -- reporting
    that as "the policy followed the box to the other object" would be false,
    and it is exactly the reading the bare boolean invites. So the size of the
    change is classified separately from its detectability, and both are shown.
    """
    if not exceeds(effect, floor, margin):
        return "inert", angle_deg(effect["cos_dir"])
    deg = angle_deg(effect["cos_dir"])
    if deg >= REDIRECTED_DEG:
        return "redirected", deg
    if deg <= PERTURBED_DEG:
        return "perturbed", deg
    return "ambiguous", deg


def exceeds(effect, floor, margin=1.0):
    """Is this condition's difference bigger than doing nothing twice?

    `margin` scales the floor before comparing. 1.0 means "strictly outside
    anything the baseline did to itself"; raise it to demand headroom.

    Both tests must agree. A condition that changes the aim's DIRECTION but not
    its MAGNITUDE, or the reverse, is reported as not-exceeding rather than
    quietly counted on whichever metric happened to clear -- picking the metric
    after seeing the numbers is how a null result becomes a positive one.
    """
    if floor.get("deterministic"):
        # No spread at all: any difference is real, however small. Report it,
        # and let magnitude decide whether it MATTERS -- that is a separate
        # question from whether it EXISTS.
        return effect["d_dir"] > 0.0
    return (effect["cos_dir"] < floor["cos_dir"]
            and effect["d_dir"] > floor["d_dir"] * margin)
