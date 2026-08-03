"""Load upstream ReKep config and deep-merge our overrides onto it.

Keeps `third_party/ReKep/configs/config.yaml` pristine: our changes live in
`configs/rekep_libero.yaml` and are applied at load time. The workspace bounds
are fanned out to every consumer, because upstream relies on a YAML anchor
(`&bounds_min`) that a merge cannot reach through.
"""

import os

import yaml

from . import REPO_ROOT, REKEP_DIR

UPSTREAM_CONFIG = os.path.join(REKEP_DIR, "configs", "config.yaml")
OVERRIDE_CONFIG = os.path.join(REPO_ROOT, "configs", "rekep_libero.yaml")

# sections that carry their own copy of the workspace bounds upstream
_BOUNDS_CONSUMERS = ("main", "env", "path_solver", "subgoal_solver", "keypoint_proposer", "visualizer")


def deep_merge(base, override):
    """Recursively merge `override` into `base`, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(override_path=OVERRIDE_CONFIG):
    with open(UPSTREAM_CONFIG) as f:
        config = yaml.safe_load(f)
    with open(override_path) as f:
        overrides = yaml.safe_load(f) or {}

    workspace = overrides.pop("workspace", None)
    config = deep_merge(config, overrides)

    if workspace is not None:
        # Upstream defines bounds once and shares them via a YAML anchor, which
        # resolves at parse time — so overriding one section would silently
        # leave the others on OmniGibson's numbers. Set all of them explicitly.
        for section in _BOUNDS_CONSUMERS:
            if section in config:
                config[section]["bounds_min"] = list(workspace["bounds_min"])
                config[section]["bounds_max"] = list(workspace["bounds_max"])
        config["workspace"] = workspace

    return config
