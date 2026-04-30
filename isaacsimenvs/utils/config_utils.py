"""Config helpers for Isaac Lab configclass/YAML overlays."""

from __future__ import annotations

import copy
from collections.abc import Mapping


def apply_env_cfg_dict(env_cfg, overlay: Mapping) -> None:
    """Apply an env dictionary onto an Isaac Lab configclass.

    Isaac Lab 5.1's ``configclass.from_dict`` checks scalar values against the
    runtime type of the default value. ``RenderCfg`` fields default to ``None``
    but are annotated as optional literals, so setting values such as
    ``rendering_mode: performance`` through ``from_dict`` raises a false type
    error. Everything except ``sim.render`` is still handled by ``from_dict``.
    """

    data = copy.deepcopy(dict(overlay))
    sim_data = data.get("sim")
    render_data = None
    if isinstance(sim_data, dict):
        render_data = sim_data.pop("render", None)

    env_cfg.from_dict(data)

    if render_data is None:
        return
    if not isinstance(render_data, Mapping):
        raise TypeError(f"sim.render overlay must be a mapping, got {type(render_data)!r}")

    for key, value in render_data.items():
        if not hasattr(env_cfg.sim.render, key):
            raise KeyError(f"Unknown sim.render config field: {key}")
        setattr(env_cfg.sim.render, key, value)
