"""Config helpers for Isaac Lab configclass/YAML overlays."""

from __future__ import annotations

import copy
from collections.abc import Mapping


def _apply_none_default_overrides(obj, data: dict) -> None:
    """Apply overrides for config fields whose current value is ``None``.

    Isaac Lab 5.1's configclass runtime checker compares against the current
    value's concrete type. For fields annotated as optional but defaulted to
    ``None``, valid non-None YAML values fail type checking before annotations
    are considered. Apply those values directly, remove them from the overlay,
    then let ``from_dict`` handle the rest normally.
    """

    for key in list(data.keys()):
        if not hasattr(obj, key):
            continue
        value = data[key]
        current = getattr(obj, key)
        if isinstance(value, Mapping) and current is not None:
            nested = dict(value)
            _apply_none_default_overrides(current, nested)
            if nested:
                data[key] = nested
            else:
                data.pop(key)
            continue
        if current is None and value is not None:
            setattr(obj, key, value)
            data.pop(key)


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

    _apply_none_default_overrides(env_cfg, data)
    env_cfg.from_dict(data)

    if render_data is None:
        return
    if not isinstance(render_data, Mapping):
        raise TypeError(f"sim.render overlay must be a mapping, got {type(render_data)!r}")

    for key, value in render_data.items():
        if not hasattr(env_cfg.sim.render, key):
            raise KeyError(f"Unknown sim.render config field: {key}")
        setattr(env_cfg.sim.render, key, value)
