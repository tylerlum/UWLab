"""Cartpole task registration.

Importing this subpackage fires `gym.register`, exposing ``Isaacsimenvs-Cartpole-Direct-v0``
to the gymnasium registry. The ``Isaacsimenvs-`` prefix (vs. Isaac Lab's ``Isaac-``)
avoids collision with ``isaaclab_tasks.direct.cartpole``, which registers
``Isaac-Cartpole-Direct-v0`` on its own import and would otherwise silently
override our entry points, steering ``load_cfg_from_registry`` at Isaac Lab's
stock YAML instead of ours.

Entry points:
- ``env_cfg_entry_point``     → CartpoleEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``→ cfg/task/Cartpole.yaml (overlay of defaults,
                                 read by our custom hydra decorator)
- ``rl_games_cfg_entry_point``→ cfg/train/CartpolePPO.yaml (rl_games algo cfg)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/CartpoleSAPG.yaml
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .cartpole_env import CartpoleEnv, CartpoleEnvCfg

__all__ = ["CartpoleEnv", "CartpoleEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-Cartpole-Direct-v0",
    entry_point="isaacsimenvs.tasks.cartpole.cartpole_env:CartpoleEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.cartpole.cartpole_env:CartpoleEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Cartpole.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "CartpolePPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "CartpoleSAPG.yaml"),
    },
)
