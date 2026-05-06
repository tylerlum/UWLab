"""FurnitureBench square-leg SimToolReal finetuning task registration."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .furniturebench_leg_env import FurnitureBenchLegEnv
from .furniturebench_leg_env_cfg import FurnitureBenchLegEnvCfg


__all__ = ["FurnitureBenchLegEnv", "FurnitureBenchLegEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-FurnitureBenchLeg-Direct-v0",
    entry_point="isaacsimenvs.tasks.furniturebench_leg.furniturebench_leg_env:FurnitureBenchLegEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaacsimenvs.tasks.furniturebench_leg.furniturebench_leg_env_cfg:"
            "FurnitureBenchLegEnvCfg"
        ),
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "FurnitureBenchLeg.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "SimToolRealSAPG.yaml"),
    },
)
