"""Config for the peg-in-hole SimToolReal variant."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import AssetsCfg, SimToolRealEnvCfg


VALID_GOAL_MODES = ("dense", "preInsertAndFinal", "finalGoalOnly")


@configclass
class PegInHoleAssetsCfg(AssetsCfg):
    object_name: str = "peg"
    peg_urdf: str = "assets/urdf/peg_in_hole/peg/peg.urdf"
    table_urdf: str = "assets/urdf/peg_in_hole/scenes/scene_0000/scene_tol00.urdf"
    peg_scale: tuple[float, float, float] = (6.25, 0.75, 0.5)
    num_assets_per_type: int = 1


@configclass
class PegInHoleCfg:
    scenes_path: str = "assets/urdf/peg_in_hole/scenes/scenes.npz"
    goal_mode: str = "dense"

    goal_xy_obs_noise: float = 0.002

    enable_retract: bool = True
    retract_reward_scale: float = 1.0
    retract_distance_threshold: float = 0.1
    retract_success_bonus: float = 1000.0
    retract_success_tolerance: float = 0.005

    force_scene_tol_combo: tuple[int, int] | None = None
    force_tightest_tol_per_scene: bool = False
    tightest_n_tol_slots_per_scene: int = -1
    force_peg_idx: int | None = None

    # Extra initial-object randomization for distillation experiments. "scene"
    # uses the scene file exactly; "yaw_only" keeps the T normal upright and
    # samples yaw around +Z; "full" samples a uniformly random orientation.
    object_init_position_noise_xy: tuple[float, float] = (0.0, 0.0)
    object_init_position_noise_z: float = 0.0
    object_init_orientation_mode: str = "scene"  # scene | yaw_only | full
    object_init_yaw_range_degrees: float = 180.0


@configclass
class PegInHoleEnvCfg(SimToolRealEnvCfg):
    assets: PegInHoleAssetsCfg = PegInHoleAssetsCfg()
    peg_in_hole: PegInHoleCfg = PegInHoleCfg()


__all__ = [
    "PegInHoleEnvCfg",
    "PegInHoleAssetsCfg",
    "PegInHoleCfg",
    "VALID_GOAL_MODES",
]
