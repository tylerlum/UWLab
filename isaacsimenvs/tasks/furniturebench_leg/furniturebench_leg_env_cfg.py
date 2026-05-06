"""Config for the FurnitureBench square-leg SimToolReal finetuning task."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg


VALID_GOAL_MODES = ("dense", "preInsertAndFinal", "finalGoalOnly")
VALID_INIT_MODES = ("random_table", "upright_fixed", "omnireset_partial_assemblies")


@configclass
class FurnitureBenchLegCfg:
    goal_mode: str = "finalGoalOnly"
    initialization_mode: str = "omnireset_partial_assemblies"
    physics_profile: str = "omnireset"

    hole_index: int = 0
    use_omnireset_final_height: bool = True
    fixture_clearance: float = 0.002
    hover_height: float = 0.12
    insert_height: float = 0.038
    preinsert_final_z_offset: float = 0.015427
    preinsert_yaw_offset_deg: float = 20.73494
    final_yaw_offset_deg: float = 0.0
    dense_descend_steps: int = 10
    dense_screw_turns: float = 1.0
    force_lifted_for_keypoint_reward: bool = True
    policy_frame_pos_offset_asset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    random_start_xy_center: tuple[float, float] = (0.10, 0.08)
    random_start_xy_range: tuple[float, float] = (0.025, 0.025)
    random_start_z_offset_from_hole: float = 0.18
    random_start_z_range: float = 0.01
    random_start_yaw_center_deg: float = 0.0
    random_start_yaw_range_deg: float = 25.0
    random_start_roll_pitch_range_deg: tuple[float, float] = (10.0, 10.0)

    partial_assemblies_path: str = (
        ".pretrained_checkpoints/SimToolReal/omnireset_datasets/"
        "Resets/SquareLeg__SquareTableTop/partial_assemblies.pt"
    )


@configclass
class FurnitureBenchLegEnvCfg(SimToolRealEnvCfg):
    furniturebench_leg: FurnitureBenchLegCfg = FurnitureBenchLegCfg()


__all__ = [
    "FurnitureBenchLegEnvCfg",
    "FurnitureBenchLegCfg",
    "VALID_GOAL_MODES",
    "VALID_INIT_MODES",
]
