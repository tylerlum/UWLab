"""Peg-in-hole scene setup."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import (
    _bake_usd,
    _convert_urdf_to_usd,
    _log_scene_step,
    _materialize_env_prims,
    _robot_joint_drive_cfg,
    build_rigid_object_cfg,
    build_robot_articulation_usd_cfg,
    hide_goal_viz_for_student_camera,
    setup_student_camera,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _asset_path(path: str) -> str:
    asset_path = Path(path)
    if not asset_path.is_absolute():
        asset_path = REPO_ROOT / asset_path
    return str(asset_path)


def _scene_key(urdf_path: str) -> str:
    path = Path(urdf_path)
    return f"{path.parent.name}_{path.stem}"


def setup_scene(env) -> None:
    """Build robot, peg, per-env hole scenes, goal marker, ground, and light."""
    assets_cfg = env.cfg.assets
    setup_t0 = time.perf_counter()
    _log_scene_step(setup_t0, f"peg-in-hole setup start num_envs={env.num_envs}")

    env._tmp_asset_dir = tempfile.mkdtemp(prefix="peg_in_hole_assets_")
    env._object_urdf_paths = [_asset_path(assets_cfg.peg_urdf)]
    env._table_urdf_paths = [_asset_path(path) for path in env._pih_scene_urdfs]

    usd_work_dir = Path(env._tmp_asset_dir) / "usd"
    bake_root = Path(env._tmp_asset_dir) / "baked_usd"
    usd_work_dir.mkdir(parents=True, exist_ok=True)

    peg_raw_usd = _convert_urdf_to_usd(
        _asset_path(assets_cfg.peg_urdf), usd_work_dir, fix_base=False
    )
    object_usd_path = _bake_usd(
        peg_raw_usd,
        bake_root,
        "object",
        props=dict(
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=1000.0,
            articulation_enabled=False,
        ),
    )
    goalviz_usd_path = _bake_usd(
        peg_raw_usd,
        bake_root,
        "goalviz",
        props=dict(
            kinematic_enabled=True,
            disable_gravity=True,
            articulation_enabled=False,
        ),
        collision_enabled=False,
    )

    robot_usd_path = _bake_usd(
        _convert_urdf_to_usd(
            _asset_path(assets_cfg.robot_urdf),
            usd_work_dir,
            fix_base=True,
            self_collision=False,
            joint_drive=_robot_joint_drive_cfg(),
        ),
        bake_root,
        "robot",
        props=dict(
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
            enabled_self_collisions=False,
            solver_position_iterations=8,
            solver_velocity_iterations=0,
        ),
        apply_physx_articulation=True,
    )

    table_usd_by_urdf = {}
    for urdf in sorted(set(env._pih_scene_urdfs)):
        key = _scene_key(urdf)
        raw_usd = _convert_urdf_to_usd(
            _asset_path(urdf), usd_work_dir / "tables" / key, fix_base=False
        )
        table_usd_by_urdf[urdf] = _bake_usd(
            raw_usd,
            bake_root,
            f"table/{key}",
            props=dict(
                kinematic_enabled=True,
                disable_gravity=True,
                articulation_enabled=False,
            ),
        )
    table_usd_paths = [table_usd_by_urdf[urdf] for urdf in env._pih_scene_urdfs]
    _log_scene_step(setup_t0, f"converted {len(table_usd_by_urdf)} scene URDFs")

    _materialize_env_prims(env)

    env.robot = Articulation(build_robot_articulation_usd_cfg(robot_usd_path))
    env.table = RigidObject(
        build_rigid_object_cfg("/World/envs/env_.*/Table", table_usd_paths)
    )
    env.object = RigidObject(
        build_rigid_object_cfg("/World/envs/env_.*/Object", [object_usd_path])
    )
    env.goal_viz = RigidObject(
        build_rigid_object_cfg("/World/envs/env_.*/GoalViz", [goalviz_usd_path])
    )
    _log_scene_step(setup_t0, "spawned robot/table/object/goalviz")

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    env._object_scale_per_env = torch.tensor(
        assets_cfg.peg_scale,
        device=env.device,
        dtype=torch.float32,
    ).expand(env.num_envs, -1).contiguous()
    env._object_asset_index_per_env = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.long
    )

    env.scene.articulations["robot"] = env.robot
    env.scene.rigid_objects["table"] = env.table
    env.scene.rigid_objects["object"] = env.object
    env.scene.rigid_objects["goal_viz"] = env.goal_viz
    hide_goal_viz_for_student_camera(env)
    setup_student_camera(env)
    _log_scene_step(setup_t0, "registered assets with scene")


__all__ = ["setup_scene"]
