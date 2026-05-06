"""FurnitureBench square-leg screw task using SimToolReal observations/actions."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    subtract_frame_transforms,
    wrap_to_pi,
)

from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
from isaacsimenvs.tasks.simtoolreal.utils.logging_utils import log_step_metrics
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import build_observations, compute_intermediate_values
from isaacsimenvs.tasks.simtoolreal.utils.reward_utils import compute_rewards, update_near_goal_steps
from isaacsimenvs.tasks.simtoolreal.utils.termination_utils import update_tolerance_curriculum

from .furniturebench_leg_env_cfg import (
    FurnitureBenchLegEnvCfg,
    VALID_GOAL_MODES,
    VALID_INIT_MODES,
    VALID_SUCCESS_MODES,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FURNITUREBENCH_REPO_ROOT = Path("/home/tylerlum/github_repos/furniture-bench")
FURNITUREBENCH_RAW_URDF_BASE = (
    "https://raw.githubusercontent.com/clvrai/furniture-bench/main/"
    "furniture_bench/assets/furniture/urdf/square_table"
)
OMNIRESET_DATASET_ROOT_URL = (
    "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Datasets/OmniReset"
)
OMNIRESET_DATASET_CACHE_ROOT = (
    REPO_ROOT / ".pretrained_checkpoints" / "SimToolReal" / "omnireset_datasets"
)
OMNIRESET_PAIR_DIR = "SquareLeg__SquareTableTop"

R_USD_POLICY_LEG_WXYZ = (0.5, 0.5, 0.5, -0.5)
LEG_OBJECT_SCALE = (1.56375, 0.75442625, 0.75442625)

TABLE_HOLES = (
    (0.05625, 0.05625, 0.0020),
    (-0.05625, 0.05625, 0.0020),
    (-0.05625, -0.05625, 0.0020),
    (0.05625, -0.05625, 0.0020),
)
TABLE_ASSEMBLED_OFFSET = (0.05625, 0.05625, -0.009435)
LEG_ASSEMBLED_OFFSET = (0.0, 0.0, -0.056658)
FIXTURE_LOCAL_BOTTOM_Z = -0.01558619
SIMTOOLREAL_TABLE_HALF_HEIGHT = 0.15


def _repo_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str(REPO_ROOT / candidate)


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _default_partial_assemblies_url() -> str:
    return (
        f"{OMNIRESET_DATASET_ROOT_URL}/Resets/"
        f"{OMNIRESET_PAIR_DIR}/partial_assemblies.pt"
    )


def _resolve_partial_assemblies_path(path_or_url: str) -> Path:
    if _is_url(path_or_url):
        parsed = urlparse(path_or_url)
        marker = "/Datasets/OmniReset/"
        if marker in parsed.path:
            rel_path = Path(parsed.path.split(marker, 1)[1])
        else:
            rel_path = Path(parsed.path.lstrip("/")).name
        target = OMNIRESET_DATASET_CACHE_ROOT / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"[FurnitureBenchLegEnv] downloading {path_or_url} -> {target}", flush=True)
            urlretrieve(path_or_url, target)
        return target

    path = Path(_repo_path(path_or_url))
    if path.exists():
        return path

    default_cache = OMNIRESET_DATASET_CACHE_ROOT / "Resets" / OMNIRESET_PAIR_DIR / "partial_assemblies.pt"
    if path == default_cache:
        return _resolve_partial_assemblies_path(_default_partial_assemblies_url())
    return path


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _quat_yaw(device, yaw_rad: float) -> torch.Tensor:
    zero = torch.zeros(1, device=device)
    yaw = torch.tensor([float(yaw_rad)], device=device)
    return quat_from_euler_xyz(zero, zero, yaw)[0]


def _quat_rpy(device, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    return quat_from_euler_xyz(roll, pitch, yaw)


def _goal_count_from_mode(goal_mode: str, dense_descend_steps: int) -> int:
    if goal_mode in ("finalGoalOnly", "highHover"):
        return 1
    if goal_mode in ("preInsertAndFinal", "highHoverAndFinal"):
        return 2
    if goal_mode == "preInsertDenseFinal":
        return 1 + max(1, int(dense_descend_steps))
    if goal_mode == "dense":
        return 2 + max(1, int(dense_descend_steps))
    raise ValueError(f"goal_mode must be one of {VALID_GOAL_MODES}, got {goal_mode!r}")


def _viewer_urdf_path(local_path: Path, raw_name: str) -> str:
    if local_path.exists():
        return str(local_path)
    return f"{FURNITUREBENCH_RAW_URDF_BASE}/{raw_name}"


class FurnitureBenchLegEnv(SimToolRealEnv):
    cfg: FurnitureBenchLegEnvCfg

    def __init__(self, cfg: FurnitureBenchLegEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        if str(cfg.furniturebench_leg.success_mode) not in VALID_SUCCESS_MODES:
            raise ValueError(
                f"success_mode must be one of {VALID_SUCCESS_MODES}, "
                f"got {cfg.furniturebench_leg.success_mode!r}"
            )
        self._configure_assets(cfg)
        if str(cfg.furniturebench_leg.physics_profile).lower() == "omnireset":
            self._apply_omnireset_physics_profile(cfg)
        cfg.termination.max_consecutive_successes = _goal_count_from_mode(
            cfg.furniturebench_leg.goal_mode,
            cfg.furniturebench_leg.dense_descend_steps,
        )
        super().__init__(cfg, render_mode, **kwargs)

        self._leg_fixture_root_local = self._compute_fixture_root_local()
        self._leg_goal_root_local = self._compute_goal_root_local()
        self._leg_goals_asset_t = self._build_goal_sequence_asset()
        self._leg_goal_uses_omnireset_success_t = self._build_goal_success_modes()
        self._leg_num_goals = int(self._leg_goals_asset_t.shape[0])
        cfg.termination.max_consecutive_successes = self._leg_num_goals

        self.env_max_goals = torch.full(
            (self.num_envs,), self._leg_num_goals, dtype=torch.long, device=self.device
        )
        self.prev_episode_env_max_goals = self.env_max_goals.clone()

        self._table_assembled_offset_t = torch.tensor(
            TABLE_ASSEMBLED_OFFSET, dtype=torch.float32, device=self.device
        )
        self._leg_assembled_offset_t = torch.tensor(
            LEG_ASSEMBLED_OFFSET, dtype=torch.float32, device=self.device
        )
        self.retract_phase = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.retract_succeeded = torch.zeros_like(self.retract_phase)
        self._just_entered_retract = torch.zeros_like(self.retract_phase)
        self._just_retracted = torch.zeros_like(self.retract_phase)
        self._omnireset_pos_align_error = torch.zeros(self.num_envs, device=self.device)
        self._omnireset_xy_rot_align_error = torch.zeros(self.num_envs, device=self.device)
        self._omnireset_position_aligned = torch.zeros_like(self.retract_phase)
        self._omnireset_orientation_aligned = torch.zeros_like(self.retract_phase)
        self._keypoint_near_goal_for_active_goal = torch.zeros_like(self.retract_phase)
        self._leg_prev_yaw = torch.zeros(self.num_envs, device=self.device)
        self._leg_unwrapped_yaw = torch.zeros(self.num_envs, device=self.device)
        self._leg_entered_hole = torch.zeros_like(self.retract_phase)
        self._leg_hole_entry_yaw = torch.zeros(self.num_envs, device=self.device)
        self._leg_hole_entry_z = torch.zeros(self.num_envs, device=self.device)
        self._leg_screw_cw_turns = torch.zeros(self.num_envs, device=self.device)
        self._leg_screw_max_cw_turns = torch.zeros(self.num_envs, device=self.device)
        self._leg_screw_depth = torch.zeros(self.num_envs, device=self.device)
        self._leg_screw_max_depth = torch.zeros(self.num_envs, device=self.device)
        self._leg_screw_radial_error = torch.zeros(self.num_envs, device=self.device)
        self._leg_screw_phase_error = torch.zeros(self.num_envs, device=self.device)
        self._leg_screw_pushthrough = torch.zeros_like(self.retract_phase)

        self._partial_pos_t: torch.Tensor | None = None
        self._partial_quat_t: torch.Tensor | None = None
        if str(cfg.furniturebench_leg.initialization_mode) == "omnireset_partial_assemblies":
            self._load_partial_assemblies()

        print(
            "[FurnitureBenchLegEnv] "
            f"goal_mode={cfg.furniturebench_leg.goal_mode} "
            f"init={cfg.furniturebench_leg.initialization_mode} "
            f"success_mode={cfg.furniturebench_leg.success_mode} "
            f"goals={self._leg_num_goals}",
            flush=True,
        )

    def _configure_assets(self, cfg: FurnitureBenchLegEnvCfg) -> None:
        assets = cfg.assets
        assets.object_name = "furniturebench_square_leg"
        assets.object_urdf_paths = ()
        assets.object_usd_paths = (
            ".pretrained_checkpoints/SimToolReal/omnireset_assets/"
            "FurnitureBench/SquareLeg/square_leg.usd",
        )
        assets.object_scales = (LEG_OBJECT_SCALE,)
        assets.fixture_usd_path = (
            ".pretrained_checkpoints/SimToolReal/omnireset_assets/"
            "FurnitureBench/SquareTableTop/square_table_top.usd"
        )
        assets.object_policy_frame_quat_wxyz = R_USD_POLICY_LEG_WXYZ
        assets.object_policy_frame_pos_offset = tuple(
            float(x) for x in cfg.furniturebench_leg.policy_frame_pos_offset_asset
        )
        assets.object_viewer_urdf_path = _viewer_urdf_path(
            FURNITUREBENCH_REPO_ROOT
            / "furniture_bench/assets/furniture/urdf/square_table/square_table_leg1.urdf",
            "square_table_leg1.urdf",
        )
        assets.fixture_viewer_urdf_path = _viewer_urdf_path(
            FURNITUREBENCH_REPO_ROOT
            / "furniture_bench/assets/furniture/urdf/square_table/square_table_top.urdf",
            "square_table_top.urdf",
        )
        fixture_root_z = (
            float(cfg.reset.table_reset_z)
            + SIMTOOLREAL_TABLE_HALF_HEIGHT
            - FIXTURE_LOCAL_BOTTOM_Z
            + float(cfg.furniturebench_leg.fixture_clearance)
        )
        fixture_xy = tuple(float(x) for x in cfg.furniturebench_leg.fixture_xy_offset)
        cfg.reset.fixed_fixture_pose = (
            fixture_xy[0],
            fixture_xy[1],
            fixture_root_z,
            1.0,
            0.0,
            0.0,
            0.0,
        )

    def _apply_omnireset_physics_profile(self, cfg: FurnitureBenchLegEnvCfg) -> None:
        physx = cfg.sim.physx
        physx.solver_type = 1
        physx.max_position_iteration_count = 192
        physx.max_velocity_iteration_count = 1
        physx.bounce_threshold_velocity = 0.02
        physx.friction_offset_threshold = 0.01
        physx.friction_correlation_distance = 0.0005
        physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        physx.gpu_total_aggregate_pairs_capacity = 2**23
        physx.gpu_max_rigid_contact_count = 2**23
        physx.gpu_max_rigid_patch_count = 2**23
        physx.gpu_collision_stack_size = 2**31

        assets = cfg.assets
        assets.object_mass = 0.05
        assets.fixture_mass = 0.5
        assets.object_solver_position_iteration_count = 4
        assets.object_solver_velocity_iteration_count = 0
        assets.fixture_solver_position_iteration_count = 4
        assets.fixture_solver_velocity_iteration_count = 0
        assets.object_friction = 1.5
        assets.object_dynamic_friction = 1.4
        assets.fixture_friction = 0.4
        assets.fixture_dynamic_friction = 0.325
        assets.table_friction = 0.45
        assets.table_dynamic_friction = 0.35

    def _compute_fixture_root_local(self) -> torch.Tensor:
        leg_cfg = self.cfg.furniturebench_leg
        fixture_xy = tuple(float(x) for x in leg_cfg.fixture_xy_offset)
        z = (
            float(self.cfg.reset.table_reset_z)
            + SIMTOOLREAL_TABLE_HALF_HEIGHT
            - FIXTURE_LOCAL_BOTTOM_Z
            + float(leg_cfg.fixture_clearance)
        )
        return torch.tensor([fixture_xy[0], fixture_xy[1], z], device=self.device, dtype=torch.float32)

    def _compute_goal_root_local(self) -> torch.Tensor:
        leg_cfg = self.cfg.furniturebench_leg
        goal_xy = tuple(float(x) for x in leg_cfg.goal_xy_offset)
        z = (
            float(self.cfg.reset.table_reset_z)
            + SIMTOOLREAL_TABLE_HALF_HEIGHT
            - FIXTURE_LOCAL_BOTTOM_Z
            + float(leg_cfg.fixture_clearance)
        )
        return torch.tensor([goal_xy[0], goal_xy[1], z], device=self.device, dtype=torch.float32)

    def _hole_pos_local(self) -> torch.Tensor:
        hole_index = int(self.cfg.furniturebench_leg.hole_index)
        if hole_index < 0 or hole_index >= len(TABLE_HOLES):
            raise ValueError(f"hole_index must be in [0, {len(TABLE_HOLES)}), got {hole_index}")
        return self._leg_goal_root_local + torch.tensor(TABLE_HOLES[hole_index], device=self.device)

    def _asset_pose(self, pos: torch.Tensor, yaw_rad: float) -> torch.Tensor:
        pos = pos.to(self.device, dtype=torch.float32).reshape(3)
        return torch.cat([pos, _quat_yaw(self.device, yaw_rad)], dim=0)

    def _screw_reference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        leg_cfg = self.cfg.furniturebench_leg
        hole = self._hole_pos_local()
        if bool(leg_cfg.use_omnireset_final_height):
            final_z = (
                self._leg_fixture_root_local[2]
                + float(TABLE_ASSEMBLED_OFFSET[2])
                - float(LEG_ASSEMBLED_OFFSET[2])
            )
        else:
            final_z = hole[2] + float(leg_cfg.insert_height)
        final_z = torch.as_tensor(final_z, device=self.device, dtype=torch.float32)
        pre_z = final_z + float(leg_cfg.preinsert_final_z_offset)
        pre_yaw = torch.deg2rad(
            torch.tensor(float(leg_cfg.preinsert_yaw_offset_deg), device=self.device, dtype=torch.float32)
        )
        final_yaw = torch.deg2rad(
            torch.tensor(float(leg_cfg.final_yaw_offset_deg), device=self.device, dtype=torch.float32)
        )
        total_yaw_delta = final_yaw - pre_yaw - 2.0 * torch.pi * float(leg_cfg.dense_screw_turns)
        return hole, pre_z, final_z, pre_yaw, total_yaw_delta

    def _build_goal_sequence_asset(self) -> torch.Tensor:
        leg_cfg = self.cfg.furniturebench_leg
        if leg_cfg.goal_mode not in VALID_GOAL_MODES:
            raise ValueError(f"goal_mode must be one of {VALID_GOAL_MODES}, got {leg_cfg.goal_mode!r}")

        hole, pre_z, final_z, pre_yaw_t, total_yaw_delta_t = self._screw_reference()
        pre_yaw = float(pre_yaw_t.detach().cpu().item())
        final_yaw = torch.deg2rad(torch.tensor(float(leg_cfg.final_yaw_offset_deg), device=self.device)).item()
        final_pos = torch.stack([hole[0], hole[1], final_z])
        pre_pos = torch.stack([hole[0], hole[1], pre_z])
        final = self._asset_pose(final_pos, final_yaw)
        pre = self._asset_pose(pre_pos, pre_yaw)
        hover = self._asset_pose(
            torch.stack([hole[0], hole[1], hole[2] + float(leg_cfg.hover_height)]),
            final_yaw,
        )

        if leg_cfg.goal_mode == "finalGoalOnly":
            goals = [final]
        elif leg_cfg.goal_mode == "highHover":
            goals = [hover]
        elif leg_cfg.goal_mode == "preInsertAndFinal":
            goals = [pre, final]
        elif leg_cfg.goal_mode == "highHoverAndFinal":
            goals = [hover, final]
        else:
            if leg_cfg.goal_mode == "preInsertDenseFinal":
                goals = [pre]
            else:
                goals = [hover, pre]
            steps = max(1, int(leg_cfg.dense_descend_steps))
            total_yaw_delta = float(total_yaw_delta_t.detach().cpu().item())
            for i in range(1, steps + 1):
                frac = float(i) / float(steps)
                z = pre_z * (1.0 - frac) + final_z * frac
                yaw = pre_yaw + float(total_yaw_delta) * frac
                goals.append(self._asset_pose(torch.stack([hole[0], hole[1], torch.as_tensor(z, device=self.device)]), yaw))
        return torch.stack(goals, dim=0).contiguous()

    def _build_goal_success_modes(self) -> torch.Tensor:
        """Return True for subgoals that should use OmniReset final alignment."""
        mode = str(self.cfg.furniturebench_leg.goal_mode)
        if mode == "finalGoalOnly":
            flags = [True]
        elif mode == "highHover":
            flags = [False]
        elif mode == "preInsertAndFinal":
            flags = [False, True]
        elif mode == "highHoverAndFinal":
            flags = [False, True]
        elif mode in ("dense", "preInsertDenseFinal"):
            flags = [False] * max(1, self._leg_goals_asset_t.shape[0])
            flags[-1] = True
        else:
            raise ValueError(f"goal_mode must be one of {VALID_GOAL_MODES}, got {mode!r}")
        return torch.tensor(flags, dtype=torch.bool, device=self.device)

    def _load_partial_assemblies(self) -> None:
        path = _resolve_partial_assemblies_path(self.cfg.furniturebench_leg.partial_assemblies_path)
        if not path.exists():
            raise FileNotFoundError(f"OmniReset partial assemblies file not found: {path}")
        data = _torch_load_cpu(path)
        self._partial_pos_t = data["relative_position"].to(self.device, dtype=torch.float32)
        self._partial_quat_t = data["relative_orientation"].to(self.device, dtype=torch.float32)
        if self._partial_pos_t.shape[0] != self._partial_quat_t.shape[0]:
            raise ValueError("partial_assemblies.pt relative_position/orientation length mismatch")
        print(f"[FurnitureBenchLegEnv] loaded {self._partial_pos_t.shape[0]} partial assembly starts", flush=True)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "prev_episode_env_max_goals"):
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]
        super()._reset_idx(env_ids)
        self._reset_leg_episode(env_ids)

    def _write_fixture_pose(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        pose = torch.zeros(n, 7, device=self.device)
        pose[:, 0:3] = self._leg_fixture_root_local.unsqueeze(0) + self.scene.env_origins[env_ids]
        pose[:, 3] = 1.0
        self.fixture.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.fixture.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)

    def _sample_start_pose_asset(self, env_ids: torch.Tensor) -> torch.Tensor:
        leg_cfg = self.cfg.furniturebench_leg
        n = env_ids.numel()
        mode = str(leg_cfg.initialization_mode)
        if mode not in VALID_INIT_MODES:
            raise ValueError(f"initialization_mode must be one of {VALID_INIT_MODES}, got {mode!r}")

        if mode == "omnireset_partial_assemblies":
            assert self._partial_pos_t is not None and self._partial_quat_t is not None
            indices = torch.randint(0, self._partial_pos_t.shape[0], (n,), device=self.device)
            pose = torch.zeros(n, 7, device=self.device)
            pose[:, 0:3] = self._leg_fixture_root_local.unsqueeze(0) + self._partial_pos_t[indices]
            pose[:, 3:7] = self._partial_quat_t[indices]
            return pose

        hole = self._hole_pos_local()
        pose = torch.zeros(n, 7, device=self.device)
        if mode == "upright_fixed":
            xy_center = torch.tensor(leg_cfg.random_start_xy_center, device=self.device)
            pose[:, 0:2] = xy_center.unsqueeze(0)
            pose[:, 2] = hole[2] + float(leg_cfg.random_start_z_offset_from_hole)
            pose[:, 3:7] = _quat_yaw(
                self.device,
                torch.deg2rad(torch.tensor(float(leg_cfg.final_yaw_offset_deg), device=self.device)).item(),
            )
            return pose

        xy_center = torch.tensor(leg_cfg.random_start_xy_center, device=self.device)
        xy_range = torch.tensor(leg_cfg.random_start_xy_range, device=self.device)
        noise = torch.empty(n, 3, device=self.device).uniform_(-1.0, 1.0)
        pose[:, 0:2] = xy_center.unsqueeze(0) + noise[:, 0:2] * xy_range.unsqueeze(0)
        pose[:, 2] = (
            hole[2]
            + float(leg_cfg.random_start_z_offset_from_hole)
            + noise[:, 2] * float(leg_cfg.random_start_z_range)
        )
        rp_range = torch.tensor(leg_cfg.random_start_roll_pitch_range_deg, device=self.device) * torch.pi / 180.0
        yaw_center = torch.deg2rad(torch.tensor(float(leg_cfg.random_start_yaw_center_deg), device=self.device))
        yaw_range = torch.deg2rad(torch.tensor(float(leg_cfg.random_start_yaw_range_deg), device=self.device))
        rpy_noise = torch.empty(n, 3, device=self.device).uniform_(-1.0, 1.0)
        roll = rpy_noise[:, 0] * rp_range[0]
        pitch = rpy_noise[:, 1] * rp_range[1]
        yaw = yaw_center + rpy_noise[:, 2] * yaw_range
        pose[:, 3:7] = _quat_rpy(self.device, roll, pitch, yaw)
        return pose

    def _reset_leg_episode(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        self.env_max_goals[env_ids] = self._leg_num_goals
        self._write_fixture_pose(env_ids)

        start_local = self._sample_start_pose_asset(env_ids)
        start_pose = start_local.clone()
        start_pose[:, 0:3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(start_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        self._object_init_z[env_ids] = start_local[:, 2]

        self._successes[env_ids] = 0
        self._lifted_object[env_ids] = bool(self.cfg.furniturebench_leg.force_lifted_for_keypoint_reward)
        self.retract_phase[env_ids] = False
        self.retract_succeeded[env_ids] = False
        self._just_entered_retract[env_ids] = False
        self._just_retracted[env_ids] = False
        self._reset_screw_metrics(env_ids, start_local)
        self._clear_goal_trackers(env_ids)
        self._write_goal_pose(env_ids)

    def _reset_screw_metrics(self, env_ids: torch.Tensor, start_local: torch.Tensor) -> None:
        _, _, yaw = euler_xyz_from_quat(start_local[:, 3:7])
        self._leg_prev_yaw[env_ids] = yaw
        self._leg_unwrapped_yaw[env_ids] = yaw
        self._leg_entered_hole[env_ids] = False
        self._leg_hole_entry_yaw[env_ids] = yaw
        self._leg_hole_entry_z[env_ids] = start_local[:, 2]
        self._leg_screw_cw_turns[env_ids] = 0.0
        self._leg_screw_max_cw_turns[env_ids] = 0.0
        self._leg_screw_depth[env_ids] = 0.0
        self._leg_screw_max_depth[env_ids] = 0.0
        self._leg_screw_radial_error[env_ids] = 0.0
        self._leg_screw_phase_error[env_ids] = torch.pi
        self._leg_screw_pushthrough[env_ids] = False

    def _update_screw_metrics(self) -> None:
        leg_cfg = self.cfg.furniturebench_leg
        env_origins = self.scene.env_origins
        object_pos = self.object.data.root_pos_w - env_origins
        _, _, yaw = euler_xyz_from_quat(self.object.data.root_quat_w)

        yaw_delta = wrap_to_pi(yaw - self._leg_prev_yaw)
        self._leg_unwrapped_yaw += yaw_delta
        self._leg_prev_yaw = yaw

        hole, pre_z, final_z, pre_yaw, total_yaw_delta = self._screw_reference()
        radial_error = torch.norm(object_pos[:, 0:2] - hole[0:2].unsqueeze(0), dim=-1)
        self._leg_screw_radial_error = radial_error

        entry = (
            (radial_error <= float(leg_cfg.screw_metric_hole_radius))
            & (object_pos[:, 2] <= pre_z + float(leg_cfg.screw_metric_entry_z_margin))
        )
        new_entry = entry & ~self._leg_entered_hole
        self._leg_entered_hole |= entry
        self._leg_hole_entry_yaw = torch.where(new_entry, self._leg_unwrapped_yaw, self._leg_hole_entry_yaw)
        self._leg_hole_entry_z = torch.where(new_entry, object_pos[:, 2], self._leg_hole_entry_z)

        cw_turns = (self._leg_hole_entry_yaw - self._leg_unwrapped_yaw) / (2.0 * torch.pi)
        depth = self._leg_hole_entry_z - object_pos[:, 2]
        cw_turns = torch.where(self._leg_entered_hole, torch.clamp(cw_turns, min=0.0), torch.zeros_like(cw_turns))
        depth = torch.where(self._leg_entered_hole, torch.clamp(depth, min=0.0), torch.zeros_like(depth))
        self._leg_screw_cw_turns = cw_turns
        self._leg_screw_depth = depth
        self._leg_screw_max_cw_turns = torch.maximum(self._leg_screw_max_cw_turns, cw_turns)
        self._leg_screw_max_depth = torch.maximum(self._leg_screw_max_depth, depth)

        denom = torch.clamp(pre_z - final_z, min=1.0e-6)
        frac = torch.clamp((pre_z - object_pos[:, 2]) / denom, 0.0, 1.0)
        expected_yaw = pre_yaw + total_yaw_delta * frac
        self._leg_screw_phase_error = torch.abs(wrap_to_pi(yaw - expected_yaw))

        deep = (
            self._leg_entered_hole
            & (radial_error <= float(leg_cfg.screw_metric_hole_radius))
            & (object_pos[:, 2] <= final_z + float(leg_cfg.screw_metric_final_depth_margin))
        )
        self._leg_screw_pushthrough |= (
            deep
            & (self._leg_screw_max_cw_turns < float(leg_cfg.screw_metric_min_turns_for_insert))
        )

    def _screw_insert_like(self) -> tuple[torch.Tensor, torch.Tensor]:
        leg_cfg = self.cfg.furniturebench_leg
        _, pre_z, final_z, _, _ = self._screw_reference()
        required_depth = torch.clamp(
            pre_z - final_z - float(leg_cfg.screw_metric_final_depth_margin),
            min=0.0,
        )
        insert_like = (
            self._leg_entered_hole
            & (self._leg_screw_max_cw_turns >= float(leg_cfg.screw_metric_min_turns_for_insert))
            & (self._leg_screw_max_depth >= required_depth)
            & ~self._leg_screw_pushthrough
        )
        return insert_like, required_depth

    def _clear_goal_trackers(self, env_ids: torch.Tensor) -> None:
        self._closest_keypoint_max_dist[env_ids] = -1.0
        self._closest_fingertip_dist[env_ids] = -1.0
        self._near_goal_steps[env_ids] = 0

    def _write_goal_pose(self, env_ids: torch.Tensor) -> None:
        subgoal_idx = (self._successes[env_ids] % self.env_max_goals[env_ids]).long()
        goal_local = self._leg_goals_asset_t[subgoal_idx]
        pose = goal_local.clone()
        pose[:, 0:3] += self.scene.env_origins[env_ids]
        self.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.goal_viz.write_root_velocity_to_sim(
            torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids
        )

    def _compute_omnireset_alignment_near_goal(self) -> torch.Tensor:
        """Apply OmniReset's assembled-frame final-goal check.

        OmniReset compares the insertive and receptive assembled frames, then
        ignores yaw.  This matters for threaded objects where final yaw is not
        uniquely observable from insertion depth.
        """

        leg_cfg = self.cfg.furniturebench_leg
        env_origins = self.scene.env_origins

        object_pos = self.object.data.root_pos_w - env_origins
        object_quat = self.object.data.root_quat_w
        fixture_pos = self.fixture.data.root_pos_w - env_origins
        fixture_quat = self.fixture.data.root_quat_w

        insertive_pos = object_pos + quat_apply(
            object_quat,
            self._leg_assembled_offset_t.unsqueeze(0).expand(self.num_envs, -1),
        )
        receptive_pos = fixture_pos + quat_apply(
            fixture_quat,
            self._table_assembled_offset_t.unsqueeze(0).expand(self.num_envs, -1),
        )
        insertive_quat = object_quat
        receptive_quat = fixture_quat

        rel_pos, rel_quat = subtract_frame_transforms(
            receptive_pos,
            receptive_quat,
            insertive_pos,
            insertive_quat,
        )
        e_x, e_y, _ = euler_xyz_from_quat(rel_quat)
        xy_rot_error = wrap_to_pi(e_x).abs() + wrap_to_pi(e_y).abs()
        pos_error = torch.norm(rel_pos, dim=-1)

        self._omnireset_pos_align_error = pos_error
        self._omnireset_xy_rot_align_error = xy_rot_error
        pos_threshold = float(leg_cfg.omnireset_position_success_threshold)
        ori_threshold = float(leg_cfg.omnireset_orientation_success_threshold)
        self._omnireset_position_aligned = pos_error < pos_threshold
        self._omnireset_orientation_aligned = xy_rot_error < ori_threshold

        return self._omnireset_position_aligned & self._omnireset_orientation_aligned

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        use_omnireset_success = str(self.cfg.furniturebench_leg.success_mode) == "omnireset_alignment"
        near_goal_steps_before = self._near_goal_steps.clone() if use_omnireset_success else None
        compute_intermediate_values(self)
        keypoint_near_goal = self._near_goal.clone()
        self._keypoint_near_goal_for_active_goal = keypoint_near_goal
        omnireset_near_goal = self._compute_omnireset_alignment_near_goal()

        if use_omnireset_success:
            active_goal_idx = (self._successes % self.env_max_goals).long()
            uses_omnireset_success = self._leg_goal_uses_omnireset_success_t[active_goal_idx]
            self._near_goal = torch.where(
                uses_omnireset_success, omnireset_near_goal, keypoint_near_goal
            )
            self._near_goal_steps = update_near_goal_steps(
                near_goal=self._near_goal,
                near_goal_steps=near_goal_steps_before,
                force_consecutive=self.cfg.termination.force_consecutive_near_goal_steps,
            )
            self._is_success = self._near_goal_steps >= self.cfg.termination.success_steps
            is_success = self._is_success
        else:
            is_success = self._is_success
        if self.cfg.furniturebench_leg.enable_retract:
            is_success = is_success & ~self.retract_phase
            self._is_success = is_success

        self._successes += is_success.long()
        self._successes.clamp_(max=self._leg_num_goals)
        success_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() > 0:
            self.episode_length_buf[success_ids] = 0

        self._just_entered_retract[:] = False
        self._just_retracted[:] = False
        if self.cfg.furniturebench_leg.enable_retract:
            self._just_entered_retract = (
                (self._successes >= self.env_max_goals) & ~self.retract_phase
            )
            self.retract_phase |= self._just_entered_retract

            if str(self.cfg.furniturebench_leg.success_mode) == "omnireset_alignment":
                object_at_goal = self._omnireset_pos_align_error <= float(
                    self.cfg.furniturebench_leg.retract_success_tolerance
                )
            else:
                object_at_goal = (
                    self._keypoints_max_dist
                    <= self.cfg.furniturebench_leg.retract_success_tolerance
                    * self.cfg.reward.keypoint_scale
                )
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
            self._just_retracted = (
                (mean_fingertip_dist > self.cfg.furniturebench_leg.retract_distance_threshold)
                & self.retract_phase
                & ~self.retract_succeeded
                & object_at_goal
            )
            self.retract_succeeded |= self._just_retracted

        if success_ids.numel() > 0:
            if self.cfg.furniturebench_leg.enable_retract:
                next_goal = is_success & ~self.retract_phase
                next_goal_ids = next_goal.nonzero(as_tuple=False).squeeze(-1)
            else:
                next_goal_ids = success_ids[self._successes[success_ids] < self.env_max_goals[success_ids]]
            if next_goal_ids.numel() > 0:
                self._clear_goal_trackers(next_goal_ids)
                self._write_goal_pose(next_goal_ids)

        object_z_local = self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fall = object_z_local < 0.1
        if self.cfg.furniturebench_leg.enable_retract:
            max_successes = self.retract_succeeded
            hand_far = (
                self._curr_fingertip_distances.max(dim=-1).values > 1.5
            ) & ~self.retract_phase
        else:
            max_successes = self._successes >= self.env_max_goals
            hand_far = self._curr_fingertip_distances.max(dim=-1).values > 1.5
        terminated = fall | max_successes | hand_far
        truncated = self.episode_length_buf >= self.max_episode_length
        self._termination_reasons = {
            "fall": fall,
            "max_successes": max_successes,
            "hand_far": hand_far,
            "timeout": truncated,
        }
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)

        if self.cfg.furniturebench_leg.enable_retract:
            leg_cfg = self.cfg.furniturebench_leg
            if str(leg_cfg.success_mode) == "omnireset_alignment":
                object_at_goal = (
                    self._omnireset_pos_align_error <= float(leg_cfg.retract_success_tolerance)
                ).float()
            else:
                object_at_goal = (
                    self._keypoints_max_dist
                    <= leg_cfg.retract_success_tolerance * self.cfg.reward.keypoint_scale
                ).float()
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
            retract_rew = (
                mean_fingertip_dist * leg_cfg.retract_reward_scale * object_at_goal
                + self._just_retracted.float() * leg_cfg.retract_success_bonus
            ) * self.retract_phase.float()

            already_in_retract = self.retract_phase & ~self._just_entered_retract
            action_penalty = (
                self._reward_terms["kuka_actions_penalty"]
                + self._reward_terms["hand_actions_penalty"]
            )
            reward = torch.where(already_in_retract, action_penalty + retract_rew, reward)
            self._reward_terms["retract_rew"] = retract_rew
            self._reward_terms["total_reward"] = reward
            self.extras["retract_phase_ratio"] = self.retract_phase.float().mean()
            self.extras["retract_success_ratio"] = self.retract_succeeded.float().mean()

        log_step_metrics(self)
        self._log_leg_metrics()
        return reward

    def _log_leg_metrics(self) -> None:
        self._update_screw_metrics()
        screw_insert_like, screw_required_depth = self._screw_insert_like()
        success_ratio = self._successes.float() / self.env_max_goals.clamp_min(1).float()
        episode_final = self.extras.setdefault("episode_final", {})
        episode_final["success_ratio"] = success_ratio
        episode_final["all_goals_hit"] = (self._successes >= self.env_max_goals).float()
        episode_final["screw_cw_turns_after_entry"] = self._leg_screw_max_cw_turns
        episode_final["screw_depth_after_entry_m"] = self._leg_screw_max_depth
        episode_final["screw_pushthrough"] = self._leg_screw_pushthrough.float()
        episode_final["screw_insert_like"] = screw_insert_like.float()
        if self.cfg.furniturebench_leg.enable_retract:
            episode_final["retract_success"] = self.retract_succeeded.float()
        self.extras["omnireset_pos_align_error"] = self._omnireset_pos_align_error.mean()
        self.extras["omnireset_xy_rot_align_error"] = self._omnireset_xy_rot_align_error.mean()
        self.extras["omnireset_pos_align_error_min"] = self._omnireset_pos_align_error.min()
        self.extras["omnireset_pos_align_error_median"] = self._omnireset_pos_align_error.median()
        self.extras["omnireset_xy_rot_align_error_min"] = self._omnireset_xy_rot_align_error.min()
        self.extras["omnireset_xy_rot_align_error_median"] = self._omnireset_xy_rot_align_error.median()
        self.extras["omnireset_position_aligned_ratio"] = self._omnireset_position_aligned.float().mean()
        self.extras["omnireset_orientation_aligned_ratio"] = self._omnireset_orientation_aligned.float().mean()
        self.extras["near_goal_ratio"] = self._near_goal.float().mean()
        self.extras["near_goal_steps_max"] = self._near_goal_steps.max()
        active_goal_idx = (self._successes % self.env_max_goals).long()
        uses_omnireset_success = self._leg_goal_uses_omnireset_success_t[active_goal_idx]
        self.extras["active_goal_index_mean"] = active_goal_idx.float().mean()
        self.extras["active_goal_index_max"] = active_goal_idx.max()
        self.extras["active_final_goal_ratio"] = uses_omnireset_success.float().mean()
        self.extras["keypoint_near_goal_ratio"] = (
            self._keypoint_near_goal_for_active_goal.float().mean()
        )
        self.extras["keypoint_near_goal_nonfinal_ratio"] = (
            self._keypoint_near_goal_for_active_goal
            & (active_goal_idx < (self.env_max_goals - 1))
        ).float().mean()
        self.extras["lifted_object_ratio"] = self._lifted_object.float().mean()
        self.extras["keypoints_max_dist"] = self._keypoints_max_dist.mean()
        self.extras["keypoints_max_dist_min"] = self._keypoints_max_dist.min()
        self.extras["keypoints_max_dist_median"] = self._keypoints_max_dist.median()
        self.extras["screw_entry_ratio"] = self._leg_entered_hole.float().mean()
        self.extras["screw_cw_turns_after_entry"] = self._leg_screw_cw_turns.mean()
        self.extras["screw_cw_turns_after_entry_max"] = self._leg_screw_max_cw_turns.max()
        self.extras["screw_depth_after_entry_m"] = self._leg_screw_depth.mean()
        self.extras["screw_depth_after_entry_m_max"] = self._leg_screw_max_depth.max()
        self.extras["screw_radial_error_m"] = self._leg_screw_radial_error.mean()
        self.extras["screw_radial_error_m_min"] = self._leg_screw_radial_error.min()
        self.extras["screw_phase_error_rad"] = self._leg_screw_phase_error.mean()
        self.extras["screw_phase_error_rad_min"] = self._leg_screw_phase_error.min()
        self.extras["screw_pushthrough_ratio"] = self._leg_screw_pushthrough.float().mean()
        self.extras["screw_insert_like_ratio"] = screw_insert_like.float().mean()
        self.extras["screw_required_depth_m"] = screw_required_depth
        prev_ratio = (
            self._prev_episode_successes.float()
            / self.prev_episode_env_max_goals.clamp_min(1).float()
        )
        self.extras["success_ratio"] = prev_ratio.mean()
        self.extras["all_goals_hit_ratio"] = (
            self._prev_episode_successes >= self.prev_episode_env_max_goals
        ).float().mean()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)


__all__ = ["FurnitureBenchLegEnv", "FurnitureBenchLegEnvCfg"]
