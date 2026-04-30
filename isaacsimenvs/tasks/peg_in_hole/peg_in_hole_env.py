"""Peg-in-hole task built as a thin SimToolRealEnv variant."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, random_orientation

from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
from isaacsimenvs.tasks.simtoolreal.utils.logging_utils import log_step_metrics
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import (
    OBS_FIELD_SIZES,
    build_observations,
    compute_intermediate_values,
)
from isaacsimenvs.tasks.simtoolreal.utils.reward_utils import compute_rewards
from isaacsimenvs.tasks.simtoolreal.utils.termination_utils import (
    update_tolerance_curriculum,
)

from .peg_in_hole_env_cfg import PegInHoleEnvCfg, VALID_GOAL_MODES
from .scene_utils import REPO_ROOT, setup_scene


def _xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[:, 3:4], quat[:, 0:3]], dim=-1)


def _obs_field_slice(fields: tuple[str, ...], field: str) -> slice | None:
    offset = 0
    for name in fields:
        size = OBS_FIELD_SIZES[name]
        if name == field:
            return slice(offset, offset + size)
        offset += size
    return None


class PegInHoleEnv(SimToolRealEnv):
    cfg: PegInHoleEnvCfg

    def __init__(
        self, cfg: PegInHoleEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        self._load_scene_data(cfg)
        cfg.assets.object_name = "peg"
        cfg.assets.table_urdf = self._pih_scene_urdfs[0]
        cfg.termination.max_consecutive_successes = self._pih_max_traj_len

        super().__init__(cfg, render_mode, **kwargs)

        self._pih_start_poses_t = torch.as_tensor(
            self._pih_start_poses, dtype=torch.float32, device=self.device
        )
        self._pih_goals_t = torch.as_tensor(
            self._pih_goals, dtype=torch.float32, device=self.device
        )
        self._pih_traj_lengths_t = torch.as_tensor(
            self._pih_traj_lengths, dtype=torch.long, device=self.device
        )
        self._pih_env_scene_idx_t = torch.as_tensor(
            self._pih_env_scene_idx, dtype=torch.long, device=self.device
        )
        self._pih_env_tol_slot_t = torch.as_tensor(
            self._pih_env_tol_slot_idx, dtype=torch.long, device=self.device
        )

        self.env_peg_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.env_max_goals = self._pih_traj_lengths_t[
            self._pih_env_scene_idx_t, self.env_peg_idx
        ].clone()
        self.prev_episode_env_max_goals = self.env_max_goals.clone()

        self.retract_phase = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.retract_succeeded = torch.zeros_like(self.retract_phase)
        self._just_entered_retract = torch.zeros_like(self.retract_phase)
        self._just_retracted = torch.zeros_like(self.retract_phase)

        self.goal_pos_obs_noise = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self._goal_kp_obs_slice = _obs_field_slice(
            tuple(cfg.obs.obs_list), "keypoints_rel_goal"
        )

    def _load_scene_data(self, cfg: PegInHoleEnvCfg) -> None:
        pih_cfg = cfg.peg_in_hole
        if pih_cfg.goal_mode not in VALID_GOAL_MODES:
            raise ValueError(
                f"goal_mode must be one of {VALID_GOAL_MODES}, got {pih_cfg.goal_mode!r}"
            )

        scenes_path = Path(pih_cfg.scenes_path)
        if not scenes_path.is_absolute():
            scenes_path = REPO_ROOT / scenes_path
        if not scenes_path.exists():
            raise FileNotFoundError(f"Peg-in-hole scenes file not found: {scenes_path}")

        data = np.load(scenes_path)
        start_poses = data["start_poses"].astype(np.float32)
        goals = data["goals"].astype(np.float32)
        traj_lengths = data["traj_lengths"].astype(np.int64)
        tol_pool_m = data["tolerance_pool_m"].astype(np.float32)
        scene_tol_indices = data["scene_tolerance_indices"].astype(np.int64)

        num_scenes, num_pegs, _ = start_poses.shape
        _, _, max_traj_len, _ = goals.shape
        num_tol_slots = scene_tol_indices.shape[1]
        if traj_lengths.shape != (num_scenes, num_pegs):
            raise ValueError(
                f"traj_lengths shape {traj_lengths.shape} does not match "
                f"({num_scenes}, {num_pegs})"
            )

        if pih_cfg.goal_mode == "finalGoalOnly":
            scene_idx = np.arange(num_scenes)[:, None].repeat(num_pegs, axis=1)
            peg_idx = np.arange(num_pegs)[None, :].repeat(num_scenes, axis=0)
            final_idx = traj_lengths - 1
            goals = goals[scene_idx, peg_idx, final_idx][:, :, None, :]
            traj_lengths = np.ones_like(traj_lengths)
            max_traj_len = 1
        elif pih_cfg.goal_mode == "preInsertAndFinal":
            scene_idx = np.arange(num_scenes)[:, None].repeat(num_pegs, axis=1)
            peg_idx = np.arange(num_pegs)[None, :].repeat(num_scenes, axis=0)
            final_idx = traj_lengths - 1
            pre_idx = np.clip(traj_lengths - 2, a_min=0, a_max=None)
            goals = np.stack(
                [goals[scene_idx, peg_idx, pre_idx], goals[scene_idx, peg_idx, final_idx]],
                axis=2,
            )
            traj_lengths = np.clip(traj_lengths, a_min=None, a_max=2)
            max_traj_len = 2

        num_envs = int(cfg.scene.num_envs)
        combo_count = num_scenes * num_tol_slots
        force_combo = pih_cfg.force_scene_tol_combo
        if force_combo is not None:
            scene_id, tol_slot = int(force_combo[0]), int(force_combo[1])
            if not (0 <= scene_id < num_scenes and 0 <= tol_slot < num_tol_slots):
                raise ValueError(
                    f"force_scene_tol_combo=({scene_id}, {tol_slot}) out of range "
                    f"(num_scenes={num_scenes}, num_tol_slots={num_tol_slots})"
                )
            env_scene_idx = np.full(num_envs, scene_id, dtype=np.int64)
            env_tol_slot_idx = np.full(num_envs, tol_slot, dtype=np.int64)
        else:
            combo_ids = np.arange(num_envs) % combo_count
            env_scene_idx = (combo_ids // num_tol_slots).astype(np.int64)
            env_tol_slot_idx = (combo_ids % num_tol_slots).astype(np.int64)

        if pih_cfg.force_tightest_tol_per_scene:
            tightest_slot = np.argmin(tol_pool_m[scene_tol_indices], axis=1).astype(np.int64)
            env_tol_slot_idx = tightest_slot[env_scene_idx]

        tightest_n = int(pih_cfg.tightest_n_tol_slots_per_scene)
        if (
            tightest_n > 0
            and tightest_n < num_tol_slots
            and not pih_cfg.force_tightest_tol_per_scene
        ):
            slots = np.argsort(tol_pool_m[scene_tol_indices], axis=1)[:, :tightest_n]
            env_tol_slot_idx = slots[env_scene_idx, np.arange(num_envs) % tightest_n]

        self._pih_start_poses = start_poses
        self._pih_goals = goals
        self._pih_traj_lengths = traj_lengths
        self._pih_num_scenes = num_scenes
        self._pih_num_pegs = num_pegs
        self._pih_num_tol_slots = num_tol_slots
        self._pih_max_traj_len = max_traj_len
        self._pih_env_scene_idx = env_scene_idx
        self._pih_env_tol_slot_idx = env_tol_slot_idx

        table_pattern_len = 1 if force_combo is not None else min(num_envs, combo_count)
        self._pih_scene_urdfs = [
            f"assets/urdf/peg_in_hole/scenes/scene_{scene_id:04d}/scene_tol{tol_slot:02d}.urdf"
            for scene_id, tol_slot in zip(
                env_scene_idx[:table_pattern_len], env_tol_slot_idx[:table_pattern_len]
            )
        ]

        print(
            f"[PegInHoleEnv] loaded {num_scenes} scenes x {num_pegs} pegs x "
            f"{num_tol_slots} tol slots "
            f"(goal_mode={pih_cfg.goal_mode}, max_traj_len={max_traj_len})",
            flush=True,
        )

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "prev_episode_env_max_goals"):
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]
        super()._reset_idx(env_ids)
        self._reset_peg_episode(env_ids)

    def _reset_peg_episode(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        pih_cfg = self.cfg.peg_in_hole
        if pih_cfg.force_peg_idx is None:
            peg_idx = torch.randint(
                0, self._pih_num_pegs, (n,), dtype=torch.long, device=self.device
            )
        else:
            peg_idx = torch.full(
                (n,), int(pih_cfg.force_peg_idx), dtype=torch.long, device=self.device
            )
            if not (0 <= int(pih_cfg.force_peg_idx) < self._pih_num_pegs):
                raise ValueError(
                    f"force_peg_idx={pih_cfg.force_peg_idx} out of range "
                    f"[0, {self._pih_num_pegs})"
                )

        scene_idx = self._pih_env_scene_idx_t[env_ids]
        self.env_peg_idx[env_ids] = peg_idx
        self.env_max_goals[env_ids] = self._pih_traj_lengths_t[scene_idx, peg_idx]

        start = self._pih_start_poses_t[scene_idx, peg_idx]
        pos_local = start[:, 0:3].clone()
        pos_local[:, 2] = (
            self._table_z_per_env[env_ids] + self.cfg.reset.table_object_z_offset
        )
        pos_noise_xy = torch.as_tensor(
            pih_cfg.object_init_position_noise_xy,
            device=self.device,
            dtype=torch.float32,
        )
        if torch.any(pos_noise_xy > 0.0) or pih_cfg.object_init_position_noise_z > 0.0:
            pos_noise = torch.empty(n, 3, device=self.device).uniform_(-1.0, 1.0)
            pos_local[:, 0:2] += pos_noise[:, 0:2] * pos_noise_xy
            pos_local[:, 2] += pos_noise[:, 2] * float(pih_cfg.object_init_position_noise_z)

        quat = _xyzw_to_wxyz(start[:, 3:7])
        orientation_mode = str(pih_cfg.object_init_orientation_mode).lower()
        if orientation_mode == "scene":
            pass
        elif orientation_mode == "yaw_only":
            angle = torch.empty(n, device=self.device).uniform_(
                -float(pih_cfg.object_init_yaw_range_degrees),
                float(pih_cfg.object_init_yaw_range_degrees),
            ) * (torch.pi / 180.0)
            axis = torch.zeros(n, 3, device=self.device)
            axis[:, 2] = 1.0
            quat = quat_mul(quat_from_angle_axis(angle, axis), quat)
        elif orientation_mode == "full":
            quat = random_orientation(n, device=self.device)
        else:
            raise ValueError(
                "cfg.peg_in_hole.object_init_orientation_mode must be one of "
                f"('scene', 'yaw_only', 'full'), got {orientation_mode!r}."
            )
        pose = torch.cat([pos_local + self.scene.env_origins[env_ids], quat], dim=-1)
        self.object.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )
        self._object_init_z[env_ids] = pos_local[:, 2]

        noise = pih_cfg.goal_xy_obs_noise
        self.goal_pos_obs_noise[env_ids] = 0.0
        if noise > 0:
            self.goal_pos_obs_noise[env_ids, 0:2] = torch.empty(
                n, 2, device=self.device
            ).uniform_(-noise, noise)

        self.retract_phase[env_ids] = False
        self.retract_succeeded[env_ids] = False
        self._clear_goal_trackers(env_ids)
        self._write_goal_pose(env_ids)

    def _write_goal_pose(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        scene_idx = self._pih_env_scene_idx_t[env_ids]
        peg_idx = self.env_peg_idx[env_ids]
        subgoal_idx = (self._successes[env_ids] % self.env_max_goals[env_ids]).long()

        goal = self._pih_goals_t[scene_idx, peg_idx, subgoal_idx]
        pos_local = goal[:, 0:3].clone()
        pos_local[:, 2] += self._table_z_per_env[env_ids] - self.cfg.reset.table_reset_z
        quat = _xyzw_to_wxyz(goal[:, 3:7])
        pose = torch.cat([pos_local + self.scene.env_origins[env_ids], quat], dim=-1)

        self.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.goal_viz.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )

    def _clear_goal_trackers(self, env_ids: torch.Tensor) -> None:
        self._closest_keypoint_max_dist[env_ids] = -1.0
        self._closest_fingertip_dist[env_ids] = -1.0
        self._near_goal_steps[env_ids] = 0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)

        pih_cfg = self.cfg.peg_in_hole
        is_success = self._is_success.clone()
        if pih_cfg.enable_retract:
            is_success &= ~self.retract_phase
        self._is_success = is_success

        self._successes += is_success.long()
        self._successes.clamp_(max=self._pih_max_traj_len)
        success_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() > 0:
            self.episode_length_buf[success_ids] = 0

        self._just_entered_retract[:] = False
        self._just_retracted[:] = False
        if pih_cfg.enable_retract:
            self._just_entered_retract = (
                (self._successes >= self.env_max_goals) & ~self.retract_phase
            )
            self.retract_phase |= self._just_entered_retract

            object_at_goal = (
                self._keypoints_max_dist
                <= pih_cfg.retract_success_tolerance * self.cfg.reward.keypoint_scale
            )
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
            self._just_retracted = (
                (mean_fingertip_dist > pih_cfg.retract_distance_threshold)
                & self.retract_phase
                & ~self.retract_succeeded
                & object_at_goal
            )
            self.retract_succeeded |= self._just_retracted

        if success_ids.numel() > 0:
            if pih_cfg.enable_retract:
                next_goal = is_success & ~self.retract_phase
            else:
                next_goal = is_success & (self._successes < self.env_max_goals)
            next_goal_ids = next_goal.nonzero(as_tuple=False).squeeze(-1)
            if next_goal_ids.numel() > 0:
                self._clear_goal_trackers(next_goal_ids)
                self._write_goal_pose(next_goal_ids)

        object_z_local = self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fall = object_z_local < 0.1
        if pih_cfg.enable_retract:
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

        if self.cfg.peg_in_hole.enable_retract:
            pih_cfg = self.cfg.peg_in_hole
            object_at_goal = (
                self._keypoints_max_dist
                <= pih_cfg.retract_success_tolerance * self.cfg.reward.keypoint_scale
            ).float()
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
            retract_rew = (
                mean_fingertip_dist * pih_cfg.retract_reward_scale * object_at_goal
                + self._just_retracted.float() * pih_cfg.retract_success_bonus
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
            self.extras["retract_success_tolerance"] = float(
                pih_cfg.retract_success_tolerance
            )

        log_step_metrics(self)
        self._log_peg_metrics()
        return reward

    def _log_peg_metrics(self) -> None:
        success_ratio = self._successes.float() / self.env_max_goals.clamp_min(1).float()
        all_goals_hit = self._successes >= self.env_max_goals

        episode_final = self.extras.setdefault("episode_final", {})
        episode_final["success_ratio"] = success_ratio
        episode_final["all_goals_hit"] = all_goals_hit.float()
        if self.cfg.peg_in_hole.enable_retract:
            episode_final["retract_success"] = self.retract_succeeded.float()

        prev_ratio = (
            self._prev_episode_successes.float()
            / self.prev_episode_env_max_goals.clamp_min(1).float()
        )
        self.extras["success_ratio"] = prev_ratio.mean()
        self.extras["all_goals_hit_ratio"] = (
            self._prev_episode_successes >= self.prev_episode_env_max_goals
        ).float().mean()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = build_observations(self)
        if self._goal_kp_obs_slice is not None:
            policy = obs["policy"]
            policy[:, self._goal_kp_obs_slice].view(self.num_envs, -1, 3).sub_(
                self.goal_pos_obs_noise.unsqueeze(1)
            )
            clip = self.cfg.obs.clamp_abs_observations
            obs["policy"] = policy.clamp(-clip, clip)
        return obs


__all__ = ["PegInHoleEnv", "PegInHoleEnvCfg"]
