"""Headless probes for FurnitureBench-leg reward, success, and reset accounting.

Examples:
    env_uwlab/bin/python isaacsimenvs/debug_furniturebench_leg_rewards.py \
        --headless --drive teleport_goals --goal_mode preInsertAndFinal

    env_uwlab/bin/python isaacsimenvs/debug_furniturebench_leg_rewards.py \
        --headless --drive policy --goal_mode finalGoalOnly \
        --success_mode simtoolreal_keypoints \
        --fixture_xy_offset -0.25 0.0 \
        --random_start_xy_center 0.20 0.08
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RL_GAMES = REPO_ROOT / "rl_games"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if LOCAL_RL_GAMES.exists():
    sys.path.insert(0, str(LOCAL_RL_GAMES))

POLICY_DIR = REPO_ROOT / ".pretrained_checkpoints" / "SimToolReal" / "pretrained_policy"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument(
        "--drive",
        choices=("teleport_goals", "policy", "zero"),
        default="teleport_goals",
    )
    parser.add_argument(
        "--goal_mode",
        choices=("finalGoalOnly", "preInsertAndFinal", "dense"),
        default="preInsertAndFinal",
    )
    parser.add_argument(
        "--initialization_mode",
        choices=("random_table", "upright_fixed", "omnireset_partial_assemblies"),
        default="upright_fixed",
    )
    parser.add_argument(
        "--success_mode",
        choices=("omnireset_alignment", "simtoolreal_keypoints"),
        default="omnireset_alignment",
    )
    parser.add_argument("--physics_profile", choices=("omnireset", "simtoolreal"), default="omnireset")
    parser.add_argument("--success_steps", type=int, default=10)
    parser.add_argument("--episode_length", type=int, default=600)
    parser.add_argument("--policy_steps", type=int, default=600)
    parser.add_argument("--print_interval", type=int, default=60)
    parser.add_argument("--checkpoint", default=str(POLICY_DIR / "model.pth"))
    parser.add_argument("--config", default=str(POLICY_DIR / "config.yaml"))
    parser.add_argument("--fixture_xy_offset", nargs=2, type=float, default=None)
    parser.add_argument("--goal_xy_offset", nargs=2, type=float, default=None)
    parser.add_argument("--random_start_xy_center", nargs=2, type=float, default=None)
    parser.add_argument("--random_start_xy_range", nargs=2, type=float, default=None)
    parser.add_argument("--random_start_roll_pitch_range_deg", nargs=2, type=float, default=None)
    parser.add_argument("--random_start_yaw_range_deg", type=float, default=None)
    parser.add_argument("--disable_randomness", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write_object_every_step", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero_object_velocity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--steps_per_goal",
        type=int,
        default=0,
        help="Defaults to success_steps + 3 for teleport_goals.",
    )
    return parser


def _launch_app():
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    return app, args


_app, _args = _launch_app()


def _disable_randomness(cfg) -> None:
    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)

    reset = cfg.reset
    reset.reset_position_noise_x = 0.0
    reset.reset_position_noise_y = 0.0
    reset.reset_position_noise_z = 0.0
    reset.reset_dof_pos_random_interval_arm = 0.0
    reset.reset_dof_pos_random_interval_fingers = 0.0
    reset.reset_dof_vel_random_interval = 0.0
    reset.table_reset_z_range = 0.0


def _make_cfg(args):
    import yaml

    from isaacsimenvs.tasks.furniturebench_leg.furniturebench_leg_env_cfg import FurnitureBenchLegEnvCfg
    from isaacsimenvs.utils.config_utils import apply_env_cfg_dict

    cfg = FurnitureBenchLegEnvCfg()
    yaml_path = REPO_ROOT / "isaacsimenvs" / "cfg" / "task" / "FurnitureBenchLeg.yaml"
    with yaml_path.open() as f:
        apply_env_cfg_dict(cfg, yaml.safe_load(f) or {})

    cfg.scene.num_envs = int(args.num_envs)
    cfg.sim.device = str(args.sim_device)
    cfg.furniturebench_leg.goal_mode = str(args.goal_mode)
    cfg.furniturebench_leg.initialization_mode = str(args.initialization_mode)
    cfg.furniturebench_leg.success_mode = str(args.success_mode)
    cfg.furniturebench_leg.physics_profile = str(args.physics_profile)
    cfg.termination.success_steps = int(args.success_steps)
    cfg.termination.episode_length = int(args.episode_length)
    cfg.episode_length_s = float(args.episode_length) / 60.0

    if args.fixture_xy_offset is not None:
        cfg.furniturebench_leg.fixture_xy_offset = tuple(float(x) for x in args.fixture_xy_offset)
    if args.goal_xy_offset is not None:
        cfg.furniturebench_leg.goal_xy_offset = tuple(float(x) for x in args.goal_xy_offset)
    if args.random_start_xy_center is not None:
        cfg.furniturebench_leg.random_start_xy_center = tuple(float(x) for x in args.random_start_xy_center)
    if args.random_start_xy_range is not None:
        cfg.furniturebench_leg.random_start_xy_range = tuple(float(x) for x in args.random_start_xy_range)
    if args.random_start_roll_pitch_range_deg is not None:
        cfg.furniturebench_leg.random_start_roll_pitch_range_deg = tuple(
            float(x) for x in args.random_start_roll_pitch_range_deg
        )
    if args.random_start_yaw_range_deg is not None:
        cfg.furniturebench_leg.random_start_yaw_range_deg = float(args.random_start_yaw_range_deg)
    if bool(args.disable_randomness):
        _disable_randomness(cfg)
    return cfg


def _mean_dict(values: dict) -> dict[str, float]:
    out = {}
    for key, value in values.items():
        if hasattr(value, "float"):
            out[key] = float(value.float().mean().detach().cpu().item())
        else:
            out[key] = float(value)
    return out


def _env_scalar(value, env_id: int = 0) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
        if value.ndim == 0:
            return float(value.cpu().item())
        return float(value.reshape(-1)[env_id].cpu().item())
    return float(value)


def _print_snapshot(label: str, inner, reward=None, terminated=None, truncated=None) -> None:
    active_goal_idx = (inner._successes % inner.env_max_goals).long()
    reward_terms = _mean_dict(getattr(inner, "_reward_terms", {}))
    done_reasons = _mean_dict(getattr(inner, "_termination_reasons", {}))
    fields = {
        "successes0": int(inner._successes[0].detach().cpu().item()),
        "prev_episode_successes0": int(inner._prev_episode_successes[0].detach().cpu().item()),
        "active_goal0": int(active_goal_idx[0].detach().cpu().item()),
        "near0": bool(inner._near_goal[0].detach().cpu().item()),
        "near_steps0": int(inner._near_goal_steps[0].detach().cpu().item()),
        "lifted_mean": float(inner._lifted_object.float().mean().detach().cpu().item()),
        "kp_dist0": _env_scalar(inner._keypoints_max_dist, 0),
    }
    if hasattr(inner, "_omnireset_pos_align_error"):
        fields["omni_pos0"] = _env_scalar(inner._omnireset_pos_align_error, 0)
        fields["omni_xyrot0"] = _env_scalar(inner._omnireset_xy_rot_align_error, 0)
        fields["omni_pos_aligned_mean"] = float(
            inner._omnireset_position_aligned.float().mean().detach().cpu().item()
        )
        fields["omni_ori_aligned_mean"] = float(
            inner._omnireset_orientation_aligned.float().mean().detach().cpu().item()
        )
    if reward is not None:
        fields["reward0"] = _env_scalar(reward, 0)
    if terminated is not None:
        fields["terminated_mean"] = float(terminated.float().mean().detach().cpu().item())
    if truncated is not None:
        fields["truncated_mean"] = float(truncated.float().mean().detach().cpu().item())

    compact_fields = " ".join(
        f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in fields.items()
    )
    print(f"[probe] {label} {compact_fields}", flush=True)
    if reward_terms:
        print(f"[probe] {label} reward_terms={reward_terms}", flush=True)
    if done_reasons:
        print(f"[probe] {label} done_reasons={done_reasons}", flush=True)


def _write_object_to_goal(inner, *, zero_velocity: bool) -> None:
    import torch

    env_ids = torch.arange(inner.num_envs, device=inner.device, dtype=torch.long)
    pose = torch.cat(
        [inner.goal_viz.data.root_pos_w, inner.goal_viz.data.root_quat_w],
        dim=-1,
    )
    inner.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    if zero_velocity:
        inner.object.write_root_velocity_to_sim(
            torch.zeros(inner.num_envs, 6, device=inner.device),
            env_ids=env_ids,
        )


def _run_teleport_goals(env, inner, args) -> None:
    import torch

    actions = torch.zeros((inner.num_envs, inner.cfg.action_space), device=inner.device)
    steps_per_goal = int(args.steps_per_goal) if int(args.steps_per_goal) > 0 else int(args.success_steps) + 3
    _print_snapshot("after_reset", inner)
    for goal_round in range(int(inner._leg_num_goals)):
        active_goal_before = int((inner._successes[0] % inner.env_max_goals[0]).detach().cpu().item())
        _write_object_to_goal(inner, zero_velocity=bool(args.zero_object_velocity))
        print(
            f"[probe] teleport goal_round={goal_round} active_goal_before={active_goal_before} "
            f"steps_per_goal={steps_per_goal}",
            flush=True,
        )
        prev_successes = int(inner._successes[0].detach().cpu().item())
        for step in range(steps_per_goal):
            if bool(args.write_object_every_step):
                _write_object_to_goal(inner, zero_velocity=bool(args.zero_object_velocity))
            _, reward, terminated, truncated, _ = env.step(actions)
            if step in {0, steps_per_goal - 1}:
                _print_snapshot(f"goal{goal_round}_step{step}", inner, reward, terminated, truncated)
            successes = int(inner._successes[0].detach().cpu().item())
            if successes > prev_successes:
                _print_snapshot(f"goal{goal_round}_success_step{step}", inner, reward, terminated, truncated)
                break
            if bool(terminated.any()) or bool(truncated.any()):
                _print_snapshot(f"goal{goal_round}_done_step{step}", inner, reward, terminated, truncated)
                break
        else:
            _print_snapshot(f"goal{goal_round}_missed", inner, reward, terminated, truncated)


def _run_policy_or_zero(env, inner, args) -> None:
    import torch

    player = None
    if args.drive == "policy":
        from deployment.rl_player import RlPlayer

        player = RlPlayer(
            num_observations=140,
            num_actions=inner.cfg.action_space,
            config_path=str(args.config),
            checkpoint_path=str(args.checkpoint),
            device=args.rl_device,
            num_envs=inner.num_envs,
        )
        player.player.init_rnn()

    obs = inner._get_observations()
    _print_snapshot("policy_start", inner)
    for step in range(int(args.policy_steps)):
        if player is None:
            action = torch.zeros((inner.num_envs, inner.cfg.action_space), device=inner.device)
        else:
            policy_obs = obs["policy"].to(args.rl_device)
            action = player.get_normalized_action(policy_obs, deterministic_actions=True).to(inner.device)
        obs, reward, terminated, truncated, _ = env.step(action)
        if step % int(args.print_interval) == 0 or bool(terminated.any()) or bool(truncated.any()):
            _print_snapshot(f"policy_step{step}", inner, reward, terminated, truncated)


def main() -> int:
    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401

    args = _args
    torch.manual_seed(0)
    cfg = _make_cfg(args)
    env = gym.make("Isaacsimenvs-FurnitureBenchLeg-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    env.reset()
    env.step(torch.zeros((inner.num_envs, inner.cfg.action_space), device=inner.device))

    print(
        "[probe] cfg "
        f"drive={args.drive} goal_mode={cfg.furniturebench_leg.goal_mode} "
        f"init={cfg.furniturebench_leg.initialization_mode} "
        f"success_mode={cfg.furniturebench_leg.success_mode} "
        f"fixture_xy_offset={cfg.furniturebench_leg.fixture_xy_offset} "
        f"goal_xy_offset={cfg.furniturebench_leg.goal_xy_offset} "
        f"random_start_xy_center={cfg.furniturebench_leg.random_start_xy_center} "
        f"force_lifted={cfg.furniturebench_leg.force_lifted_for_keypoint_reward}",
        flush=True,
    )
    print(
        f"[probe] leg_goals_asset={inner._leg_goals_asset_t.detach().cpu().tolist()}",
        flush=True,
    )

    if args.drive == "teleport_goals":
        _run_teleport_goals(env, inner, args)
    else:
        _run_policy_or_zero(env, inner, args)

    env.close()
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        del _app
        os._exit(code)
