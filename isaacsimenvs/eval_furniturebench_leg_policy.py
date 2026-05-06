"""Evaluate a FurnitureBench leg finetuned SimToolReal policy.

This runs the actual ``Isaacsimenvs-FurnitureBenchLeg-Direct-v0`` task, not the
older standalone rollout shim.  It records one MP4 and one pose-only HTML file
per episode and writes a compact JSON summary with success/fail labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RL_GAMES = REPO_ROOT / "rl_games"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if LOCAL_RL_GAMES.exists():
    sys.path.insert(0, str(LOCAL_RL_GAMES))

DEFAULT_CONFIG = REPO_ROOT / ".pretrained_checkpoints" / "SimToolReal" / "pretrained_policy" / "config.yaml"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "eval_furniturebench_leg_policy"
FURNITUREBENCH_LEG_FIXED_SIZE_M = (0.06255, 0.03017705, 0.03017705)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--num_episodes", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--goal_mode", default="preInsertAndFinal")
    parser.add_argument("--initialization_mode", default="omnireset_partial_assemblies")
    parser.add_argument("--success_mode", default="omnireset_alignment")
    parser.add_argument("--force_lifted_for_keypoint_reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixture_xy_offset", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--goal_xy_offset", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--hover_height", type=float, default=0.4)
    parser.add_argument("--dense_descend_steps", type=int, default=10)
    parser.add_argument("--dense_screw_turns", type=float, default=1.0)
    parser.add_argument("--final_success_requires_screw_insert_like", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--success_tolerance", type=float, default=0.01)
    parser.add_argument("--target_success_tolerance", type=float, default=0.0025)
    parser.add_argument("--eval_success_tolerance", type=float, default=None)
    parser.add_argument("--omnireset_position_success_threshold", type=float, default=0.0025)
    parser.add_argument("--omnireset_orientation_success_threshold", type=float, default=0.025)
    parser.add_argument("--success_steps", type=int, default=10)
    parser.add_argument("--frame_every", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--record_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture_html", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue_after_success_steps", type=int, default=45)
    parser.add_argument(
        "--stop_on_env_done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop each rollout when the env reports done; disable to record a fixed step window across auto-resets.",
    )
    return parser


def _launch_app():
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.record_video:
        args.enable_cameras = True
    app = AppLauncher(args).app
    return app, args


_app, _args = _launch_app()


def _make_cfg(args):
    from isaacsimenvs.tasks.furniturebench_leg.furniturebench_leg_env_cfg import FurnitureBenchLegEnvCfg

    cfg = FurnitureBenchLegEnvCfg()
    cfg.sim.device = args.sim_device
    cfg.scene.num_envs = 1
    cfg.episode_length_s = max(10.0, float(args.max_steps) / 60.0)

    leg = cfg.furniturebench_leg
    leg.goal_mode = str(args.goal_mode)
    leg.initialization_mode = str(args.initialization_mode)
    leg.success_mode = str(args.success_mode)
    leg.physics_profile = "omnireset"
    leg.hover_height = float(args.hover_height)
    leg.dense_descend_steps = int(args.dense_descend_steps)
    leg.dense_screw_turns = float(args.dense_screw_turns)
    leg.final_success_requires_screw_insert_like = bool(args.final_success_requires_screw_insert_like)
    leg.omnireset_position_success_threshold = float(args.omnireset_position_success_threshold)
    leg.omnireset_orientation_success_threshold = float(args.omnireset_orientation_success_threshold)
    leg.force_lifted_for_keypoint_reward = bool(args.force_lifted_for_keypoint_reward)
    leg.fixture_xy_offset = tuple(float(x) for x in args.fixture_xy_offset)
    leg.goal_xy_offset = tuple(float(x) for x in args.goal_xy_offset)

    cfg.reward.fixed_size = FURNITUREBENCH_LEG_FIXED_SIZE_M
    cfg.reward.fixed_size_keypoint_reward = True

    cfg.reset.reset_position_noise_x = 0.0
    cfg.reset.reset_position_noise_y = 0.0
    cfg.reset.reset_position_noise_z = 0.0
    cfg.reset.table_reset_z_range = 0.0

    term = cfg.termination
    term.success_steps = int(args.success_steps)
    term.force_consecutive_near_goal_steps = True
    term.auto_goal_reset_on_success = False
    term.success_tolerance = float(args.success_tolerance)
    term.target_success_tolerance = float(args.target_success_tolerance)
    term.eval_success_tolerance = args.eval_success_tolerance

    # Match the cluster matrix: no random pushes/torques.  For video review,
    # default to clean observations/actions so failures are easier to interpret.
    dr = cfg.domain_randomization
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    if bool(args.clean_eval):
        dr.use_obs_delay = False
        dr.use_action_delay = False
        dr.use_object_state_delay_noise = False
        dr.joint_velocity_obs_noise_std = 0.0
        cfg.reset.reset_dof_pos_random_interval_arm = 0.0
        cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
        cfg.reset.reset_dof_vel_random_interval = 0.0

    return cfg


def _render_frame(env) -> np.ndarray | None:
    frame = env.render()
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def _termination_reasons(inner) -> dict[str, bool]:
    reasons = {}
    for name, value in getattr(inner, "_termination_reasons", {}).items():
        if hasattr(value, "detach"):
            reasons[name] = bool(value[0].detach().cpu().item())
        else:
            reasons[name] = bool(value)
    return reasons


def _root_pose_xyzw(inner, rigid_object) -> np.ndarray:
    pos = rigid_object.data.root_pos_w[0] - inner.scene.env_origins[0]
    quat_wxyz = rigid_object.data.root_quat_w[0]
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = pos.detach().cpu().numpy()
    q = quat_wxyz.detach().cpu().numpy()
    pose[3:7] = q[[1, 2, 3, 0]]
    return pose


def _scalar_from_tensor(value, default=0.0):
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value[0].detach().cpu().item() if value.ndim > 0 else value.detach().cpu().item()
    return float(value)


def main() -> int:
    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from deployment.rl_player import RlPlayer
    from isaacsimenvs.tasks.simtoolreal.pose_viewer import (
        build_pose_viewer_html,
        capture_pose_viewer_frame,
        fixture_urdf_text_for_env,
        object_urdf_text_for_env,
        table_urdf_text_for_env,
    )

    args = _args
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _make_cfg(args)
    env = gym.make(
        "Isaacsimenvs-FurnitureBenchLeg-Direct-v0",
        cfg=cfg,
        render_mode="rgb_array" if args.record_video else None,
    )
    inner = env.unwrapped

    player = RlPlayer(
        num_observations=140,
        num_actions=cfg.action_space,
        config_path=str(args.config),
        checkpoint_path=str(args.checkpoint),
        device=str(args.rl_device),
        num_envs=cfg.scene.num_envs,
    )

    object_urdf_text = object_urdf_text_for_env(inner, 0) if args.capture_html else ""
    table_urdf_text = table_urdf_text_for_env(inner, 0) if args.capture_html else ""
    fixture_urdf_text = fixture_urdf_text_for_env(inner, 0) if args.capture_html else None

    summaries = []
    print(
        "[eval_fbleg] "
        f"episodes={args.num_episodes} max_steps={args.max_steps} "
        f"goal_mode={cfg.furniturebench_leg.goal_mode} "
        f"init={cfg.furniturebench_leg.initialization_mode} "
        f"success_mode={cfg.furniturebench_leg.success_mode} "
        f"checkpoint={args.checkpoint}",
        flush=True,
    )

    for episode in range(int(args.num_episodes)):
        obs, _ = env.reset(seed=int(args.seed) + episode)
        player.reset()
        player.player.init_rnn()

        frames: list[np.ndarray] = []
        html_frames = []
        object_pose_log: list[np.ndarray] = []
        goal_pose_log: list[np.ndarray] = []
        action_log: list[np.ndarray] = []
        reward_log: list[float] = []
        successes_log: list[int] = []
        near_goal_log: list[bool] = []
        is_success_log: list[bool] = []
        keypoints_max_dist_log: list[float] = []
        omnireset_pos_error_log: list[float] = []
        omnireset_xy_rot_error_log: list[float] = []
        screw_cw_turns_log: list[float] = []
        screw_max_cw_turns_log: list[float] = []
        screw_depth_log: list[float] = []
        screw_max_depth_log: list[float] = []
        screw_radial_error_log: list[float] = []
        screw_phase_error_log: list[float] = []
        screw_pushthrough_log: list[bool] = []
        screw_insert_like_log: list[bool] = []
        screw_required_depth_value = 0.0
        reward_terms_log: list[dict[str, float]] = []
        termination_reasons_log: list[dict[str, bool]] = []
        goal_hit_steps: list[dict[str, int]] = []
        env_max_goals = int(inner.env_max_goals[0].detach().cpu().item())
        max_successes_seen = int(inner._successes[0].detach().cpu().item())
        done = False
        success = False
        done_step = int(args.max_steps) - 1
        post_success_left = -1

        for step in range(int(args.max_steps)):
            if args.capture_html:
                html_frames.append(capture_pose_viewer_frame(inner, 0))
            if args.record_video and step % int(args.frame_every) == 0:
                frame = _render_frame(env)
                if frame is not None:
                    frames.append(frame)

            with torch.no_grad():
                policy_obs = obs["policy"].to(args.rl_device)
                action = player.get_normalized_action(policy_obs, deterministic_actions=True)
            obs, reward, terminated, truncated, _ = env.step(action.to(inner.device))

            object_pose_log.append(_root_pose_xyzw(inner, inner.object))
            goal_pose_log.append(_root_pose_xyzw(inner, inner.goal_viz))
            action_log.append(action[0].detach().cpu().numpy())
            reward_log.append(float(reward[0].detach().cpu().item()))
            successes_log.append(int(inner._successes[0].detach().cpu().item()))
            near_goal_log.append(bool(inner._near_goal[0].detach().cpu().item()))
            is_success_log.append(bool(inner._is_success[0].detach().cpu().item()))
            keypoints_max_dist_log.append(_scalar_from_tensor(getattr(inner, "_keypoints_max_dist", None)))
            omnireset_pos_error_log.append(_scalar_from_tensor(getattr(inner, "_omnireset_pos_align_error", None)))
            omnireset_xy_rot_error_log.append(
                _scalar_from_tensor(getattr(inner, "_omnireset_xy_rot_align_error", None))
            )
            screw_cw_turns_log.append(_scalar_from_tensor(getattr(inner, "_leg_screw_cw_turns", None)))
            screw_max_cw_turns_log.append(_scalar_from_tensor(getattr(inner, "_leg_screw_max_cw_turns", None)))
            screw_depth_log.append(_scalar_from_tensor(getattr(inner, "_leg_screw_depth", None)))
            screw_max_depth_log.append(_scalar_from_tensor(getattr(inner, "_leg_screw_max_depth", None)))
            screw_radial_error_log.append(_scalar_from_tensor(getattr(inner, "_leg_screw_radial_error", None)))
            screw_phase_error_log.append(_scalar_from_tensor(getattr(inner, "_leg_screw_phase_error", None)))
            screw_pushthrough_log.append(bool(inner._leg_screw_pushthrough[0].detach().cpu().item()))
            screw_insert_like, screw_required_depth = inner._screw_insert_like()
            screw_insert_like_log.append(bool(screw_insert_like[0].detach().cpu().item()))
            screw_required_depth_value = _scalar_from_tensor(screw_required_depth)
            reward_terms_log.append(
                {
                    name: _scalar_from_tensor(value)
                    for name, value in getattr(inner, "_reward_terms", {}).items()
                }
            )
            termination_reasons_log.append(_termination_reasons(inner))

            successes = int(inner._successes[0].detach().cpu().item())
            if successes > max_successes_seen:
                for goal_idx in range(max_successes_seen, successes):
                    goal_hit_steps.append({"goal_index": int(goal_idx), "step": int(step)})
                max_successes_seen = successes

            reasons = _termination_reasons(inner)
            env_done = bool(terminated[0].item()) or bool(truncated[0].item())
            if bool(reasons.get("max_successes", False)) and max_successes_seen < env_max_goals:
                # Isaac Lab may auto-reset the env before user code observes the
                # final _successes value.  The termination reason is still the
                # authoritative signal that the final goal was reached.
                for goal_idx in range(max_successes_seen, env_max_goals):
                    goal_hit_steps.append({"goal_index": int(goal_idx), "step": int(step)})
                max_successes_seen = env_max_goals
            all_goals_hit = max_successes_seen >= env_max_goals

            if all_goals_hit and post_success_left < 0:
                success = True
                done = True
                done_step = step
                post_success_left = int(args.continue_after_success_steps)

            if post_success_left >= 0:
                post_success_left -= 1
                if post_success_left <= 0:
                    break
            elif env_done and bool(args.stop_on_env_done):
                done = True
                success = bool(reasons.get("max_successes", False))
                done_step = step
                break

        if args.capture_html and html_frames:
            html_path = out_dir / f"episode_{episode:02d}_{'success' if success else 'fail'}.html"
            html = build_pose_viewer_html(
                frames=html_frames,
                object_urdf_text=object_urdf_text,
                table_urdf_text=table_urdf_text,
                fixture_urdf_text=fixture_urdf_text,
                url_check="skip",
            )
            html_path.write_text(html, encoding="utf-8")
        else:
            html_path = None

        if args.record_video and frames:
            import imageio

            video_path = out_dir / f"episode_{episode:02d}_{'success' if success else 'fail'}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=int(args.video_fps))
        else:
            video_path = None

        npz_path = out_dir / f"episode_{episode:02d}_{'success' if success else 'fail'}_trajectory.npz"
        np.savez(
            npz_path,
            object_pose_xyzw=np.asarray(object_pose_log, dtype=np.float32),
            goal_pose_xyzw=np.asarray(goal_pose_log, dtype=np.float32),
            actions=np.asarray(action_log, dtype=np.float32),
            rewards=np.asarray(reward_log, dtype=np.float32),
            successes=np.asarray(successes_log, dtype=np.int32),
            near_goal=np.asarray(near_goal_log, dtype=np.bool_),
            is_success=np.asarray(is_success_log, dtype=np.bool_),
            keypoints_max_dist=np.asarray(keypoints_max_dist_log, dtype=np.float32),
            omnireset_pos_align_error=np.asarray(omnireset_pos_error_log, dtype=np.float32),
            omnireset_xy_rot_align_error=np.asarray(omnireset_xy_rot_error_log, dtype=np.float32),
            screw_cw_turns_after_entry=np.asarray(screw_cw_turns_log, dtype=np.float32),
            screw_max_cw_turns_after_entry=np.asarray(screw_max_cw_turns_log, dtype=np.float32),
            screw_depth_after_entry_m=np.asarray(screw_depth_log, dtype=np.float32),
            screw_max_depth_after_entry_m=np.asarray(screw_max_depth_log, dtype=np.float32),
            screw_radial_error_m=np.asarray(screw_radial_error_log, dtype=np.float32),
            screw_phase_error_rad=np.asarray(screw_phase_error_log, dtype=np.float32),
            screw_pushthrough=np.asarray(screw_pushthrough_log, dtype=np.bool_),
            screw_insert_like=np.asarray(screw_insert_like_log, dtype=np.bool_),
            screw_required_depth_m=np.asarray([screw_required_depth_value], dtype=np.float32),
            reward_terms_json=np.asarray([json.dumps(reward_terms_log)], dtype=object),
            termination_reasons_json=np.asarray([json.dumps(termination_reasons_log)], dtype=object),
            goal_hit_steps_json=np.asarray([json.dumps(goal_hit_steps)], dtype=object),
            success=np.asarray([bool(success)], dtype=np.bool_),
            env_max_goals=np.asarray([int(env_max_goals)], dtype=np.int32),
            step_dt=np.asarray([float(inner.step_dt)], dtype=np.float32),
        )

        min_pos_error = float(min(omnireset_pos_error_log) if omnireset_pos_error_log else 0.0)
        min_xy_rot_error = float(min(omnireset_xy_rot_error_log) if omnireset_xy_rot_error_log else 0.0)
        min_keypoints_max_dist = float(min(keypoints_max_dist_log) if keypoints_max_dist_log else 0.0)
        min_screw_radial_error = float(min(screw_radial_error_log) if screw_radial_error_log else 0.0)
        min_screw_phase_error = float(min(screw_phase_error_log) if screw_phase_error_log else 0.0)

        summary = {
            "episode": int(episode),
            "label": "success" if success else "fail",
            "success": bool(success),
            "done": bool(done),
            "done_step": int(done_step),
            "goal_hit_steps": goal_hit_steps,
            "successes": int(max_successes_seen),
            "env_max_goals": int(env_max_goals),
            "termination_reasons": _termination_reasons(inner),
            "omnireset_pos_align_error_end": _scalar_from_tensor(getattr(inner, "_omnireset_pos_align_error", None)),
            "omnireset_xy_rot_align_error_end": _scalar_from_tensor(
                getattr(inner, "_omnireset_xy_rot_align_error", None)
            ),
            "keypoints_max_dist_end": _scalar_from_tensor(getattr(inner, "_keypoints_max_dist", None)),
            "omnireset_pos_align_error_min": min_pos_error,
            "omnireset_xy_rot_align_error_min": min_xy_rot_error,
            "keypoints_max_dist_min": min_keypoints_max_dist,
            "screw_radial_error_min": min_screw_radial_error,
            "screw_phase_error_min": min_screw_phase_error,
            "screw_max_cw_turns_after_entry": float(max(screw_max_cw_turns_log) if screw_max_cw_turns_log else 0.0),
            "screw_max_depth_after_entry_m": float(max(screw_max_depth_log) if screw_max_depth_log else 0.0),
            "screw_pushthrough": bool(any(screw_pushthrough_log)),
            "screw_insert_like": bool(any(screw_insert_like_log)),
            "video": str(video_path) if video_path is not None else None,
            "html": str(html_path) if html_path is not None else None,
            "npz": str(npz_path),
        }
        summaries.append(summary)
        print(
            "[eval_fbleg] "
            f"episode={episode:02d} label={summary['label']} "
            f"done_step={summary['done_step']} goals={summary['successes']}/{summary['env_max_goals']} "
            f"hits={goal_hit_steps} "
            f"min_pos_err={summary['omnireset_pos_align_error_min']:.4f} "
            f"min_ori_err={summary['omnireset_xy_rot_align_error_min']:.4f} "
            f"min_kp={summary['keypoints_max_dist_min']:.4f} "
            f"screw_turns={summary['screw_max_cw_turns_after_entry']:.3f} "
            f"screw_depth={summary['screw_max_depth_after_entry_m']:.4f} "
            f"pushthrough={summary['screw_pushthrough']} "
            f"insert_like={summary['screw_insert_like']} "
            f"video={video_path} html={html_path}",
            flush=True,
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[eval_fbleg] wrote {summary_path}", flush=True)

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
