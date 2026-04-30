# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out an OmniReset leg policy and estimate axial threading turns."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Analyze OmniReset FurnitureBench leg spin during rollout.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during rollout.")
parser.add_argument("--video_length", type=int, default=240, help="Length of the recorded video in policy steps.")
parser.add_argument("--max_steps", type=int, default=None, help="Maximum policy steps to roll out.")
parser.add_argument("--output_dir", type=str, default="outputs/omnireset_leg_spin_analysis")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the published checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--entry_xy_threshold",
    type=float,
    default=0.006,
    help="Radial xy distance to the hole center used for first-hole-entry detection.",
)
parser.add_argument(
    "--entry_z_threshold",
    type=float,
    default=0.012,
    help=(
        "Relative z threshold, in meters above the final assembled offset, used for "
        "first-hole-entry detection."
    ),
)
parser.add_argument(
    "--success_consecutive_steps",
    type=int,
    default=3,
    help="Consecutive task-success steps required to call the leg fully threaded/down.",
)
parser.add_argument(
    "--stop_after_success_steps",
    type=int,
    default=20,
    help="Stop this many policy steps after the first success window starts; set <=0 to disable.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat, subtract_frame_transforms, wrap_to_pi
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from uwlab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401


def _first_consecutive_true(mask: np.ndarray, count: int) -> int | None:
    count = max(1, int(count))
    if len(mask) < count:
        return None
    streak = 0
    for idx, value in enumerate(mask):
        streak = streak + 1 if bool(value) else 0
        if streak >= count:
            return idx - count + 1
    return None


def _analyze_turns(data: dict[str, np.ndarray], args) -> dict[str, float | int | None | bool]:
    rel_pos = data["rel_align_pos"]
    rel_quat_wxyz = data["rel_align_quat_wxyz"]
    success = data["task_success"].astype(bool)

    radial_xy = np.linalg.norm(rel_pos[:, :2], axis=1)
    rel_z = rel_pos[:, 2]
    entry_mask = (radial_xy < float(args.entry_xy_threshold)) & (rel_z < float(args.entry_z_threshold))
    success_step = _first_consecutive_true(success, int(args.success_consecutive_steps))

    entry_step = None
    if success_step is not None:
        candidates = np.flatnonzero(entry_mask[: success_step + 1])
        if len(candidates):
            entry_step = int(candidates[0])
    if entry_step is None:
        candidates = np.flatnonzero(entry_mask)
        if len(candidates):
            entry_step = int(candidates[0])

    rel_quat_xyzw = rel_quat_wxyz[:, [1, 2, 3, 0]]
    rot_mats = R.from_quat(rel_quat_xyzw).as_matrix()
    yaw = np.unwrap(np.arctan2(rot_mats[:, 1, 0], rot_mats[:, 0, 0]))
    data["rel_yaw_unwrapped_rad"] = yaw.astype(np.float32)

    analysis: dict[str, float | int | None | bool] = {
        "num_steps": int(len(success)),
        "success_step": None if success_step is None else int(success_step),
        "entry_step": None if entry_step is None else int(entry_step),
        "success_consecutive_steps": int(args.success_consecutive_steps),
        "entry_xy_threshold_m": float(args.entry_xy_threshold),
        "entry_z_threshold_m": float(args.entry_z_threshold),
        "succeeded": bool(success_step is not None),
    }
    if entry_step is not None:
        analysis.update(
            {
                "entry_time_s": float(data["time_s"][entry_step]),
                "entry_radial_xy_m": float(radial_xy[entry_step]),
                "entry_rel_z_m": float(rel_z[entry_step]),
                "entry_yaw_rad": float(yaw[entry_step]),
            }
        )
    if success_step is not None:
        analysis.update(
            {
                "success_time_s": float(data["time_s"][success_step]),
                "success_radial_xy_m": float(radial_xy[success_step]),
                "success_rel_z_m": float(rel_z[success_step]),
                "success_yaw_rad": float(yaw[success_step]),
                "success_xyz_distance_m": float(data["xyz_distance"][success_step]),
                "success_euler_xy_error_rad": float(data["euler_xy_distance"][success_step]),
            }
        )
    if entry_step is not None and success_step is not None and success_step >= entry_step:
        delta_yaw = float(yaw[success_step] - yaw[entry_step])
        analysis.update(
            {
                "signed_turns_entry_to_success": delta_yaw / (2.0 * np.pi),
                "absolute_turns_entry_to_success": abs(delta_yaw) / (2.0 * np.pi),
                "duration_entry_to_success_s": float(data["time_s"][success_step] - data["time_s"][entry_step]),
                "steps_entry_to_success": int(success_step - entry_step),
            }
        )
    return analysis


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            raise RuntimeError(f"No published checkpoint found for task {train_task_name}.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        from isaaclab_tasks.utils import get_checkpoint_path

        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    manager_env = env.unwrapped

    if args_cli.video:
        video_kwargs = {
            "video_folder": str(out_dir / "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": int(args_cli.video_length),
            "disable_logger": True,
        }
        print("[INFO] Recording videos during rollout.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=manager_env.device)
    policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))

    dt = float(env.unwrapped.step_dt)
    max_steps = int(args_cli.max_steps if args_cli.max_steps is not None else args_cli.video_length)

    command_term = manager_env.command_manager.get_term("task_command")
    insertive = manager_env.scene["insertive_object"]
    receptive = manager_env.scene["receptive_object"]

    logs: dict[str, list[np.ndarray | float | bool]] = {
        "time_s": [],
        "insertive_root_pos_w": [],
        "insertive_root_quat_wxyz": [],
        "receptive_root_pos_w": [],
        "receptive_root_quat_wxyz": [],
        "insertive_align_pos_w": [],
        "insertive_align_quat_wxyz": [],
        "receptive_align_pos_w": [],
        "receptive_align_quat_wxyz": [],
        "rel_align_pos": [],
        "rel_align_quat_wxyz": [],
        "xyz_distance": [],
        "euler_xy_distance": [],
        "task_success": [],
        "done": [],
    }

    obs = env.get_observations()
    first_success_step = None
    timestep = 0
    while simulation_app.is_running() and timestep < max_steps:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if policy_nn is not None:
                policy_nn.reset(dones)

            ins_align_pos, ins_align_quat = command_term.insertive_asset_offset.apply(insertive)
            rec_align_pos, rec_align_quat = command_term.receptive_asset_offset.apply(receptive)
            rel_pos, rel_quat = subtract_frame_transforms(rec_align_pos, rec_align_quat, ins_align_pos, ins_align_quat)
            # Recompute via Isaac Lab helper to match task-command success metric.
            e_x, e_y, _ = euler_xyz_from_quat(rel_quat)
            euler_xy_distance = wrap_to_pi(e_x).abs() + wrap_to_pi(e_y).abs()
            xyz_distance = torch.norm(rel_pos, dim=1)
            task_success = (
                (xyz_distance < float(command_term.success_position_threshold))
                & (euler_xy_distance < float(command_term.success_orientation_threshold))
            )

        logs["time_s"].append(float(timestep * dt))
        logs["insertive_root_pos_w"].append(insertive.data.root_pos_w[0].detach().cpu().numpy())
        logs["insertive_root_quat_wxyz"].append(insertive.data.root_quat_w[0].detach().cpu().numpy())
        logs["receptive_root_pos_w"].append(receptive.data.root_pos_w[0].detach().cpu().numpy())
        logs["receptive_root_quat_wxyz"].append(receptive.data.root_quat_w[0].detach().cpu().numpy())
        logs["insertive_align_pos_w"].append(ins_align_pos[0].detach().cpu().numpy())
        logs["insertive_align_quat_wxyz"].append(ins_align_quat[0].detach().cpu().numpy())
        logs["receptive_align_pos_w"].append(rec_align_pos[0].detach().cpu().numpy())
        logs["receptive_align_quat_wxyz"].append(rec_align_quat[0].detach().cpu().numpy())
        logs["rel_align_pos"].append(rel_pos[0].detach().cpu().numpy())
        logs["rel_align_quat_wxyz"].append(rel_quat[0].detach().cpu().numpy())
        logs["xyz_distance"].append(float(xyz_distance[0].detach().cpu().item()))
        logs["euler_xy_distance"].append(float(euler_xy_distance[0].detach().cpu().item()))
        logs["task_success"].append(bool(task_success[0].detach().cpu().item()))
        logs["done"].append(bool(dones[0].detach().cpu().item()))

        if bool(task_success[0].detach().cpu().item()) and first_success_step is None:
            first_success_step = timestep
            print(f"[analysis] first task-success step={timestep} time={timestep * dt:.3f}s", flush=True)

        timestep += 1
        if (
            first_success_step is not None
            and int(args_cli.stop_after_success_steps) > 0
            and timestep >= first_success_step + int(args_cli.stop_after_success_steps)
        ):
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

    data = {
        key: np.asarray(value, dtype=np.float32 if key not in {"task_success", "done"} else np.bool_)
        for key, value in logs.items()
    }
    analysis = _analyze_turns(data, args_cli)

    npz_path = out_dir / "omnireset_leg_spin_trace.npz"
    np.savez(npz_path, **data)
    analysis_path = out_dir / "omnireset_leg_spin_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"[analysis] wrote {npz_path}", flush=True)
    print(f"[analysis] wrote {analysis_path}", flush=True)
    print("[analysis] " + json.dumps(analysis, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
