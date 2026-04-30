from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.observation_action_utils_sharpa import compute_joint_pos_targets
from isaacsim_conversion.distill_env import IsaacSimDistillEnv
from isaacsim_conversion.isaacsim_env import CONTROL_DT, _log
from isaacsim_conversion.interactive_viewer import write_pose_viewer_html
from isaacsim_conversion.student_policy import (
    MLPPolicy,
    MLPRecurrentPolicy,
    MonoTransformerRecurrentPolicy,
    preprocess_image,
    resize_image,
)
from isaacsim_conversion.task_utils import (
    CameraIntrinsics,
    CameraPose,
    default_real_camera_pose,
    default_real_camera_transform,
    load_task_spec,
    load_yaml,
)


def launch_app():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "train_online", "teacher_eval", "student_eval", "mixed_eval", "camera_debug"],
        default="train",
    )
    parser.add_argument("--task_source", choices=["fabrica", "dextoolbench"], default="dextoolbench")
    parser.add_argument("--assembly", default="beam")
    parser.add_argument("--part_id", default="2")
    parser.add_argument("--collision_method", default="coacd")
    parser.add_argument("--object_category", default="hammer")
    parser.add_argument("--object_name", default="claw_hammer")
    parser.add_argument("--task_name", default="swing_down")
    parser.add_argument("--teacher_checkpoint", default="pretrained_policy/model.pth")
    parser.add_argument("--teacher_config", default="pretrained_policy/config.yaml")
    parser.add_argument("--student_checkpoint", default=None)
    parser.add_argument("--student_arch", default="mono_transformer_recurrent")
    parser.add_argument("--student_input", choices=["camera", "teacher_obs"], default="camera")
    parser.add_argument("--student_modality", choices=["depth", "rgb", "rgbd"], default=None)
    parser.add_argument("--distill_config", default="isaacsim_conversion/configs/hammer_distill.yaml")
    parser.add_argument("--camera_config", default="isaacsim_conversion/configs/hammer_camera.yaml")
    parser.add_argument("--camera_backend", choices=["tiled", "standard"], default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--env_spacing", type=float, default=None)
    parser.add_argument("--ground_plane_size", type=float, default=None)
    parser.add_argument("--object_start_mode", choices=["fixed", "randomized"], default=None)
    parser.add_argument(
        "--beta_mode",
        choices=["metric_driven", "fixed_decay", "always_teacher", "always_student"],
        default=None,
    )
    parser.add_argument("--beta_start", type=float, default=None)
    parser.add_argument("--beta_end", type=float, default=None)
    parser.add_argument("--beta_decay", type=float, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--online_num_iters", type=int, default=None)
    parser.add_argument("--online_log_interval", type=int, default=None)
    parser.add_argument("--online_update_interval", type=int, default=None)
    parser.add_argument("--episode_length", type=int, default=None)
    parser.add_argument("--reset_when_dropped", action="store_true")
    parser.add_argument("--capture_frames", action="store_true")
    parser.add_argument("--capture_frame_stride", type=int, default=1)
    parser.add_argument("--capture_viewer", action="store_true")
    parser.add_argument("--capture_viewer_len", type=int, default=700)
    parser.add_argument("--capture_viewer_interval", type=int, default=None)
    parser.add_argument("--capture_viewer_env_id", type=int, default=0)
    parser.add_argument("--capture_viewer_wandb_key", default="interactive_viewer")
    parser.add_argument("--capture_viewer_video", action="store_true")
    parser.add_argument("--capture_viewer_video_wandb_key", default="rollout_video")
    parser.add_argument("--capture_viewer_video_fps", type=int, default=60)
    parser.add_argument("--debug_policy_image_stats", action="store_true")
    parser.add_argument("--debug_policy_image_stats_stride", type=int, default=100)
    parser.add_argument("--debug_policy_image_stats_env_ids", default="0")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="depthbasedRL-isaacsim-distill")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    return app_launcher.app, args


_app, _args = launch_app()


@dataclass
class DistillSettings:
    num_episodes: int = 10
    max_steps: int = 700
    action_loss_weight: float = 1.0
    aux_object_pos_weight: float = 1.0
    hand_moving_average: float = 0.1
    arm_moving_average: float = 0.1
    dof_speed_scale: float = 1.5
    control_dt: float = 1.0 / 60.0
    beta_mode: str = "metric_driven"
    beta_start: float = 1.0
    beta_end: float = 0.0
    beta_decay: float = 0.1
    beta_metric_threshold: float = 0.8
    beta_update_episodes: int = 2
    beta_hold_episodes: int = 0
    checkpoint_interval: int = 5
    eval_interval: int = 999
    eval_num_envs: int = 0
    eval_max_steps: int = 700
    image_height: int = 224
    image_width: int = 224
    learning_rate: float = 1e-4
    num_envs: int = 1
    env_spacing: float = 2.0
    ground_plane_size: float = 500.0
    object_start_mode: str = "fixed"
    object_pos_noise_xyz: tuple[float, float, float] = (0.03, 0.03, 0.01)
    object_yaw_noise_deg: float = 20.0
    monitor_num_envs: int = 0
    monitor_log_window: int = 10
    camera_backend: str = "tiled"
    depth_preprocess_mode: str = "clip_divide"
    depth_min_m: float = 0.0
    depth_max_m: float = 5.0
    episode_length: int = 600
    reset_when_dropped: bool = False
    online_num_iters: int = 10000
    online_log_interval: int = 500
    online_update_interval: int = 1


class BetaScheduler:
    def __init__(self, cfg: DistillSettings):
        self.cfg = cfg
        self.beta = cfg.beta_start
        self.episode_counter = 0
        self.window: list[float] = []

    def value(self) -> float:
        if self.cfg.beta_mode == "always_teacher":
            return 1.0
        if self.cfg.beta_mode == "always_student":
            return 0.0
        return self.beta

    def update(self, goal_completion_ratio: float):
        self.episode_counter += 1
        self.window.append(goal_completion_ratio)
        if self.episode_counter <= self.cfg.beta_hold_episodes:
            return
        if self.cfg.beta_mode == "fixed_decay":
            self.beta = max(self.cfg.beta_end, self.beta - self.cfg.beta_decay)
            return
        if self.cfg.beta_mode != "metric_driven":
            return
        if len(self.window) < self.cfg.beta_update_episodes:
            return
        avg_metric = float(np.mean(self.window[-self.cfg.beta_update_episodes :]))
        if avg_metric >= self.cfg.beta_metric_threshold:
            self.beta = max(self.cfg.beta_end, self.beta - self.cfg.beta_decay)


def load_distill_settings(path: Path, args) -> DistillSettings:
    cfg = load_yaml(path)
    settings = DistillSettings(**cfg)
    if args.max_steps is not None:
        settings.max_steps = args.max_steps
    if args.num_episodes is not None:
        settings.num_episodes = args.num_episodes
    if args.num_envs is not None:
        settings.num_envs = args.num_envs
    if args.env_spacing is not None:
        settings.env_spacing = args.env_spacing
    if args.ground_plane_size is not None:
        settings.ground_plane_size = args.ground_plane_size
    if args.object_start_mode is not None:
        settings.object_start_mode = args.object_start_mode
    if args.beta_mode is not None:
        settings.beta_mode = args.beta_mode
    if args.beta_start is not None:
        settings.beta_start = args.beta_start
    if args.beta_end is not None:
        settings.beta_end = args.beta_end
    if args.beta_decay is not None:
        settings.beta_decay = args.beta_decay
    if args.eval_interval is not None:
        settings.eval_interval = args.eval_interval
    if args.checkpoint_interval is not None:
        settings.checkpoint_interval = args.checkpoint_interval
    if args.online_num_iters is not None:
        settings.online_num_iters = args.online_num_iters
    if args.online_log_interval is not None:
        settings.online_log_interval = args.online_log_interval
    if args.online_update_interval is not None:
        settings.online_update_interval = args.online_update_interval
    if args.episode_length is not None:
        settings.episode_length = args.episode_length
    if args.reset_when_dropped:
        settings.reset_when_dropped = True
    return settings


def validate_distill_settings(settings: DistillSettings):
    if settings.monitor_num_envs < 0:
        raise ValueError("monitor_num_envs must be >= 0")
    if settings.monitor_num_envs >= settings.num_envs:
        raise ValueError(
            f"monitor_num_envs ({settings.monitor_num_envs}) must be smaller than num_envs ({settings.num_envs})"
        )
    if settings.monitor_log_window <= 0:
        raise ValueError("monitor_log_window must be > 0")
    if settings.depth_preprocess_mode not in {"clip_divide", "window_normalize", "metric"}:
        raise ValueError(f"Unsupported depth_preprocess_mode={settings.depth_preprocess_mode!r}")
    if settings.depth_max_m <= settings.depth_min_m:
        raise ValueError("depth_max_m must be greater than depth_min_m")
    if settings.episode_length <= 0:
        raise ValueError("episode_length must be > 0")
    if settings.online_num_iters <= 0:
        raise ValueError("online_num_iters must be > 0")
    if settings.online_log_interval <= 0:
        raise ValueError("online_log_interval must be > 0")
    if settings.online_update_interval <= 0:
        raise ValueError("online_update_interval must be > 0")


def load_camera_pose(path: Path) -> tuple[str, CameraPose, CameraIntrinsics]:
    cfg = load_yaml(path)
    modality = cfg.get("modality", "depth")
    intrinsics = CameraIntrinsics(
        width=int(cfg.get("width", 640)),
        height=int(cfg.get("height", 480)),
        focal_length=float(cfg.get("focal_length", 24.0)),
        horizontal_aperture=float(cfg.get("horizontal_aperture", 20.955)),
        focus_distance=float(cfg.get("focus_distance", 400.0)),
        clipping_range=tuple(float(x) for x in cfg.get("clipping_range", [0.1, 100.0])),
    )
    if cfg.get("pose_source") == "real_camera_t_w_c":
        pose = default_real_camera_pose()
        return modality, CameraPose(
            pos=pose.pos,
            quat_wxyz=pose.quat_wxyz,
            convention=str(cfg.get("convention", pose.convention)),
            mount=str(cfg.get("mount", pose.mount)),
            link_name=cfg.get("link_name", pose.link_name),
        ), intrinsics
    pose_cfg = cfg.get("pose", {})
    return modality, CameraPose(
        pos=tuple(float(x) for x in pose_cfg["pos"]),
        quat_wxyz=tuple(float(x) for x in pose_cfg["quat_wxyz"]),
        convention=str(cfg.get("convention", pose_cfg.get("convention", "ros"))),
        mount=str(cfg.get("mount", pose_cfg.get("mount", "world"))),
        link_name=cfg.get("link_name", pose_cfg.get("link_name")),
    ), intrinsics


def resolve_repo_path(repo_root: Path, maybe_relative: str | None) -> str | None:
    if maybe_relative is None:
        return None
    if maybe_relative.startswith("/"):
        return maybe_relative
    return str(repo_root / maybe_relative)


def choose_stepping_action(beta: float, teacher_action: torch.Tensor, student_action: torch.Tensor) -> torch.Tensor:
    if beta >= 1.0:
        return teacher_action
    if beta <= 0.0:
        return student_action
    use_teacher = torch.rand(teacher_action.shape[0], device=teacher_action.device) < beta
    mask = use_teacher.unsqueeze(-1)
    return torch.where(mask, teacher_action, student_action)


def get_monitor_env_mask(num_envs: int, monitor_num_envs: int) -> np.ndarray:
    mask = np.zeros(num_envs, dtype=bool)
    if monitor_num_envs > 0:
        mask[-monitor_num_envs:] = True
    return mask


def subset_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if mask.size == 0 or not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def compute_subset_progress_metrics(env: IsaacSimDistillEnv, sim_state, mask: np.ndarray) -> dict[str, float]:
    if mask.size == 0 or not np.any(mask):
        return {
            "goal_idx": 0.0,
            "goal_completion_ratio": 0.0,
            "kp_dist": 0.0,
            "near_goal_steps": 0.0,
        }
    progress_goal_idx = np.maximum(env.goal_idx, env.max_goal_idx).astype(np.float32)
    goal_completion = progress_goal_idx / len(env.task_spec.goals)
    return {
        "goal_idx": subset_mean(progress_goal_idx, mask),
        "goal_completion_ratio": subset_mean(goal_completion, mask),
        "kp_dist": subset_mean(sim_state.kp_dist.astype(np.float32), mask),
        "near_goal_steps": subset_mean(env.near_goal_steps.astype(np.float32), mask),
    }


def stack_student_image(student_obs: dict[str, torch.Tensor], modality: str, settings: DistillSettings) -> torch.Tensor:
    images = student_obs["images"]
    if modality == "depth":
        image = preprocess_image(images["depth"], modality)
    elif modality == "rgb":
        image = preprocess_image(images["rgb"], modality)
    elif modality == "rgbd":
        image = torch.cat([images["rgb"], images["depth"]], dim=1)
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return resize_image(image, settings.image_height, settings.image_width)


def parse_env_id_list(value: str, num_envs: int) -> list[int]:
    env_ids = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        env_id = int(part)
        if env_id < 0:
            env_id = num_envs + env_id
        if env_id < 0 or env_id >= num_envs:
            raise ValueError(f"debug env id {env_id} out of range for num_envs={num_envs}")
        env_ids.append(env_id)
    return sorted(set(env_ids))


def tensor_image_stats(image: torch.Tensor, env_ids: list[int], prefix: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    with torch.no_grad():
        for env_id in env_ids:
            img = image[env_id].detach().float()
            stats[f"{prefix}/env{env_id}/mean"] = float(img.mean().item())
            stats[f"{prefix}/env{env_id}/std"] = float(img.std(unbiased=False).item())
            stats[f"{prefix}/env{env_id}/min"] = float(img.min().item())
            stats[f"{prefix}/env{env_id}/max"] = float(img.max().item())
            stats[f"{prefix}/env{env_id}/bright_frac"] = float((img > 0.86).float().mean().item())
            stats[f"{prefix}/env{env_id}/dark_frac"] = float((img < 0.14).float().mean().item())
    return stats


def append_debug_policy_image_stats(
    run_dir: Path,
    iter_idx: int,
    env_ids: list[int],
    student_obs: dict[str, torch.Tensor],
    student_image: torch.Tensor,
    modality: str,
):
    row: dict[str, float | int | str] = {
        "iter": iter_idx,
        "modality": modality,
        "env_ids": ",".join(str(env_id) for env_id in env_ids),
    }
    images = student_obs["images"]
    if "rgb" in images:
        row.update(tensor_image_stats(images["rgb"], env_ids, "raw_rgb"))
    if "depth" in images:
        row.update(tensor_image_stats(images["depth"], env_ids, "raw_depth"))
    row.update(tensor_image_stats(student_image, env_ids, "policy_image"))
    path = run_dir / "policy_image_stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_student(args, env: IsaacSimDistillEnv, settings: DistillSettings):
    if args.student_input == "teacher_obs":
        if args.student_arch == "mlp_recurrent":
            return MLPRecurrentPolicy(
                obs_dim=140,
                action_dim=env.action_dim,
                aux_heads={"object_pos": 3},
            ).to(env.device)
        if args.student_arch == "mlp":
            return MLPPolicy(
                obs_dim=140,
                action_dim=env.action_dim,
                aux_heads={"object_pos": 3},
            ).to(env.device)
        raise ValueError("teacher_obs student_input currently requires --student_arch mlp_recurrent or mlp")
    if args.student_arch != "mono_transformer_recurrent":
        raise ValueError(f"Unsupported student_arch for camera input: {args.student_arch}")
    image_channels = {"depth": 1, "rgb": 3, "rgbd": 4}[env.camera_modality]
    return MonoTransformerRecurrentPolicy(
        image_channels=image_channels,
        proprio_dim=env.student_proprio_dim,
        action_dim=env.action_dim,
        aux_heads={"object_pos": 3},
    ).to(env.device)


def save_checkpoint(path: Path, student: torch.nn.Module, optimizer: torch.optim.Optimizer, episode: int, best_metric: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "best_metric": best_metric,
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_student_checkpoint(path: str | None, student: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> tuple[int, float]:
    if not path:
        return 0, -1.0
    ckpt = torch.load(path, map_location="cpu")
    student.load_state_dict(ckpt["student_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt.get("episode", 0)), float(ckpt.get("best_metric", -1.0))


def ensure_run_dir(repo_root: Path, args, settings: DistillSettings) -> Path:
    if args.run_dir:
        run_dir = Path(resolve_repo_path(repo_root, args.run_dir))
    else:
        run_name = f"{args.object_category}_{args.object_name}_{args.task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = repo_root / "distillation_runs" / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "camera_debug").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_distill_config.yaml", "w") as f:
        yaml.safe_dump(settings.__dict__, f, sort_keys=False)
    return run_dir


def log_csv_row(csv_path: Path, row: dict[str, float | int | str]):
    fieldnames = [
        "episode",
        "mode",
        "goal_idx",
        "goal_completion_ratio",
        "kp_dist",
        "near_goal_steps",
        "episode_steps",
        "episode_wall_time_s",
        "env_steps_per_s",
        "action_loss",
        "action_rmse",
        "aux_object_pos_loss",
        "aux_object_pos_rmse_m",
        "beta",
        "goal_completion_ratio_window",
        "goal_idx_window",
        "kp_dist_window",
        "action_loss_window",
        "action_rmse_window",
        "aux_object_pos_loss_window",
        "aux_object_pos_rmse_m_window",
        "reset_object_z_low",
        "reset_time_limit",
        "reset_max_goals",
        "reset_hand_far",
        "reset_dropped_after_lift",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    normalized_row = {name: row.get(name, "") for name in fieldnames}
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row)


def record_metrics(run_dir: Path, row: dict[str, float | int | str], wandb_run=None, step: int | None = None):
    log_csv_row(run_dir / "metrics.csv", row)
    wandb_payload = {
        f"{row['mode']}/{key}": value
        for key, value in row.items()
        if key not in ("episode", "mode")
    }
    wandb_payload["episode"] = row["episode"]
    wandb_log(wandb_run, wandb_payload, step=step)


def force_exit(code: int = 0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def detach_hidden_state(hidden_state):
    if hidden_state is None:
        return None
    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    if isinstance(hidden_state, tuple):
        return tuple(detach_hidden_state(item) for item in hidden_state)
    if isinstance(hidden_state, list):
        return [detach_hidden_state(item) for item in hidden_state]
    return hidden_state


def reset_hidden_state(hidden_state, env_ids: np.ndarray):
    if hidden_state is None or env_ids.size == 0:
        return hidden_state
    if isinstance(hidden_state, torch.Tensor):
        hidden_state = hidden_state.clone()
        env_ids_t = torch.as_tensor(env_ids, device=hidden_state.device, dtype=torch.long)
        hidden_state.index_fill_(0, env_ids_t, 0.0)
        return hidden_state
    if isinstance(hidden_state, tuple):
        return tuple(reset_hidden_state(item, env_ids) for item in hidden_state)
    if isinstance(hidden_state, list):
        return [reset_hidden_state(item, env_ids) for item in hidden_state]
    return hidden_state


def init_wandb(args, run_dir: Path, settings: DistillSettings):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        _log("wandb requested but not installed; continuing without wandb logging")
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or run_dir.name,
        dir=str(run_dir),
        config={
            "mode": args.mode,
            "task_source": args.task_source,
            "object_category": args.object_category,
            "object_name": args.object_name,
            "task_name": args.task_name,
            **settings.__dict__,
        },
        reinit=True,
    )
    return run


def wandb_log(run, payload: dict[str, float | int | str], step: int | None = None):
    if run is None:
        return
    import wandb

    wandb.log(payload, step=step)


def _capture_rgb_frame(env: IsaacSimDistillEnv, env_id: int) -> np.ndarray | None:
    if env.camera is None:
        return None
    outputs = env.camera.data.output
    if outputs.get("rgb") is not None:
        rgb = outputs["rgb"][env_id].detach().cpu().numpy()[..., :3]
        return rgb.astype(np.uint8)
    if outputs.get("distance_to_image_plane") is None:
        return None
    depth = outputs["distance_to_image_plane"][env_id].detach().cpu().numpy()
    if depth.ndim == 3:
        depth = depth[..., 0]
    finite_mask = np.isfinite(depth)
    depth_vis = np.zeros_like(depth, dtype=np.float32)
    if np.any(finite_mask):
        finite_depth = depth[finite_mask]
        depth_min = float(np.min(finite_depth))
        depth_max = float(np.max(finite_depth))
        denom = max(depth_max - depth_min, 1e-6)
        depth_vis[finite_mask] = np.clip((depth[finite_mask] - depth_min) / denom, 0.0, 1.0)
    depth_img = (depth_vis * 255).astype(np.uint8)
    return np.repeat(depth_img[..., None], 3, axis=-1)


def _pad_video_frames_for_codec(frames: list[np.ndarray]) -> list[np.ndarray]:
    padded = []
    for frame in frames:
        h, w = frame.shape[:2]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h == 0 and pad_w == 0:
            padded.append(frame)
            continue
        padded.append(np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge"))
    return padded


def _log_viewer_artifacts(
    *,
    run_dir: Path,
    mode: str,
    frames: list[dict],
    video_frames: list[np.ndarray],
    wandb_run,
    viewer_key: str,
    video_key: str,
    video_fps: int,
    num_goals: int,
    step: int | None = None,
    artifact_label: str | None = None,
):
    if not frames:
        return
    timestamps = np.arange(len(frames), dtype=np.float32) * CONTROL_DT
    artifact_stem = f"{mode}_rollout" if artifact_label is None else f"{mode}_{artifact_label}"
    payload = {
        "title": f"{mode} rollout",
        "mode": mode,
        "env_id": frames[0]["env_id"],
        "num_goals": num_goals,
        "timestamps": timestamps.tolist(),
        "robot_joint_names": frames[0]["robot_joint_names"],
        "robot_joint_positions": np.stack([f["robot_joint_pos"] for f in frames]).tolist(),
        "robot_base_poses": np.stack([f["robot_base_pose"] for f in frames]).tolist(),
        "robot_body_names": frames[0]["robot_body_names"],
        "robot_body_positions": np.stack([f["robot_body_positions"] for f in frames]).tolist(),
        "object_poses": np.stack([f["object_pose"] for f in frames]).tolist(),
        "goal_poses": np.stack([f["goal_pose"] for f in frames]).tolist(),
        "table_poses": np.stack([f["table_pose"] for f in frames]).tolist(),
        "goal_idx": [int(f["goal_idx"]) for f in frames],
        "kp_dist": [float(f["kp_dist"]) for f in frames],
    }
    viewer_path = run_dir / "interactive_viewer" / f"{artifact_stem}.html"
    html_text = write_pose_viewer_html(viewer_path, payload, title=f"{mode} rollout")
    _log(f"Saved interactive viewer: {viewer_path}")

    video_path = None
    if video_frames:
        import imageio.v2 as iio

        video_path = run_dir / "interactive_viewer" / f"{artifact_stem}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        iio.mimsave(video_path, _pad_video_frames_for_codec(video_frames), fps=video_fps, macro_block_size=1)
        _log(f"Saved rollout video: {video_path}")

    if wandb_run is not None:
        import wandb

        wandb_payload = {viewer_key: wandb.Html(html_text)}
        if video_path is not None:
            wandb_payload[video_key] = wandb.Video(str(video_path), fps=video_fps, format="mp4")
        if step is None:
            wandb_run.log(wandb_payload)
        else:
            wandb_run.log(wandb_payload, step=step)
        _log(f"Logged interactive viewer artifacts to wandb keys: {list(wandb_payload)}")


def _finalize_online_viewer_capture(
    *,
    run_dir: Path,
    frames: list[dict] | None,
    video_frames: list[np.ndarray] | None,
    wandb_run,
    viewer_key: str,
    video_key: str,
    video_fps: int,
    num_goals: int,
    step: int,
) -> tuple[list[dict] | None, list[np.ndarray] | None]:
    if not frames:
        return None, None
    _log_viewer_artifacts(
        run_dir=run_dir,
        mode="train_online",
        frames=frames,
        video_frames=[] if video_frames is None else video_frames,
        wandb_run=wandb_run,
        viewer_key=viewer_key,
        video_key=video_key,
        video_fps=video_fps,
        num_goals=num_goals,
        step=step,
        artifact_label=f"rollout_step_{step:07d}",
    )
    return None, None


def run_episode(
    mode: str,
    env: IsaacSimDistillEnv,
    teacher: RlPlayer,
    student,
    optimizer: torch.optim.Optimizer | None,
    settings: DistillSettings,
    beta_scheduler: BetaScheduler,
    run_dir: Path | None = None,
    capture_frames: bool = False,
    capture_frame_stride: int = 1,
    capture_viewer: bool = False,
    capture_viewer_len: int = 700,
    capture_viewer_env_id: int = 0,
    capture_viewer_video: bool = False,
    wandb_run=None,
    capture_viewer_wandb_key: str = "interactive_viewer",
    capture_viewer_video_wandb_key: str = "rollout_video",
    capture_viewer_video_fps: int = 60,
) -> tuple[dict[str, float], dict[str, float] | None]:
    episode_start_time = time.perf_counter()
    teacher.reset()
    env.reset()
    student_hidden = None if student is None else student.initial_state(batch_size=env.num_envs, device=env.device)
    total_action_loss_per_env = np.zeros(env.num_envs, dtype=np.float64)
    total_aux_loss_per_env = np.zeros(env.num_envs, dtype=np.float64)
    total_steps = 0
    monitor_mask = get_monitor_env_mask(env.num_envs, settings.monitor_num_envs if mode == "train" else 0)
    train_mask = ~monitor_mask if mode == "train" else np.ones(env.num_envs, dtype=bool)
    viewer_frames: list[dict] = []
    viewer_video_frames: list[np.ndarray] = []
    if capture_viewer and run_dir is None:
        raise ValueError("capture_viewer requires run_dir")

    for step_i in range(settings.max_steps):
        sim_state = env.compute_sim_state()
        teacher_obs = env.build_teacher_obs(sim_state)
        teacher_obs_tensor = torch.from_numpy(teacher_obs).float().to(env.device)
        with torch.no_grad():
            teacher_action = teacher.get_normalized_action(teacher_obs_tensor, deterministic_actions=True)

        student_out = None
        if student is None or mode == "teacher_eval":
            student_action = teacher_action
        elif _args.student_input == "teacher_obs":
            if mode == "train":
                student_out, student_hidden = student(teacher_obs_tensor, student_hidden)
            else:
                with torch.no_grad():
                    student_out, student_hidden = student(teacher_obs_tensor, student_hidden)
            student_action = student_out.action
        else:
            student_obs = env.build_student_obs(sim_state, camera_modality=env.camera_modality)
            student_image = stack_student_image(student_obs, env.camera_modality, settings)
            if mode == "train":
                student_out, student_hidden = student(student_image, student_obs["proprio"], student_hidden)
            else:
                with torch.no_grad():
                    student_out, student_hidden = student(student_image, student_obs["proprio"], student_hidden)
            student_action = student_out.action

        beta = beta_scheduler.value() if mode in ("train", "mixed_eval") else (1.0 if mode == "teacher_eval" else 0.0)
        stepping_action = choose_stepping_action(beta, teacher_action, student_action)
        if mode == "train" and np.any(monitor_mask):
            monitor_mask_t = torch.from_numpy(monitor_mask).to(device=env.device, dtype=torch.bool)
            stepping_action = stepping_action.clone()
            stepping_action[monitor_mask_t] = student_action[monitor_mask_t]
        targets = compute_joint_pos_targets(
            actions=stepping_action.detach().cpu().numpy(),
            prev_targets=env.prev_targets,
            hand_moving_average=settings.hand_moving_average,
            arm_moving_average=settings.arm_moving_average,
            hand_dof_speed_scale=settings.dof_speed_scale,
            dt=settings.control_dt,
        )
        env.apply_action(targets)
        env.step(render=not _args.headless)
        next_sim_state = env.compute_sim_state()
        if capture_frames and run_dir is not None and (step_i % max(capture_frame_stride, 1) == 0):
            save_camera_debug_step(env, run_dir, step_i)
        if capture_viewer and len(viewer_frames) < capture_viewer_len:
            viewer_frames.append(env.capture_viewer_frame(capture_viewer_env_id, next_sim_state))
            if capture_viewer_video:
                rgb_frame = _capture_rgb_frame(env, capture_viewer_env_id)
                if rgb_frame is not None:
                    viewer_video_frames.append(rgb_frame)
        finished = env.maybe_advance_goal(next_sim_state)
        reset_env_ids = env.reset_done_envs(next_sim_state)
        if reset_env_ids.size > 0:
            student_hidden = reset_hidden_state(student_hidden, reset_env_ids)
            next_sim_state = env.compute_sim_state()

        if student_out is None:
            action_loss_per_env = torch.zeros(env.num_envs, device=env.device)
            aux_loss_per_env = torch.zeros(env.num_envs, device=env.device)
        else:
            action_loss_per_env = torch.mean((student_action - teacher_action) ** 2, dim=1)
            aux_pred = student_out.aux.get("object_pos")
            gt_object_pos = torch.from_numpy(sim_state.object_pos_world_env_frame).float().to(env.device)
            aux_loss_per_env = (
                torch.mean((aux_pred - gt_object_pos) ** 2, dim=1)
                if aux_pred is not None
                else torch.zeros(env.num_envs, device=env.device)
            )
        action_loss = torch.mean(action_loss_per_env)
        aux_loss = torch.mean(aux_loss_per_env)
        total_loss = settings.action_loss_weight * action_loss + settings.aux_object_pos_weight * aux_loss

        if mode == "train" and optimizer is not None and student is not None:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            student_hidden = detach_hidden_state(student_hidden)

        total_action_loss_per_env += action_loss_per_env.detach().cpu().numpy()
        total_aux_loss_per_env += aux_loss_per_env.detach().cpu().numpy()
        total_steps += 1

        if step_i % 60 == 0:
            monitor_log = ""
            if mode == "train" and np.any(monitor_mask):
                monitor_metrics_step = compute_subset_progress_metrics(env, next_sim_state, monitor_mask)
                monitor_log = (
                    f", monitor_goal_mean={monitor_metrics_step['goal_idx']:.2f}/{len(env.task_spec.goals)}, "
                    f"monitor_kp_dist_mean={monitor_metrics_step['kp_dist']:.4f}"
                )
            _log(
                f"[distill step {step_i:5d}] mode={mode}, goal_mean={subset_mean(env.goal_idx.astype(np.float32), train_mask):.2f}/{len(env.task_spec.goals)}, "
                f"kp_dist_mean={subset_mean(next_sim_state.kp_dist.astype(np.float32), train_mask):.4f}, beta={beta:.3f}, num_envs={env.num_envs}{monitor_log}"
            )
        if finished:
            break

    if capture_viewer and run_dir is not None:
        _log_viewer_artifacts(
            run_dir=run_dir,
            mode=mode,
            frames=viewer_frames,
            video_frames=viewer_video_frames,
            wandb_run=wandb_run,
            viewer_key=capture_viewer_wandb_key,
            video_key=capture_viewer_video_wandb_key,
            video_fps=capture_viewer_video_fps,
            num_goals=len(env.task_spec.goals),
        )

    final_state = env.compute_sim_state()
    episode_wall_time_s = max(time.perf_counter() - episode_start_time, 1e-9)
    env_steps_per_s = env.num_envs * total_steps / episode_wall_time_s
    metrics = compute_subset_progress_metrics(env, final_state, train_mask)
    metrics.update(
        {
            "episode_steps": float(total_steps),
            "episode_wall_time_s": float(episode_wall_time_s),
            "env_steps_per_s": float(env_steps_per_s),
            "action_loss": subset_mean((total_action_loss_per_env / max(total_steps, 1)).astype(np.float32), train_mask),
            "aux_object_pos_loss": subset_mean((total_aux_loss_per_env / max(total_steps, 1)).astype(np.float32), train_mask),
            "beta": beta_scheduler.value(),
            "reset_object_z_low": float(env.reset_reason_counts["object_z_low"]),
            "reset_time_limit": float(env.reset_reason_counts["time_limit"]),
            "reset_max_goals": float(env.reset_reason_counts["max_goals"]),
            "reset_hand_far": float(env.reset_reason_counts["hand_far"]),
            "reset_dropped_after_lift": float(env.reset_reason_counts["dropped_after_lift"]),
        }
    )
    metrics["action_rmse"] = float(np.sqrt(max(metrics["action_loss"], 0.0)))
    metrics["aux_object_pos_rmse_m"] = float(np.sqrt(max(metrics["aux_object_pos_loss"], 0.0)))
    monitor_metrics = None
    if mode == "train" and np.any(monitor_mask):
        monitor_metrics = compute_subset_progress_metrics(env, final_state, monitor_mask)
        monitor_metrics.update(
            {
                "episode_steps": float(total_steps),
                "episode_wall_time_s": float(episode_wall_time_s),
                "env_steps_per_s": float(env_steps_per_s),
                "action_loss": subset_mean((total_action_loss_per_env / max(total_steps, 1)).astype(np.float32), monitor_mask),
                "aux_object_pos_loss": subset_mean((total_aux_loss_per_env / max(total_steps, 1)).astype(np.float32), monitor_mask),
                "beta": 0.0,
                "reset_object_z_low": float(env.reset_reason_counts["object_z_low"]),
                "reset_time_limit": float(env.reset_reason_counts["time_limit"]),
                "reset_max_goals": float(env.reset_reason_counts["max_goals"]),
                "reset_hand_far": float(env.reset_reason_counts["hand_far"]),
                "reset_dropped_after_lift": float(env.reset_reason_counts["dropped_after_lift"]),
            }
        )
        monitor_metrics["action_rmse"] = float(np.sqrt(max(monitor_metrics["action_loss"], 0.0)))
        monitor_metrics["aux_object_pos_rmse_m"] = float(
            np.sqrt(max(monitor_metrics["aux_object_pos_loss"], 0.0))
        )
    return metrics, monitor_metrics


def compute_distill_step(
    env: IsaacSimDistillEnv,
    teacher: RlPlayer,
    student,
    student_hidden,
    settings: DistillSettings,
    train: bool,
    run_dir: Path | None = None,
    iter_idx: int | None = None,
    debug_policy_image_stats: bool = False,
    debug_policy_image_stats_env_ids: list[int] | None = None,
):
    sim_state = env.compute_sim_state()
    teacher_obs = env.build_teacher_obs(sim_state)
    teacher_obs_tensor = torch.from_numpy(teacher_obs).float().to(env.device)
    with torch.no_grad():
        teacher_action = teacher.get_normalized_action(teacher_obs_tensor, deterministic_actions=True)

    if _args.student_input == "teacher_obs":
        if train:
            student_out, student_hidden = student(teacher_obs_tensor, student_hidden)
        else:
            with torch.no_grad():
                student_out, student_hidden = student(teacher_obs_tensor, student_hidden)
    else:
        student_obs = env.build_student_obs(sim_state, camera_modality=env.camera_modality)
        student_image = stack_student_image(student_obs, env.camera_modality, settings)
        if debug_policy_image_stats and run_dir is not None and iter_idx is not None:
            append_debug_policy_image_stats(
                run_dir=run_dir,
                iter_idx=iter_idx,
                env_ids=debug_policy_image_stats_env_ids or [0],
                student_obs=student_obs,
                student_image=student_image,
                modality=env.camera_modality,
            )
        if train:
            student_out, student_hidden = student(student_image, student_obs["proprio"], student_hidden)
        else:
            with torch.no_grad():
                student_out, student_hidden = student(student_image, student_obs["proprio"], student_hidden)

    student_action = student_out.action
    action_loss_per_env = torch.mean((student_action - teacher_action) ** 2, dim=1)
    aux_pred = student_out.aux.get("object_pos")
    gt_object_pos = torch.from_numpy(sim_state.object_pos_world_env_frame).float().to(env.device)
    aux_loss_per_env = (
        torch.mean((aux_pred - gt_object_pos) ** 2, dim=1)
        if aux_pred is not None
        else torch.zeros(env.num_envs, device=env.device)
    )
    total_loss = (
        settings.action_loss_weight * torch.mean(action_loss_per_env)
        + settings.aux_object_pos_weight * torch.mean(aux_loss_per_env)
    )
    return sim_state, student_action, total_loss, action_loss_per_env, aux_loss_per_env, student_hidden


def run_online_dagger(
    env: IsaacSimDistillEnv,
    teacher: RlPlayer,
    student,
    optimizer: torch.optim.Optimizer,
    settings: DistillSettings,
    run_dir: Path,
    wandb_run=None,
    capture_frames: bool = False,
    capture_frame_stride: int = 1,
    capture_viewer: bool = False,
    capture_viewer_len: int = 700,
    capture_viewer_interval: int | None = None,
    capture_viewer_env_id: int = 0,
    capture_viewer_video: bool = False,
    capture_viewer_wandb_key: str = "interactive_viewer",
    capture_viewer_video_wandb_key: str = "rollout_video",
    capture_viewer_video_fps: int = 60,
) -> float:
    teacher.reset()
    env.reset()
    student_hidden = student.initial_state(batch_size=env.num_envs, device=env.device)
    best_metric = -1.0
    interval_action_loss = np.zeros(env.num_envs, dtype=np.float64)
    interval_aux_loss = np.zeros(env.num_envs, dtype=np.float64)
    interval_start = time.perf_counter()
    accumulated_loss = None
    accumulated_steps = 0
    # Interactive viewer capture state machine, mirroring the env.py style:
    #   None        -> idle / not armed
    #   []          -> armed and capturing
    #   [frame, ...] -> actively accumulating a rollout window
    viewer_frames: list[dict] | None = [] if capture_viewer else None
    viewer_video_frames: list[np.ndarray] | None = [] if capture_viewer else None
    viewer_interval = None if capture_viewer_interval is None or capture_viewer_interval <= 0 else capture_viewer_interval
    debug_policy_image_stats_env_ids = parse_env_id_list(
        _args.debug_policy_image_stats_env_ids,
        env.num_envs,
    ) if _args.debug_policy_image_stats else []

    for iter_idx in range(settings.online_num_iters):
        sim_state, student_action, total_loss, action_loss_per_env, aux_loss_per_env, student_hidden = compute_distill_step(
            env=env,
            teacher=teacher,
            student=student,
            student_hidden=student_hidden,
            settings=settings,
            train=True,
            run_dir=run_dir,
            iter_idx=iter_idx + 1,
            debug_policy_image_stats=(
                _args.debug_policy_image_stats
                and ((iter_idx + 1) == 1 or (iter_idx + 1) % max(_args.debug_policy_image_stats_stride, 1) == 0)
            ),
            debug_policy_image_stats_env_ids=debug_policy_image_stats_env_ids,
        )
        accumulated_loss = total_loss if accumulated_loss is None else accumulated_loss + total_loss
        accumulated_steps += 1
        if accumulated_steps >= settings.online_update_interval:
            optimizer.zero_grad()
            (accumulated_loss / accumulated_steps).backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            student_hidden = detach_hidden_state(student_hidden)
            accumulated_loss = None
            accumulated_steps = 0

        targets = compute_joint_pos_targets(
            actions=student_action.detach().cpu().numpy(),
            prev_targets=env.prev_targets,
            hand_moving_average=settings.hand_moving_average,
            arm_moving_average=settings.arm_moving_average,
            hand_dof_speed_scale=settings.dof_speed_scale,
            dt=settings.control_dt,
        )
        env.apply_action(targets)
        env.step(render=not _args.headless)
        next_sim_state = env.compute_sim_state()
        if capture_frames and (iter_idx % max(capture_frame_stride, 1) == 0):
            save_camera_debug_step(env, run_dir, iter_idx)
        if capture_viewer and viewer_frames is None and viewer_interval is not None and (iter_idx + 1) % viewer_interval == 0:
            viewer_frames = []
            viewer_video_frames = []
        if capture_viewer and viewer_frames is not None:
            viewer_frames.append(env.capture_viewer_frame(capture_viewer_env_id, next_sim_state))
            if capture_viewer_video:
                rgb_frame = _capture_rgb_frame(env, capture_viewer_env_id)
                if rgb_frame is not None:
                    assert viewer_video_frames is not None
                    viewer_video_frames.append(rgb_frame)
            if len(viewer_frames) >= capture_viewer_len:
                iter_step = iter_idx + 1
                viewer_frames, viewer_video_frames = _finalize_online_viewer_capture(
                    run_dir=run_dir,
                    frames=viewer_frames,
                    video_frames=viewer_video_frames,
                    wandb_run=wandb_run,
                    viewer_key=capture_viewer_wandb_key,
                    video_key=capture_viewer_video_wandb_key,
                    video_fps=capture_viewer_video_fps,
                    num_goals=len(env.task_spec.goals),
                    step=iter_step,
                )
        env.maybe_advance_goal(next_sim_state)
        reset_env_ids = env.reset_done_envs(next_sim_state)
        if reset_env_ids.size > 0:
            student_hidden = reset_hidden_state(student_hidden, reset_env_ids)

        action_loss_np = action_loss_per_env.detach().cpu().numpy()
        aux_loss_np = aux_loss_per_env.detach().cpu().numpy()
        interval_action_loss += action_loss_np
        interval_aux_loss += aux_loss_np

        should_log = (iter_idx + 1) % settings.online_log_interval == 0 or iter_idx == settings.online_num_iters - 1
        if not should_log:
            continue

        interval_steps = settings.online_log_interval if (iter_idx + 1) % settings.online_log_interval == 0 else ((iter_idx + 1) % settings.online_log_interval)
        interval_steps = max(interval_steps, 1)
        final_state = env.compute_sim_state()
        interval_wall_time = max(time.perf_counter() - interval_start, 1e-9)
        metrics = env.compute_progress_metrics(final_state)
        metrics.update(
            {
                "episode_steps": float(iter_idx + 1),
                "episode_wall_time_s": float(interval_wall_time),
                "env_steps_per_s": float(env.num_envs * interval_steps / interval_wall_time),
                "action_loss": float(np.mean(interval_action_loss / interval_steps)),
                "aux_object_pos_loss": float(np.mean(interval_aux_loss / interval_steps)),
                "beta": 0.0,
                "reset_object_z_low": float(env.reset_reason_counts["object_z_low"]),
                "reset_time_limit": float(env.reset_reason_counts["time_limit"]),
                "reset_max_goals": float(env.reset_reason_counts["max_goals"]),
                "reset_hand_far": float(env.reset_reason_counts["hand_far"]),
                "reset_dropped_after_lift": float(env.reset_reason_counts["dropped_after_lift"]),
            }
        )
        metrics["action_rmse"] = float(np.sqrt(max(metrics["action_loss"], 0.0)))
        metrics["aux_object_pos_rmse_m"] = float(np.sqrt(max(metrics["aux_object_pos_loss"], 0.0)))
        _log(
            f"[online iter {iter_idx + 1}] goal_completion_ratio={metrics['goal_completion_ratio']:.3f}, "
            f"goal_idx={metrics['goal_idx']:.2f}, action_rmse={metrics['action_rmse']:.3f}, "
            f"pos_rmse_cm={100.0 * metrics['aux_object_pos_rmse_m']:.2f}, env_steps_s={metrics['env_steps_per_s']:.0f}"
        )
        record_metrics(
            run_dir,
            {"episode": iter_idx + 1, "mode": "train_online", **metrics},
            wandb_run=wandb_run,
            step=iter_idx + 1,
        )
        if capture_frames:
            save_camera_debug(env, run_dir)
        save_checkpoint(run_dir / "checkpoints" / "student_latest.pt", student, optimizer, iter_idx + 1, best_metric)
        selection_metric = metrics["goal_completion_ratio"]
        if selection_metric >= best_metric:
            best_metric = selection_metric
            save_checkpoint(run_dir / "checkpoints" / "student_best.pt", student, optimizer, iter_idx + 1, best_metric)
        if (iter_idx + 1) % max(settings.checkpoint_interval * settings.online_log_interval, 1) == 0:
            save_checkpoint(run_dir / "checkpoints" / f"student_iter_{iter_idx + 1}.pt", student, optimizer, iter_idx + 1, best_metric)
        interval_action_loss[:] = 0
        interval_aux_loss[:] = 0
        interval_start = time.perf_counter()

    if capture_viewer and viewer_frames is not None and len(viewer_frames) > 0:
        _finalize_online_viewer_capture(
            run_dir=run_dir,
            frames=viewer_frames,
            video_frames=viewer_video_frames,
            wandb_run=wandb_run,
            viewer_key=capture_viewer_wandb_key,
            video_key=capture_viewer_video_wandb_key,
            video_fps=capture_viewer_video_fps,
            num_goals=len(env.task_spec.goals),
            step=settings.online_num_iters,
        )

    return best_metric


def _make_grid(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("No images to grid")
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h, w = images[0].shape[:2]
    if images[0].ndim == 2:
        grid = np.zeros((rows * h, cols * w), dtype=images[0].dtype)
    else:
        grid = np.zeros((rows * h, cols * w, images[0].shape[2]), dtype=images[0].dtype)
    for idx, image in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w, ...] = image
    return grid


def save_camera_debug(env: IsaacSimDistillEnv, run_dir: Path):
    import imageio.v3 as iio
    import json

    if env.camera is None:
        return
    outputs = env.camera.data.output
    if outputs.get("rgb") is not None:
        rgb_batch = outputs["rgb"].cpu().numpy()[..., :3].astype(np.uint8)
        rgb_images = []
        for env_id, rgb in enumerate(rgb_batch):
            rgb_images.append(rgb)
            iio.imwrite(run_dir / "camera_debug" / f"env_{env_id}_rgb.png", rgb)
        iio.imwrite(run_dir / "camera_debug" / "rgb_grid.png", _make_grid(rgb_images))
    if outputs.get("distance_to_image_plane") is not None:
        depth_batch = outputs["distance_to_image_plane"].cpu().numpy()
        depth_images = []
        for env_id, depth in enumerate(depth_batch):
            if depth.ndim == 3:
                depth = depth[..., 0]
            finite_mask = np.isfinite(depth)
            if np.any(finite_mask):
                finite_depth = depth[finite_mask]
                depth_min = float(np.min(finite_depth))
                depth_max = float(np.max(finite_depth))
                denom = max(depth_max - depth_min, 1e-6)
                depth_vis = np.zeros_like(depth, dtype=np.float32)
                depth_vis[finite_mask] = np.clip((depth[finite_mask] - depth_min) / denom, 0.0, 1.0)
            else:
                depth_vis = np.zeros_like(depth, dtype=np.float32)
            depth_img = (depth_vis * 255).astype(np.uint8)
            depth_images.append(depth_img)
            iio.imwrite(run_dir / "camera_debug" / f"env_{env_id}_depth.png", depth_img)
        iio.imwrite(run_dir / "camera_debug" / "depth_grid.png", _make_grid(depth_images))

    with open(run_dir / "camera_debug" / "metadata.json", "w") as f:
        json.dump(
            {
                "num_envs": env.num_envs,
                "camera_convention": env.camera_pose.convention,
                "camera_pose_frame": "world_pose_via_set_world_poses",
                "camera_pose": {
                    "pos": list(env.camera_pose.pos),
                    "quat_wxyz": list(env.camera_pose.quat_wxyz),
                    "mount": env.camera_pose.mount,
                    "link_name": env.camera_pose.link_name,
                },
                "camera_intrinsics": {
                    "width": env.camera_intrinsics.width,
                    "height": env.camera_intrinsics.height,
                    "focal_length": env.camera_intrinsics.focal_length,
                    "horizontal_aperture": env.camera_intrinsics.horizontal_aperture,
                    "focus_distance": env.camera_intrinsics.focus_distance,
                    "clipping_range": list(env.camera_intrinsics.clipping_range),
                },
                "raw_T_W_C": default_real_camera_transform().tolist(),
                "resolved_camera_world_pos_w": env.camera.data.pos_w.detach().cpu().numpy().tolist(),
                "resolved_camera_world_quat_w_ros": env.camera.data.quat_w_ros.detach().cpu().numpy().tolist(),
                "env_origins": env.env_origins.detach().cpu().numpy().tolist(),
                "aux_object_pos_frame": "env_local_world",
                "object_start_mode": env.object_start_mode,
                "current_start_pose": env.current_start_pose.tolist(),
            },
            f,
            indent=2,
        )


def save_camera_debug_step(env: IsaacSimDistillEnv, run_dir: Path, step_i: int):
    import imageio.v3 as iio

    if env.camera is None:
        return
    outputs = env.camera.data.output
    grids_dir = run_dir / "camera_debug" / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)
    if outputs.get("rgb") is not None:
        rgb_batch = outputs["rgb"].cpu().numpy()[..., :3].astype(np.uint8)
        rgb_images = []
        for env_id, rgb in enumerate(rgb_batch):
            rgb_images.append(rgb)
            env_dir = run_dir / "camera_debug" / f"env_{env_id}" / "rgb"
            env_dir.mkdir(parents=True, exist_ok=True)
            iio.imwrite(env_dir / f"frame_{step_i:04d}.png", rgb)
        iio.imwrite(grids_dir / f"rgb_{step_i:04d}.png", _make_grid(rgb_images))
    if outputs.get("distance_to_image_plane") is not None:
        depth_batch = outputs["distance_to_image_plane"].cpu().numpy()
        depth_images = []
        for env_id, depth in enumerate(depth_batch):
            if depth.ndim == 3:
                depth = depth[..., 0]
            finite_mask = np.isfinite(depth)
            depth_vis = np.zeros_like(depth, dtype=np.float32)
            if np.any(finite_mask):
                finite_depth = depth[finite_mask]
                depth_min = float(np.min(finite_depth))
                depth_max = float(np.max(finite_depth))
                denom = max(depth_max - depth_min, 1e-6)
                depth_vis[finite_mask] = np.clip((depth[finite_mask] - depth_min) / denom, 0.0, 1.0)
            depth_img = (depth_vis * 255).astype(np.uint8)
            depth_images.append(depth_img)
            env_dir = run_dir / "camera_debug" / f"env_{env_id}" / "depth"
            env_dir.mkdir(parents=True, exist_ok=True)
            iio.imwrite(env_dir / f"frame_{step_i:04d}.png", depth_img)
        iio.imwrite(grids_dir / f"depth_{step_i:04d}.png", _make_grid(depth_images))


def update_monitor_history(
    monitor_history: dict[str, list[float]],
    monitor_metrics: dict[str, float] | None,
    window: int,
) -> dict[str, float]:
    if monitor_metrics is None:
        return {}
    rolling_metrics = {}
    for key in ("goal_completion_ratio", "goal_idx", "kp_dist", "action_loss", "aux_object_pos_loss"):
        history = monitor_history.setdefault(key, [])
        history.append(float(monitor_metrics[key]))
        recent = history[-window:]
        rolling_metrics[f"{key}_window"] = float(np.mean(recent))
    rolling_metrics["action_rmse_window"] = float(np.sqrt(max(rolling_metrics["action_loss_window"], 0.0)))
    rolling_metrics["aux_object_pos_rmse_m_window"] = float(
        np.sqrt(max(rolling_metrics["aux_object_pos_loss_window"], 0.0))
    )
    return rolling_metrics


def main():
    args = _args
    app = _app
    repo_root = Path(__file__).resolve().parent.parent
    teacher_config = Path(resolve_repo_path(repo_root, args.teacher_config))
    teacher_checkpoint = resolve_repo_path(repo_root, args.teacher_checkpoint)
    settings = load_distill_settings(Path(resolve_repo_path(repo_root, args.distill_config)), args)
    validate_distill_settings(settings)
    camera_modality, camera_pose, camera_intrinsics = load_camera_pose(Path(resolve_repo_path(repo_root, args.camera_config)))
    if args.student_modality:
        camera_modality = args.student_modality
    enable_camera = (
        args.mode == "camera_debug"
        or args.capture_frames
        or args.capture_viewer_video
        or args.student_input == "camera"
    )

    task_spec = load_task_spec(
        repo_root=repo_root,
        task_source=args.task_source,
        assembly=args.assembly,
        part_id=args.part_id,
        collision_method=args.collision_method,
        object_category=args.object_category,
        object_name=args.object_name,
        task_name=args.task_name,
        teacher_config_path=teacher_config,
    )
    env = IsaacSimDistillEnv(
        task_spec=task_spec,
        app=app,
        headless=args.headless,
        camera_modality=camera_modality,
        camera_pose_override=camera_pose,
        camera_intrinsics=camera_intrinsics,
        enable_camera=enable_camera,
        num_envs=settings.num_envs,
        env_spacing=settings.env_spacing,
        ground_plane_size=settings.ground_plane_size,
        object_start_mode=settings.object_start_mode,
        object_pos_noise_xyz=settings.object_pos_noise_xyz,
        object_yaw_noise_deg=settings.object_yaw_noise_deg,
        camera_backend=args.camera_backend or settings.camera_backend,
        depth_preprocess_mode=settings.depth_preprocess_mode,
        depth_min_m=settings.depth_min_m,
        depth_max_m=settings.depth_max_m,
        episode_length=settings.episode_length,
        reset_when_dropped=settings.reset_when_dropped,
    )
    teacher_inference_batch_size = 128 if args.mode == "teacher_eval" else 32
    teacher = RlPlayer(
        num_observations=140,
        num_actions=29,
        config_path=str(teacher_config),
        checkpoint_path=teacher_checkpoint,
        device=str(env.device),
        num_envs=settings.num_envs,
        inference_batch_size=min(settings.num_envs, teacher_inference_batch_size),
    )
    student = None if args.mode == "teacher_eval" else build_student(args, env, settings)
    optimizer = None if student is None else torch.optim.Adam(student.parameters(), lr=settings.learning_rate)
    if args.student_checkpoint and student is not None and optimizer is not None:
        load_student_checkpoint(resolve_repo_path(repo_root, args.student_checkpoint), student, optimizer)
    run_dir = ensure_run_dir(repo_root, args, settings)
    beta_scheduler = BetaScheduler(settings)
    wandb_run = init_wandb(args, run_dir, settings)

    _log(f"Teacher checkpoint: {teacher_checkpoint}")
    _log(f"Teacher config: {teacher_config}")
    _log(f"Run dir: {run_dir}")
    _log(f"Student arch/modality: {args.student_arch}/{camera_modality}")
    _log(f"Cameras enabled: {enable_camera}")
    _log(
        f"Parallel env config: num_envs={settings.num_envs}, env_spacing={settings.env_spacing}, "
        f"object_start_mode={settings.object_start_mode}"
    )

    if args.mode == "camera_debug":
        env.reset()
        env.step(render=not args.headless)
        save_camera_debug(env, run_dir)
        _log(f"Saved camera debug outputs under {run_dir / 'camera_debug'}")
        if wandb_run is not None:
            wandb_run.finish()
        force_exit(0)

    if args.mode == "train_online":
        if student is None or optimizer is None:
            raise RuntimeError("train_online requires a student and optimizer")
        _log(
            f"Online DAgger config: iters={settings.online_num_iters}, "
            f"log_interval={settings.online_log_interval}, update_interval={settings.online_update_interval}, beta=0.0"
        )
        best_metric = run_online_dagger(
            env=env,
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            settings=settings,
            run_dir=run_dir,
            wandb_run=wandb_run,
            capture_frames=args.capture_frames,
            capture_frame_stride=args.capture_frame_stride,
            capture_viewer=args.capture_viewer,
            capture_viewer_len=args.capture_viewer_len,
            capture_viewer_interval=(
                args.capture_viewer_interval if args.capture_viewer_interval is not None else settings.online_log_interval
            ),
            capture_viewer_env_id=args.capture_viewer_env_id,
            capture_viewer_video=args.capture_viewer_video,
            capture_viewer_wandb_key=args.capture_viewer_wandb_key,
            capture_viewer_video_wandb_key=args.capture_viewer_video_wandb_key,
            capture_viewer_video_fps=args.capture_viewer_video_fps,
        )
        _log(f"Online DAgger complete. best_goal_completion_ratio={best_metric:.3f}")
        if wandb_run is not None:
            wandb_run.finish()
        force_exit(0)

    best_metric = -1.0
    monitor_history: dict[str, list[float]] = {}
    if args.mode == "train" and settings.num_envs < 2048:
        teacher_metrics, _ = run_episode(
            "teacher_eval",
            env,
            teacher,
            student,
            None,
            settings,
            beta_scheduler,
            run_dir=None,
            capture_frames=False,
            capture_frame_stride=args.capture_frame_stride,
        )
        _log(
            f"[teacher baseline] goal_completion_ratio={teacher_metrics['goal_completion_ratio']:.3f}, "
            f"goal_idx={teacher_metrics['goal_idx']:.0f}, kp_dist={teacher_metrics['kp_dist']:.4f}"
        )
        record_metrics(
            run_dir,
            {"episode": -1, "mode": "teacher_eval_baseline", **teacher_metrics},
            wandb_run=wandb_run,
            step=0,
        )
    elif args.mode == "train":
        _log("Skipping in-run teacher baseline for large batch training; use standalone teacher_eval reference instead")

    for episode in range(settings.num_episodes if args.mode == "train" else 1):
        metrics, monitor_metrics = run_episode(
            args.mode,
            env,
            teacher,
            student,
            optimizer if args.mode == "train" else None,
            settings,
            beta_scheduler,
            run_dir=run_dir,
            capture_frames=args.capture_frames,
            capture_frame_stride=args.capture_frame_stride,
            capture_viewer=args.capture_viewer,
            capture_viewer_len=args.capture_viewer_len,
            capture_viewer_env_id=args.capture_viewer_env_id,
            capture_viewer_video=args.capture_viewer_video,
            wandb_run=wandb_run,
            capture_viewer_wandb_key=args.capture_viewer_wandb_key,
            capture_viewer_video_wandb_key=args.capture_viewer_video_wandb_key,
            capture_viewer_video_fps=args.capture_viewer_video_fps,
        )
        _log(
            f"[episode {episode}] mode={args.mode}, goal_completion_ratio={metrics['goal_completion_ratio']:.3f}, "
            f"goal_idx={metrics['goal_idx']:.0f}, action_loss={metrics['action_loss']:.6f}, "
            f"aux_object_pos_loss={metrics['aux_object_pos_loss']:.6f}, beta={metrics['beta']:.3f}"
        )
        record_metrics(
            run_dir,
            {"episode": episode, "mode": args.mode, **metrics},
            wandb_run=wandb_run,
            step=episode + 1,
        )
        if monitor_metrics is not None:
            monitor_row = {"episode": episode, "mode": "student_monitor", **monitor_metrics}
            monitor_row.update(update_monitor_history(monitor_history, monitor_metrics, settings.monitor_log_window))
            _log(
                f"[episode {episode}] mode=student_monitor, goal_completion_ratio={monitor_metrics['goal_completion_ratio']:.3f}, "
                f"goal_idx={monitor_metrics['goal_idx']:.0f}, action_loss={monitor_metrics['action_loss']:.6f}, "
                f"aux_object_pos_loss={monitor_metrics['aux_object_pos_loss']:.6f}, beta=0.000"
            )
            record_metrics(
                run_dir,
                monitor_row,
                wandb_run=wandb_run,
                step=episode + 1,
            )
        save_camera_debug(env, run_dir)
        if args.mode == "train":
            beta_scheduler.update(metrics["goal_completion_ratio"])
            selection_metric = metrics["goal_completion_ratio"]
            save_checkpoint(run_dir / "checkpoints" / "student_latest.pt", student, optimizer, episode, best_metric)
            if selection_metric >= best_metric:
                best_metric = selection_metric
                save_checkpoint(run_dir / "checkpoints" / "student_best.pt", student, optimizer, episode, best_metric)
            if (episode + 1) % settings.checkpoint_interval == 0:
                save_checkpoint(run_dir / "checkpoints" / f"student_step_{episode + 1}.pt", student, optimizer, episode, best_metric)

    if wandb_run is not None:
        wandb_run.finish()
    force_exit(0)


if __name__ == "__main__":
    main()
