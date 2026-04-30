"""Online BC/DAgger-style depth student distillation for Isaac Lab PegInHole.

This is intentionally separate from the older ``isaacsim_conversion`` trainer:
it uses Kushal's Isaac Lab environment directly, asks the env for
``get_student_obs()``, and uses a restored rl_games teacher only to produce
action labels on the student rollouts.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_DIR = Path("/juno/u/kedia/depthbasedRL/train_dir/Apr28/isaacSim_PegInHole")


def _parse_optional_pair(value: str | None) -> tuple[int, int] | None:
    if value is None or value.lower() in {"none", "null"}:
        return None
    parts = value.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"Expected two ints, got {value!r}")
    return int(parts[0]), int(parts[1])


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_env_cfg(task: str, teacher_config: Path | None, num_envs: int, sim_device: str):
    import gymnasium as gym
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    from isaacsimenvs.utils.config_utils import apply_env_cfg_dict

    spec = gym.spec(task)
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")

    task_yaml = spec.kwargs.get("env_cfg_yaml_entry_point")
    if task_yaml:
        apply_env_cfg_dict(env_cfg, _load_yaml(Path(task_yaml)))

    if teacher_config is not None:
        teacher_env = _load_yaml(teacher_config).get("env", {})
        sections = (
            "sim",
            "scene",
            "assets",
            "obs",
            "action",
            "reward",
            "reset",
            "termination",
            "domain_randomization",
            "peg_in_hole",
        )
        apply_env_cfg_dict(env_cfg, {key: teacher_env[key] for key in sections if key in teacher_env})
        # Re-apply the student-camera overlay after matching teacher dynamics.
        if task_yaml:
            task_overlay = _load_yaml(Path(task_yaml))
            student_overlay = {"student_obs": task_overlay.get("student_obs", {})}
            if "sim" in task_overlay and "render" in task_overlay["sim"]:
                student_overlay["sim"] = {"render": task_overlay["sim"]["render"]}
            apply_env_cfg_dict(env_cfg, student_overlay)

    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.sim.device = sim_device
    return env_cfg


def _load_teacher_player(env, task: str, agent: str, checkpoint: Path, rl_device: str):
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rl_games.torch_runner import Runner

    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env

    agent_cfg = load_cfg_from_registry(task, agent)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    agent_cfg["params"]["config"]["device"] = rl_device
    agent_cfg["params"]["config"]["device_name"] = rl_device

    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(env, rl_device=rl_device, clip_obs=clip_obs, clip_actions=clip_actions)

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.has_batch_dimension = True
    player.reset()
    return wrapped, player


def _student_image_channels(modality: str) -> int:
    if modality == "depth":
        return 1
    if modality == "rgb":
        return 3
    if modality == "rgbd":
        return 4
    raise ValueError(f"Unsupported student image modality: {modality!r}")


def _init_wandb(args, run_dir: Path):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group or None,
        name=args.wandb_name or run_dir.name,
        dir=str(run_dir),
        config=vars(args),
    )


def _log_metrics(run_dir: Path, row: dict, wandb_run=None, step: int | None = None) -> None:
    path = run_dir / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row))
        if write_header:
            writer.writeheader()
        writer.writerow({key: row[key] for key in sorted(row)})
    if wandb_run is not None:
        wandb_run.log({key: value for key, value in row.items() if isinstance(value, (int, float))}, step=step)


def _save_checkpoint(path: Path, student, optimizer, step: int, best_metric: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "best_metric": float(best_metric),
        },
        path,
    )


def _load_student_checkpoint(path: Path | None, student, optimizer=None, *, strict: bool = True) -> tuple[int, float]:
    if path is None:
        return 0, -1.0
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("student", ckpt)
    incompatible = student.load_state_dict(state, strict=strict)
    if not strict:
        print(
            "[distill_depth] non-strict student checkpoint load: "
            f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}",
            flush=True,
        )
    if optimizer is not None and "optimizer" in ckpt and strict:
        optimizer.load_state_dict(ckpt["optimizer"])
    elif optimizer is not None and "optimizer" in ckpt and not strict:
        print("[distill_depth] skipped optimizer restore because checkpoint load is non-strict", flush=True)
    return int(ckpt.get("step", 0)), float(ckpt.get("best_metric", -1.0))


def _object_pos_env_frame(env) -> torch.Tensor:
    return env.object.data.root_pos_w - env.scene.env_origins


def _object_keypoint_offsets(env) -> torch.Tensor:
    return env._keypoint_offsets * env._object_scale_multiplier.unsqueeze(1)


def _object_keypoints_env_frame(env) -> torch.Tensor:
    from isaaclab.utils.math import quat_apply

    pos = _object_pos_env_frame(env)
    quat = env.object.data.root_quat_w
    offsets = _object_keypoint_offsets(env)
    n_envs, n_keypoints, _ = offsets.shape
    rot = quat.unsqueeze(1).expand(-1, n_keypoints, -1).reshape(-1, 4)
    return pos.unsqueeze(1) + quat_apply(rot, offsets.reshape(-1, 3)).reshape(n_envs, n_keypoints, 3)


def _safe_normalize(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def _orthogonal_fallback(unit_vec: torch.Tensor) -> torch.Tensor:
    x_axis = torch.zeros_like(unit_vec)
    y_axis = torch.zeros_like(unit_vec)
    x_axis[..., 0] = 1.0
    y_axis[..., 1] = 1.0
    axis = torch.where(unit_vec[..., 0:1].abs() < 0.9, x_axis, y_axis)
    return axis - (axis * unit_vec).sum(dim=-1, keepdim=True) * unit_vec


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert Zhou-style 6D rotations to valid rotation matrices.

    The rot6d head is initialized near identity, but this also handles early
    near-collinear predictions without producing NaNs.
    """

    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    x_axis = torch.zeros_like(a1)
    x_axis[..., 0] = 1.0
    a1 = torch.where(a1.norm(dim=-1, keepdim=True) < 1e-5, x_axis, a1)
    b1 = _safe_normalize(a1)
    a2_orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    degenerate = a2_orthogonal.norm(dim=-1, keepdim=True) < 1e-5
    a2_orthogonal = torch.where(degenerate, _orthogonal_fallback(b1), a2_orthogonal)
    b2 = _safe_normalize(a2_orthogonal)
    b3 = _safe_normalize(torch.cross(b1, b2, dim=-1))
    b2 = torch.cross(b3, b1, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _transform_keypoints(pos: torch.Tensor, rot_matrix: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    return pos.unsqueeze(1) + torch.matmul(offsets, rot_matrix.transpose(-1, -2))


def _matrix_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to normalized wxyz quaternions for visualization."""

    m = matrix
    quat = torch.empty((*m.shape[:-2], 4), device=m.device, dtype=m.dtype)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]

    mask = trace > 0.0
    if mask.any():
        s = torch.sqrt(trace[mask] + 1.0).clamp_min(1e-8) * 2.0
        quat[mask, 0] = 0.25 * s
        quat[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
        quat[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
        quat[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s

    mask_x = (~mask) & (m[..., 0, 0] > m[..., 1, 1]) & (m[..., 0, 0] > m[..., 2, 2])
    if mask_x.any():
        s = torch.sqrt(1.0 + m[mask_x, 0, 0] - m[mask_x, 1, 1] - m[mask_x, 2, 2]).clamp_min(1e-8) * 2.0
        quat[mask_x, 0] = (m[mask_x, 2, 1] - m[mask_x, 1, 2]) / s
        quat[mask_x, 1] = 0.25 * s
        quat[mask_x, 2] = (m[mask_x, 0, 1] + m[mask_x, 1, 0]) / s
        quat[mask_x, 3] = (m[mask_x, 0, 2] + m[mask_x, 2, 0]) / s

    mask_y = (~mask) & (~mask_x) & (m[..., 1, 1] > m[..., 2, 2])
    if mask_y.any():
        s = torch.sqrt(1.0 + m[mask_y, 1, 1] - m[mask_y, 0, 0] - m[mask_y, 2, 2]).clamp_min(1e-8) * 2.0
        quat[mask_y, 0] = (m[mask_y, 0, 2] - m[mask_y, 2, 0]) / s
        quat[mask_y, 1] = (m[mask_y, 0, 1] + m[mask_y, 1, 0]) / s
        quat[mask_y, 2] = 0.25 * s
        quat[mask_y, 3] = (m[mask_y, 1, 2] + m[mask_y, 2, 1]) / s

    mask_z = (~mask) & (~mask_x) & (~mask_y)
    if mask_z.any():
        s = torch.sqrt(1.0 + m[mask_z, 2, 2] - m[mask_z, 0, 0] - m[mask_z, 1, 1]).clamp_min(1e-8) * 2.0
        quat[mask_z, 0] = (m[mask_z, 1, 0] - m[mask_z, 0, 1]) / s
        quat[mask_z, 1] = (m[mask_z, 0, 2] + m[mask_z, 2, 0]) / s
        quat[mask_z, 2] = (m[mask_z, 1, 2] + m[mask_z, 2, 1]) / s
        quat[mask_z, 3] = 0.25 * s

    return _safe_normalize(quat)


def _aux_head_dims(aux_pose_mode: str) -> dict[str, int]:
    if aux_pose_mode == "none":
        return {}
    if aux_pose_mode == "position":
        return {"object_pos": 3}
    if aux_pose_mode == "rot6d_keypoints":
        return {"object_pos": 3, "object_rot6d": 6}
    raise ValueError(f"Unsupported aux_pose_mode: {aux_pose_mode!r}")


def _initialize_rot6d_head(student) -> None:
    aux_heads = getattr(student, "aux_heads", None)
    if aux_heads is None or "object_rot6d" not in aux_heads:
        return
    head = aux_heads["object_rot6d"]
    final = head[-1]
    with torch.no_grad():
        final.weight.mul_(0.01)
        final.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=final.bias.device))


def _compute_aux_losses(env, aux: dict[str, torch.Tensor], args) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = env.num_envs
    zero = torch.zeros(batch, device=env.device)
    if args.aux_pose_mode == "none":
        return zero, {
            "pos_loss": zero,
            "keypoint_loss": zero,
            "pred_pose_wxyz": None,
        }

    target_pos = _object_pos_env_frame(env)
    pred_pos = aux["object_pos"]
    pos_loss = F.mse_loss(pred_pos, target_pos, reduction="none").mean(dim=1)
    aux_loss = args.aux_object_pos_weight * pos_loss
    pred_quat_wxyz = env.object.data.root_quat_w
    keypoint_loss = zero

    if args.aux_pose_mode == "rot6d_keypoints":
        rot_matrix = _rot6d_to_matrix(aux["object_rot6d"])
        offsets = _object_keypoint_offsets(env)
        pred_keypoints = _transform_keypoints(pred_pos, rot_matrix, offsets)
        target_keypoints = _object_keypoints_env_frame(env)
        keypoint_loss = F.mse_loss(pred_keypoints, target_keypoints, reduction="none").mean(dim=(1, 2))
        aux_loss = aux_loss + args.aux_object_keypoint_weight * keypoint_loss
        pred_quat_wxyz = _matrix_to_quat_wxyz(rot_matrix)
    elif args.aux_pose_mode != "position":
        raise ValueError(f"Unsupported aux_pose_mode: {args.aux_pose_mode!r}")

    pred_pose_wxyz = torch.cat([pred_pos, pred_quat_wxyz], dim=-1)
    return aux_loss, {
        "pos_loss": pos_loss,
        "keypoint_loss": keypoint_loss,
        "pred_pose_wxyz": pred_pose_wxyz,
    }


def _teacher_obs_tensor(obs) -> torch.Tensor:
    """Extract the policy observation tensor passed through the rl_games wrapper."""
    if isinstance(obs, torch.Tensor):
        return obs.float()
    if isinstance(obs, dict):
        for key in ("obs", "policy", "observations"):
            if key in obs:
                return _teacher_obs_tensor(obs[key])
    raise TypeError(
        "Expected rl_games observation to be a tensor or dict containing an "
        f"'obs' tensor, got {type(obs).__name__}."
    )


def _reset_hidden_for_done(hidden: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    done = dones.reshape(-1).bool()
    if not done.any():
        return hidden
    hidden = hidden.clone()
    hidden[done] = 0.0
    return hidden


def _done_success_stats(env, dones: torch.Tensor) -> tuple[float, float, int]:
    done = dones.reshape(-1).bool()
    count = int(done.sum().item())
    if count == 0:
        return 0.0, 0.0, 0
    successes = env._prev_episode_successes[done].float()
    max_goals = env.prev_episode_env_max_goals[done].clamp_min(1).float()
    return float(successes.mean().item()), float((successes / max_goals).mean().item()), count


def _capture_viewer_if_needed(
    *,
    env,
    frames: list,
    output_dir: Path,
    capture_len: int,
    step: int,
    predicted_object_pose_wxyz: torch.Tensor | None = None,
    append_frame: bool = True,
    wandb_run=None,
    wandb_key: str = "interactive_viewer",
) -> list:
    from isaacsimenvs.tasks.simtoolreal.pose_viewer import (
        build_pose_viewer_html,
        capture_pose_viewer_frame,
        object_urdf_text_for_env,
        table_urdf_text_for_env,
    )

    if append_frame:
        frames.append(capture_pose_viewer_frame(env, 0, predicted_object_pose_wxyz=predicted_object_pose_wxyz))
    if len(frames) < capture_len:
        return frames

    output_dir.mkdir(parents=True, exist_ok=True)
    html = build_pose_viewer_html(
        frames=frames,
        object_urdf_text=object_urdf_text_for_env(env, 0),
        table_urdf_text=table_urdf_text_for_env(env, 0),
        github_raw_base=None,
        url_check="skip",
    )
    path = output_dir / f"pose_viewer_step_{step:08d}.html"
    path.write_text(html, encoding="utf-8")
    print(f"[distill_depth] wrote viewer HTML: {path}", flush=True)
    if wandb_run is not None:
        import wandb

        wandb_run.log({wandb_key: wandb.Html(path.read_text(encoding="utf-8"))}, step=step)
    return []


def _log_depth_debug_media(
    *,
    paths: dict[str, Path],
    wandb_run,
    step: int,
    frames: list,
    max_frames: int,
    fps: int,
    log_video: bool,
    key_prefix: str,
) -> list:
    if wandb_run is None:
        return frames

    import numpy as np
    import wandb
    from PIL import Image

    media = {}
    for name in (
        "review_grid_png",
        "raw_window_png",
        "noisy_window_png",
        "policy_full_depth_png",
        "policy_depth_png",
    ):
        path = paths.get(name)
        if path is not None and path.exists():
            media[f"{key_prefix}/{name.removesuffix('_png')}"] = wandb.Image(str(path))

    review_path = paths.get("review_grid_png")
    if review_path is not None and review_path.exists() and max_frames > 0:
        frame = np.asarray(Image.open(review_path).convert("RGB"))
        frames.append(frame)
        if len(frames) > max_frames:
            frames = frames[-max_frames:]
        if log_video and len(frames) >= 2:
            video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
            media[f"{key_prefix}/rolling_video"] = wandb.Video(
                video,
                fps=max(1, int(fps)),
                format="mp4",
            )

    if media:
        wandb_run.log(media, step=step)
    return frames


def _window_metric_depth(depth: torch.Tensor, *, env_ids: list[int], near: float, far: float) -> torch.Tensor:
    from isaacsimenvs.distillation.depth_debug import depth_tensor_to_nchw

    image = depth_tensor_to_nchw(depth).detach().float()[env_ids, 0]
    image = torch.nan_to_num((image - near) / max(far - near, 1e-6), nan=0.0, posinf=1.0, neginf=0.0)
    return image.clamp_(0.0, 1.0)


def _normalized_policy_depth(depth: torch.Tensor, *, env_ids: list[int]) -> torch.Tensor:
    from isaacsimenvs.distillation.depth_debug import depth_tensor_to_nchw

    image = depth_tensor_to_nchw(depth).detach().float()[env_ids, 0]
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    return image.clamp_(0.0, 1.0)


def _capture_depth_rollout_frame(
    *,
    env,
    policy_depth: torch.Tensor,
    env_ids: list[int],
    near: float,
    far: float,
) -> "np.ndarray | None":
    """Capture one contiguous policy-input review frame for realtime video logging."""

    if policy_depth.ndim == 4 and policy_depth.shape[1] != 1 and policy_depth.shape[-1] != 1:
        return None

    import numpy as np
    from isaacsimenvs.distillation.depth_debug import _make_review_grid

    with torch.no_grad():
        raw_window = _window_metric_depth(
            env.student_camera.data.output["distance_to_image_plane"],
            env_ids=env_ids,
            near=near,
            far=far,
        )
        noisy_depth = getattr(env, "_student_depth_noisy_m", None)
        noisy_window = None
        if noisy_depth is not None:
            noisy_window = _window_metric_depth(noisy_depth, env_ids=env_ids, near=near, far=far)
        policy_full_depth = getattr(env, "_student_depth_policy_full", None)
        policy_full = None
        if policy_full_depth is not None:
            policy_full = _normalized_policy_depth(policy_full_depth, env_ids=env_ids)
        policy = _normalized_policy_depth(policy_depth, env_ids=env_ids)

    return _make_review_grid(
        env_ids=env_ids,
        raw_window=raw_window.cpu().numpy(),
        noisy_window=None if noisy_window is None else noisy_window.cpu().numpy(),
        policy_full=None if policy_full is None else policy_full.cpu().numpy(),
        policy=policy.cpu().numpy(),
    )


def _log_depth_rollout_video(
    *,
    frame,
    frames: list,
    wandb_run,
    step: int,
    key: str,
    fps: int,
    max_frames: int,
    interval: int,
) -> list:
    if frame is None or max_frames <= 0:
        return frames
    frames.append(frame)
    if len(frames) > max_frames:
        frames = frames[-max_frames:]
    if wandb_run is not None and len(frames) >= 2 and step % max(1, interval) == 0:
        import numpy as np
        import wandb

        video = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
        wandb_run.log({key: wandb.Video(video, fps=max(1, int(fps)), format="mp4")}, step=step)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("teacher_eval", "student_eval", "train_online"), default="teacher_eval")
    parser.add_argument("--task", default="Isaacsimenvs-PegInHoleDepthStudent-Direct-v0")
    parser.add_argument("--teacher_agent", default="rl_games_sapg_cfg_entry_point")
    parser.add_argument("--teacher_checkpoint", type=Path, default=DEFAULT_TEACHER_DIR / "model.pth")
    parser.add_argument("--teacher_config", type=Path, default=DEFAULT_TEACHER_DIR / "config.yaml")
    parser.add_argument("--student_checkpoint", type=Path, default=None)
    parser.add_argument("--student_checkpoint_strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--student_input", choices=("camera", "teacher_obs"), default="camera")
    parser.add_argument("--student_arch", choices=("mono_transformer_recurrent", "mlp_recurrent"), default=None)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_iters", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--aux_pose_mode",
        choices=("none", "position", "rot6d_keypoints"),
        default="position",
        help="Auxiliary object pose target. 'position' preserves the old object_pos-only loss.",
    )
    parser.add_argument("--aux_object_pos_weight", type=float, default=1.0)
    parser.add_argument("--aux_object_keypoint_weight", type=float, default=1.0)
    parser.add_argument("--deterministic_teacher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--force_scene_tol_combo", default=None)
    parser.add_argument("--force_peg_idx", type=int, default=None)
    parser.add_argument("--depth_noise_profile", choices=("off", "weak", "medium", "strong", "custom"), default=None)
    parser.add_argument("--depth_noise_strength", type=float, default=None)
    parser.add_argument(
        "--camera_pose_randomization_profile",
        choices=("off", "weak", "medium", "strong", "custom"),
        default=None,
    )
    parser.add_argument("--camera_pose_randomization_mode", choices=("startup", "reset"), default=None)
    parser.add_argument(
        "--camera_pos_noise_m",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Custom per-axis camera position noise half-widths in meters.",
    )
    parser.add_argument(
        "--camera_rot_noise_deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Custom per-axis camera RPY noise half-widths in degrees.",
    )
    parser.add_argument("--peg_object_init_orientation_mode", choices=("scene", "yaw_only", "full"), default=None)
    parser.add_argument("--peg_object_init_position_noise_xy", type=float, nargs=2, default=None)
    parser.add_argument("--peg_object_init_position_noise_z", type=float, default=None)
    parser.add_argument("--capture_viewer", action="store_true")
    parser.add_argument("--capture_viewer_len", type=int, default=600)
    parser.add_argument("--depth_debug_interval", type=int, default=0)
    parser.add_argument("--depth_debug_env_ids", default="0,1,2,3")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="depthbasedRL-isaacsim-distill")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_viewer_key", default="interactive_viewer")
    parser.add_argument("--wandb_depth_media_key", default="student_depth_debug")
    parser.add_argument("--wandb_log_depth_media", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_depth_video_max_frames", type=int, default=30)
    parser.add_argument("--wandb_depth_video_fps", type=int, default=4)
    parser.add_argument("--wandb_depth_video_interval", type=int, default=5000)
    parser.add_argument("--wandb_depth_rollout_video_key", default="student_depth_debug/realtime_rollout_video")
    parser.add_argument("--wandb_depth_rollout_video_env_ids", default="0")
    parser.add_argument("--wandb_depth_rollout_video_len", type=int, default=600)
    parser.add_argument("--wandb_depth_rollout_video_fps", type=int, default=60)
    parser.add_argument("--wandb_depth_rollout_video_interval", type=int, default=600)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    # Camera jobs need Isaac Lab's camera pipeline enabled. Teacher-observation
    # jobs do not, so keep them on the cheaper non-rendering path.
    args.enable_cameras = args.student_input == "camera"
    app = AppLauncher(args).app

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.distillation.depth_debug import save_depth_debug
    from isaacsimenvs.distillation.student_policy import MLPRecurrentPolicy, MonoTransformerRecurrentPolicy

    run_dir = args.run_dir or REPO_ROOT / "distillation_runs" / f"kushal_depth_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    print(f"[distill_depth] run_dir={run_dir}", flush=True)

    env_cfg = _load_env_cfg(args.task, args.teacher_config, args.num_envs, args.sim_device)
    if args.student_input == "teacher_obs":
        env_cfg.student_obs.image_enabled = False
    if args.force_scene_tol_combo is not None:
        env_cfg.peg_in_hole.force_scene_tol_combo = _parse_optional_pair(args.force_scene_tol_combo)
    if args.force_peg_idx is not None:
        env_cfg.peg_in_hole.force_peg_idx = args.force_peg_idx
    if args.depth_noise_profile is not None:
        env_cfg.student_obs.depth_noise_profile = args.depth_noise_profile
    if args.depth_noise_strength is not None:
        env_cfg.student_obs.depth_noise_strength = args.depth_noise_strength
    if args.camera_pose_randomization_profile is not None:
        env_cfg.student_obs.camera_pose_randomization_profile = args.camera_pose_randomization_profile
    if args.camera_pose_randomization_mode is not None:
        env_cfg.student_obs.camera_pose_randomization_mode = args.camera_pose_randomization_mode
    if args.camera_pos_noise_m is not None:
        env_cfg.student_obs.camera_pos_noise_m = tuple(args.camera_pos_noise_m)
    if args.camera_rot_noise_deg is not None:
        env_cfg.student_obs.camera_rot_noise_deg = tuple(args.camera_rot_noise_deg)
    if args.peg_object_init_orientation_mode is not None:
        env_cfg.peg_in_hole.object_init_orientation_mode = args.peg_object_init_orientation_mode
    if args.peg_object_init_position_noise_xy is not None:
        env_cfg.peg_in_hole.object_init_position_noise_xy = tuple(args.peg_object_init_position_noise_xy)
    if args.peg_object_init_position_noise_z is not None:
        env_cfg.peg_in_hole.object_init_position_noise_z = args.peg_object_init_position_noise_z

    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped
    wrapped, teacher = _load_teacher_player(
        env,
        task=args.task,
        agent=args.teacher_agent,
        checkpoint=args.teacher_checkpoint,
        rl_device=args.rl_device,
    )
    obs = teacher.env_reset(wrapped)

    if args.student_input == "teacher_obs":
        arch = args.student_arch or "mlp_recurrent"
        if arch != "mlp_recurrent":
            raise ValueError("--student_input teacher_obs currently requires --student_arch mlp_recurrent")
        teacher_obs = _teacher_obs_tensor(obs)
        student = MLPRecurrentPolicy(
            obs_dim=teacher_obs.shape[-1],
            action_dim=int(env_cfg.action_space),
            aux_heads=_aux_head_dims(args.aux_pose_mode),
        ).to(inner.device)
        print(
            f"[distill_depth] teacher_obs student obs_dim={teacher_obs.shape[-1]} "
            f"action_dim={int(env_cfg.action_space)} aux_pose_mode={args.aux_pose_mode}",
            flush=True,
        )
    else:
        arch = args.student_arch or "mono_transformer_recurrent"
        if arch != "mono_transformer_recurrent":
            raise ValueError("--student_input camera currently requires --student_arch mono_transformer_recurrent")
        student_obs = inner.get_student_obs()
        image = student_obs["image"]
        proprio = student_obs["proprio"]
        student = MonoTransformerRecurrentPolicy(
            image_channels=_student_image_channels(str(env_cfg.student_obs.image_modality).lower()),
            proprio_dim=proprio.shape[-1],
            action_dim=int(env_cfg.action_space),
            aux_heads=_aux_head_dims(args.aux_pose_mode),
        ).to(inner.device)
        print(
            f"[distill_depth] camera student image_shape={tuple(image.shape)} "
            f"proprio_dim={proprio.shape[-1]} action_dim={int(env_cfg.action_space)} "
            f"aux_pose_mode={args.aux_pose_mode}",
            flush=True,
        )
    _initialize_rot6d_head(student)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    start_step, best_metric = _load_student_checkpoint(
        args.student_checkpoint,
        student,
        optimizer,
        strict=args.student_checkpoint_strict,
    )
    hidden = student.initial_state(inner.num_envs, inner.device)
    wandb_run = _init_wandb(args, run_dir)

    depth_debug_env_ids = [
        int(item) for item in args.depth_debug_env_ids.replace(",", " ").split() if item
    ]
    depth_debug_env_ids = [idx for idx in depth_debug_env_ids if 0 <= idx < inner.num_envs]
    viewer_frames: list = []
    depth_video_frames: list = []
    depth_rollout_video_frames: list = []
    depth_rollout_env_ids = [
        int(item) for item in args.wandb_depth_rollout_video_env_ids.replace(",", " ").split() if item
    ]
    depth_rollout_env_ids = [idx for idx in depth_rollout_env_ids if 0 <= idx < inner.num_envs]
    interval_action_loss = 0.0
    interval_aux_loss = 0.0
    interval_aux_pos_loss = 0.0
    interval_aux_keypoint_loss = 0.0
    interval_step_count = 0
    interval_done_goal_idx = 0.0
    interval_done_completion = 0.0
    interval_done_count = 0
    interval_start = time.perf_counter()

    for local_step in range(args.num_iters):
        step = start_step + local_step + 1
        with torch.no_grad():
            teacher_action = teacher.get_action(obs, is_deterministic=args.deterministic_teacher)

        if args.student_input == "teacher_obs":
            student_out, next_hidden = student(_teacher_obs_tensor(obs), hidden)
            image = None
        else:
            student_obs = inner.get_student_obs()
            image = student_obs["image"]
            proprio = student_obs["proprio"]
            student_out, next_hidden = student(image, proprio, hidden)
        student_action = student_out.action
        action_loss = F.mse_loss(student_action, teacher_action, reduction="none").mean(dim=1)
        aux_loss, aux_info = _compute_aux_losses(inner, student_out.aux, args)
        loss = args.action_loss_weight * action_loss.mean() + aux_loss.mean()

        if args.mode == "train_online":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            hidden = next_hidden.detach()
            action_for_env = student_action.detach()
        elif args.mode == "student_eval":
            hidden = next_hidden.detach()
            action_for_env = student_action.detach()
        else:
            hidden = hidden.detach()
            action_for_env = teacher_action

        if args.capture_viewer:
            predicted_pose_for_viewer = aux_info["pred_pose_wxyz"]
            if args.mode == "teacher_eval" and args.student_checkpoint is None:
                predicted_pose_for_viewer = None
            viewer_frames = _capture_viewer_if_needed(
                env=inner,
                frames=viewer_frames,
                output_dir=run_dir / "interactive_viewer",
                capture_len=args.capture_viewer_len,
                step=step,
                predicted_object_pose_wxyz=predicted_pose_for_viewer,
                wandb_run=wandb_run,
                wandb_key=args.wandb_viewer_key,
            )

        obs, reward, dones, infos = teacher.env_step(wrapped, action_for_env)
        hidden = _reset_hidden_for_done(hidden, dones)
        done_goal_idx, done_completion, done_count = _done_success_stats(inner, dones)
        if done_count:
            interval_done_goal_idx += done_goal_idx * done_count
            interval_done_completion += done_completion * done_count
            interval_done_count += done_count

        if (
            args.student_input == "camera"
            and args.depth_debug_interval > 0
            and (step == 1 or step % args.depth_debug_interval == 0)
        ):
            raw_depth = inner.student_camera.data.output["distance_to_image_plane"]
            depth_paths = save_depth_debug(
                output_dir=run_dir / "depth_debug",
                step=step,
                env_ids=depth_debug_env_ids or [0],
                raw_depth=raw_depth,
                noisy_depth=getattr(inner, "_student_depth_noisy_m", None),
                policy_full_depth=getattr(inner, "_student_depth_policy_full", None),
                policy_depth=image,
                near=float(env_cfg.student_obs.depth_min_m),
                far=float(env_cfg.student_obs.depth_max_m),
            )
            if args.wandb_log_depth_media:
                depth_video_frames = _log_depth_debug_media(
                    paths=depth_paths,
                    wandb_run=wandb_run,
                    step=step,
                    frames=depth_video_frames,
                    max_frames=args.wandb_depth_video_max_frames,
                    fps=args.wandb_depth_video_fps,
                    log_video=step == 1 or step % max(1, args.wandb_depth_video_interval) == 0,
                    key_prefix=args.wandb_depth_media_key,
                )

        if (
            args.student_input == "camera"
            and args.wandb_log_depth_media
            and args.wandb_depth_rollout_video_len > 0
            and depth_rollout_env_ids
        ):
            frame = _capture_depth_rollout_frame(
                env=inner,
                policy_depth=image,
                env_ids=depth_rollout_env_ids,
                near=float(env_cfg.student_obs.depth_min_m),
                far=float(env_cfg.student_obs.depth_max_m),
            )
            depth_rollout_video_frames = _log_depth_rollout_video(
                frame=frame,
                frames=depth_rollout_video_frames,
                wandb_run=wandb_run,
                step=step,
                key=args.wandb_depth_rollout_video_key,
                fps=args.wandb_depth_rollout_video_fps,
                max_frames=args.wandb_depth_rollout_video_len,
                interval=args.wandb_depth_rollout_video_interval,
            )

        interval_action_loss += float(action_loss.mean().detach().cpu().item())
        interval_aux_loss += float(aux_loss.mean().detach().cpu().item())
        interval_aux_pos_loss += float(aux_info["pos_loss"].mean().detach().cpu().item())
        interval_aux_keypoint_loss += float(aux_info["keypoint_loss"].mean().detach().cpu().item())
        interval_step_count += 1

        should_log = step == 1 or step % args.log_interval == 0 or local_step == args.num_iters - 1
        if should_log:
            elapsed = max(time.perf_counter() - interval_start, 1e-6)
            current_goal_idx = float(inner._successes.float().mean().detach().cpu().item())
            current_completion = float(
                (inner._successes.float() / inner.env_max_goals.clamp_min(1).float()).mean().detach().cpu().item()
            )
            recent_goal_idx = interval_done_goal_idx / max(interval_done_count, 1)
            recent_completion = interval_done_completion / max(interval_done_count, 1)
            row = {
                "step": step,
                "mode": args.mode,
                "action_loss": interval_action_loss / max(interval_step_count, 1),
                "action_rmse": math.sqrt(max(interval_action_loss / max(interval_step_count, 1), 0.0)),
                "aux_loss": interval_aux_loss / max(interval_step_count, 1),
                "aux_object_pos_loss": interval_aux_pos_loss / max(interval_step_count, 1),
                "aux_object_pos_rmse_m": math.sqrt(max(interval_aux_pos_loss / max(interval_step_count, 1), 0.0)),
                "aux_object_keypoint_loss": interval_aux_keypoint_loss / max(interval_step_count, 1),
                "aux_object_keypoint_rmse_m": math.sqrt(
                    max(interval_aux_keypoint_loss / max(interval_step_count, 1), 0.0)
                ),
                "current_goal_idx_avg": current_goal_idx,
                "current_goal_completion_ratio_avg": current_completion,
                "recent_reset_goal_idx_avg": recent_goal_idx,
                "recent_reset_goal_completion_ratio_avg": recent_completion,
                "recent_reset_count": interval_done_count,
                "env_steps_per_s": inner.num_envs * interval_step_count / elapsed,
            }
            print(
                "[distill_depth] "
                f"step={step} mode={args.mode} current_goal_idx={current_goal_idx:.2f} "
                f"recent_reset_goal_idx={recent_goal_idx:.2f} recent_reset_count={interval_done_count} "
                f"action_rmse={row['action_rmse']:.4f} aux_pos_rmse_cm={100.0 * row['aux_object_pos_rmse_m']:.2f} "
                f"aux_kp_rmse_cm={100.0 * row['aux_object_keypoint_rmse_m']:.2f}",
                flush=True,
            )
            _log_metrics(run_dir, row, wandb_run=wandb_run, step=step)
            interval_action_loss = 0.0
            interval_aux_loss = 0.0
            interval_aux_pos_loss = 0.0
            interval_aux_keypoint_loss = 0.0
            interval_step_count = 0
            interval_done_goal_idx = 0.0
            interval_done_completion = 0.0
            interval_done_count = 0
            interval_start = time.perf_counter()

            selection_metric = recent_completion if row["recent_reset_count"] > 0 else current_completion
            if args.mode == "train_online" and selection_metric >= best_metric:
                best_metric = selection_metric
                _save_checkpoint(run_dir / "checkpoints" / "student_best.pt", student, optimizer, step, best_metric)

        if args.mode == "train_online" and step % args.save_interval == 0:
            _save_checkpoint(run_dir / "checkpoints" / "student_latest.pt", student, optimizer, step, best_metric)

    if viewer_frames:
        _capture_viewer_if_needed(
            env=inner,
            frames=viewer_frames,
            output_dir=run_dir / "interactive_viewer",
            capture_len=len(viewer_frames),
            step=start_step + args.num_iters,
            append_frame=False,
            wandb_run=wandb_run,
            wandb_key=args.wandb_viewer_key,
        )
    if args.mode == "train_online":
        _save_checkpoint(run_dir / "checkpoints" / "student_latest.pt", student, optimizer, start_step + args.num_iters, best_metric)

    env.close()
    if wandb_run is not None:
        wandb_run.finish()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
