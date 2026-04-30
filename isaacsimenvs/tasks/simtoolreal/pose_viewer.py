"""Pose-only interactive HTML capture for SimToolReal training.

This module deliberately does not use Isaac cameras, RTX sensors, Replicator, or
the Isaac viewport.  It samples one environment's state tensors and writes a
Three.js/URDF HTML viewer that can be opened locally or logged to WandB.
"""

from __future__ import annotations

import time
import urllib.request
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gymnasium as gym
import numpy as np

from isaacsim_conversion.interactive_viewer import create_html, make_embedded_robot, make_url_robot

from .utils.scene_utils import JOINT_NAMES_CANONICAL


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/tylerlum/simtoolreal/"
    "6809a978753e950913a7588bbeaef07d16f10b56/"
)
ROBOT_URDF_RELATIVE_PATH = "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
TABLE_URDF_PATH = REPO_ROOT / "assets" / "urdf" / "table_narrow.urdf"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _quat_wxyz_to_xyzw(quat) -> np.ndarray:
    quat_np = _to_numpy(quat)
    return quat_np[[1, 2, 3, 0]]


def _pose_xyzw(pos, quat_wxyz) -> np.ndarray:
    pose = np.zeros(7, dtype=np.float32)
    pose[:3] = _to_numpy(pos)
    pose[3:] = _quat_wxyz_to_xyzw(quat_wxyz)
    return pose


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _github_slug_from_remote(remote_url: str | None) -> str | None:
    if not remote_url:
        return None

    remote = remote_url.strip()
    if remote.endswith(".git"):
        remote = remote[:-4]

    prefixes = (
        "git@github.com:",
        "ssh://git@github.com/",
        "https://github.com/",
        "http://github.com/",
    )
    for prefix in prefixes:
        if remote.startswith(prefix):
            slug = remote[len(prefix):]
            parts = slug.split("/")
            if len(parts) >= 2:
                return "/".join(parts[:2])
    return None


def _derive_github_raw_base() -> str:
    remote_url = _git_output(["remote", "get-url", "origin"])
    slug = _github_slug_from_remote(remote_url)
    if slug is None:
        print(
            f"[pose_viewer] Could not derive GitHub origin from {remote_url!r}; "
            f"falling back to {DEFAULT_GITHUB_RAW_BASE}",
            flush=True,
        )
        return DEFAULT_GITHUB_RAW_BASE

    ref = _git_output(["branch", "--show-current"])
    if not ref:
        ref = _git_output(["rev-parse", "HEAD"])
    if not ref:
        print(
            f"[pose_viewer] Could not derive git branch/commit; falling back to {DEFAULT_GITHUB_RAW_BASE}",
            flush=True,
        )
        return DEFAULT_GITHUB_RAW_BASE

    raw_base = f"https://raw.githubusercontent.com/{slug}/{quote(ref, safe='')}/"
    print(f"[pose_viewer] GitHub raw base: {raw_base}", flush=True)
    return raw_base


def _normalize_raw_base(github_raw_base: str | None) -> str:
    # Default to the stable SimToolReal commit containing the robot URDF/meshes.
    # Current experiment branches are often local/unpushed, so branch-derived
    # raw GitHub URLs produce broken viewer HTML.
    base = github_raw_base or DEFAULT_GITHUB_RAW_BASE
    return base if base.endswith("/") else base + "/"


def _check_url(url: str, url_check: str) -> None:
    if url_check == "skip":
        return
    print(f"[pose_viewer] URL check ({url_check}) -> {url}", flush=True)
    start = time.monotonic()
    try:
        request = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(request, timeout=10)
        print(f"[pose_viewer]   PASSED ({time.monotonic() - start:.2f}s)", flush=True)
    except Exception as exc:
        message = f"[pose_viewer]   FAILED ({time.monotonic() - start:.2f}s): {exc}"
        if url_check == "error":
            raise RuntimeError(message) from exc
        print(message, flush=True)


def object_urdf_text_for_env(env, env_id: int) -> str:
    """Return the procedural object URDF text assigned to one env."""

    urdf_paths = getattr(env, "_object_urdf_paths", None)
    asset_indices = getattr(env, "_object_asset_index_per_env", None)
    if not urdf_paths or asset_indices is None:
        raise RuntimeError(
            "SimToolReal env does not expose object URDF mapping. "
            "Expected _object_urdf_paths and _object_asset_index_per_env."
        )

    asset_index = int(asset_indices[env_id].detach().cpu().item())
    urdf_path = Path(urdf_paths[asset_index])
    return urdf_path.read_text(encoding="utf-8")


def table_urdf_text_for_env(env, env_id: int) -> str:
    """Return the table URDF text assigned to one env."""

    table_paths = getattr(env, "_table_urdf_paths", None)
    if table_paths:
        return Path(table_paths[env_id % len(table_paths)]).read_text(encoding="utf-8")
    return TABLE_URDF_PATH.read_text(encoding="utf-8")


def capture_pose_viewer_frame(env, env_id: int, *, predicted_object_pose_wxyz=None) -> dict[str, Any]:
    """Capture one env-local frame from a live SimToolRealEnv."""

    if env_id < 0 or env_id >= env.num_envs:
        raise ValueError(f"capture_viewer_env_id={env_id} out of range for num_envs={env.num_envs}")

    origin = env.scene.env_origins[env_id]

    if hasattr(env, "_perm_lab_to_canon"):
        joint_pos = env.robot.data.joint_pos[env_id, env._perm_lab_to_canon]
        joint_names = list(JOINT_NAMES_CANONICAL)
    else:
        joint_pos = env.robot.data.joint_pos[env_id]
        joint_names = list(env.robot.data.joint_names)

    robot_root_pos = env.robot.data.root_pos_w[env_id] - origin
    object_pos = env.object.data.root_pos_w[env_id] - origin
    goal_pos = env.goal_viz.data.root_pos_w[env_id] - origin
    table_pos = env.table.data.root_pos_w[env_id] - origin

    frame = {
        "env_id": int(env_id),
        "robot_joint_names": joint_names,
        "robot_joint_pos": _to_numpy(joint_pos),
        "robot_base_pose": _pose_xyzw(robot_root_pos, env.robot.data.root_quat_w[env_id]),
        "object_pose": _pose_xyzw(object_pos, env.object.data.root_quat_w[env_id]),
        "goal_pose": _pose_xyzw(goal_pos, env.goal_viz.data.root_quat_w[env_id]),
        "table_pose": _pose_xyzw(table_pos, env.table.data.root_quat_w[env_id]),
    }
    if predicted_object_pose_wxyz is not None:
        pred = predicted_object_pose_wxyz[env_id]
        frame["object_pred_pose"] = _pose_xyzw(pred[:3], pred[3:7])
    return frame


def build_pose_viewer_html(
    *,
    frames: list[dict[str, Any]],
    object_urdf_text: str,
    table_urdf_text: str,
    github_raw_base: str | None = None,
    url_check: str = "skip",
) -> str:
    """Build a self-contained-ish viewer HTML string from captured frames.

    Object and table URDFs are embedded.  The robot URDF is URL-backed because
    the SHARPA hand references mesh files that the browser must fetch.
    """

    if not frames:
        raise ValueError("Cannot build pose viewer from zero frames.")

    raw_base = _normalize_raw_base(github_raw_base)
    robot_urdf_url = raw_base + ROBOT_URDF_RELATIVE_PATH
    _check_url(robot_urdf_url, url_check)

    timestamps = np.arange(len(frames), dtype=np.float32) / 60.0
    robots = [
        make_url_robot(name="robot", urdf_url=robot_urdf_url, animated=True),
        make_embedded_robot(name="table", urdf_text=table_urdf_text),
        make_embedded_robot(name="object", urdf_text=object_urdf_text),
        make_embedded_robot(
            name="goal",
            urdf_text=object_urdf_text,
            color_override=(0.20, 0.72, 0.31),
        ),
    ]
    has_predicted_object = all("object_pred_pose" in frame for frame in frames)
    if has_predicted_object:
        robots.append(
            make_embedded_robot(
                name="object_pred",
                urdf_text=object_urdf_text,
                color_override=(0.05, 0.28, 1.00),
            )
        )

    object_poses = {
        "table": np.stack([frame["table_pose"] for frame in frames]),
        "object": np.stack([frame["object_pose"] for frame in frames]),
        "goal": np.stack([frame["goal_pose"] for frame in frames]),
    }
    if has_predicted_object:
        object_poses["object_pred"] = np.stack([frame["object_pred_pose"] for frame in frames])

    return create_html(
        joint_names=frames[0]["robot_joint_names"],
        robot_joint_positions=np.stack([frame["robot_joint_pos"] for frame in frames]),
        robots=robots,
        object_poses=object_poses,
        robot_base_poses=np.stack([frame["robot_base_pose"] for frame in frames]),
        timestamps=timestamps,
    )


class SimToolRealPoseViewerWrapper(gym.Wrapper):
    """Gym wrapper that periodically writes pose-only interactive HTML rollouts."""

    def __init__(
        self,
        env: gym.Env,
        *,
        output_dir: str | Path,
        capture_len: int,
        capture_interval: int,
        env_id: int = 0,
        wandb_key: str = "interactive_viewer",
        github_raw_base: str | None = None,
        url_check: str = "skip",
    ) -> None:
        super().__init__(env)
        if capture_len <= 0:
            raise ValueError(f"capture_viewer_len must be > 0, got {capture_len}")
        if url_check not in {"skip", "warn", "error"}:
            raise ValueError(f"capture_viewer_url_check must be skip/warn/error, got {url_check}")

        inner = self.env.unwrapped
        if env_id < 0 or env_id >= inner.num_envs:
            raise ValueError(f"capture_viewer_env_id={env_id} out of range for num_envs={inner.num_envs}")

        self.output_dir = Path(output_dir)
        self.capture_len = int(capture_len)
        self.capture_interval = int(capture_interval)
        self.env_id = int(env_id)
        self.wandb_key = wandb_key
        self.github_raw_base = github_raw_base
        self.url_check = url_check
        self._object_urdf_text = object_urdf_text_for_env(inner, self.env_id)
        self._table_urdf_text = table_urdf_text_for_env(inner, self.env_id)

        self._step = 0
        self._capture_index = 0
        self._frames: list[dict[str, Any]] | None = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[pose_viewer] enabled: "
            f"env_id={self.env_id} len={self.capture_len} interval={self.capture_interval} "
            f"output_dir={self.output_dir}",
            flush=True,
        )

    def step(self, action):
        result = self.env.step(action)
        self._step += 1

        if self._frames is None and self.capture_interval > 0 and self._step % self.capture_interval == 0:
            self._frames = []

        if self._frames is not None:
            self._frames.append(capture_pose_viewer_frame(self.env.unwrapped, self.env_id))
            if len(self._frames) >= self.capture_len:
                self._finalize_capture()

        return result

    def close(self) -> None:
        if self._frames:
            self._finalize_capture(partial=True)
        return self.env.close()

    def _finalize_capture(self, *, partial: bool = False) -> None:
        assert self._frames is not None
        frames = self._frames
        if not frames:
            self._frames = None
            return

        suffix = "partial" if partial else f"step_{self._step:09d}"
        html_path = self.output_dir / f"pose_viewer_{suffix}_{self._capture_index:04d}.html"
        html_text = build_pose_viewer_html(
            frames=frames,
            object_urdf_text=self._object_urdf_text,
            table_urdf_text=self._table_urdf_text,
            github_raw_base=self.github_raw_base,
            url_check=self.url_check,
        )
        html_path.write_text(html_text, encoding="utf-8")
        print(f"[pose_viewer] wrote {len(frames)} frames to {html_path}", flush=True)
        self._log_wandb(html_text)

        self._capture_index += 1
        self._frames = None

    def _log_wandb(self, html_text: str) -> None:
        try:
            import wandb
        except Exception:
            return

        if wandb.run is None:
            return

        try:
            wandb.log(
                {"global_step": self._step, self.wandb_key: wandb.Html(html_text)},
                step=self._step,
            )
            print(f"[pose_viewer] logged WandB Html key={self.wandb_key} step={self._step}", flush=True)
        except Exception as exc:
            print(f"[pose_viewer] WandB log failed: {exc}", flush=True)
