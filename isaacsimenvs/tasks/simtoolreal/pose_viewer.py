"""Pose-only interactive HTML capture for SimToolReal training.

This module deliberately does not use Isaac cameras, RTX sensors, Replicator, or
the Isaac viewport.  It samples one environment's state tensors and writes a
Three.js/URDF HTML viewer that can be opened locally or logged to WandB.
"""

from __future__ import annotations

import time
import urllib.request
import subprocess
import posixpath
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

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
FURNITURE_BENCH_RAW_BASE = (
    "https://raw.githubusercontent.com/clvrai/furniture-bench/main/"
)
FURNITUREBENCH_Y_UP_TO_USD_Z_UP_RPY = (1.5707963267948966, 0.0, 0.0)
FURNITUREBENCH_VIEWER_ORIGINS = {
    # The source FurnitureBench OBJ/URDF assets are Y-up.  OmniReset's sim USDs
    # are Z-up.  These viewer-only origins make the Three.js URDF visuals match
    # the USD root frames used by physics without embedding USD geometry in HTML.
    "square_table_leg1.urdf": {
        "xyz": (0.0, 0.0006175, -0.00019225),
        "rpy": FURNITUREBENCH_Y_UP_TO_USD_Z_UP_RPY,
    },
    "square_table_top.urdf": {
        "xyz": (0.0, 0.0000005, 0.0000494),
        "rpy": FURNITUREBENCH_Y_UP_TO_USD_Z_UP_RPY,
    },
}


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


def _resolve_local_path(path: str | Path) -> Path:
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def _is_url(path: str | Path) -> bool:
    return str(path).startswith(("http://", "https://"))


def _furniturebench_urdf_relpath(urdf_source: str | Path) -> str | None:
    source = str(urdf_source)
    if _is_url(source):
        parts = [part for part in urlparse(source).path.split("/") if part]
    else:
        parts = list(Path(source).parts)
    try:
        furniture_idx = parts.index("furniture_bench")
    except ValueError:
        return None
    return posixpath.join(*parts[furniture_idx:])


def _furniturebench_viewer_origin(urdf_source: str | Path) -> dict[str, tuple[float, float, float]] | None:
    if _is_url(urdf_source):
        filename = Path(urlparse(str(urdf_source)).path).name
    else:
        filename = Path(urdf_source).name
    return FURNITUREBENCH_VIEWER_ORIGINS.get(filename)


def _format_origin_tuple(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _apply_viewer_origin_to_mesh_elements(
    root: ET.Element,
    origin_cfg: dict[str, tuple[float, float, float]],
) -> None:
    for visual_or_collision in root.findall(".//visual") + root.findall(".//collision"):
        if visual_or_collision.find("./geometry/mesh") is None:
            continue
        origin = visual_or_collision.find("origin")
        if origin is None:
            origin = ET.Element("origin")
            visual_or_collision.insert(0, origin)
        origin.attrib["xyz"] = _format_origin_tuple(origin_cfg["xyz"])
        origin.attrib["rpy"] = _format_origin_tuple(origin_cfg["rpy"])


def _rewrite_furniturebench_urdf_mesh_urls(urdf_text: str, urdf_source: str | Path) -> str:
    """Rewrite FurnitureBench URDF mesh paths to raw GitHub URLs for W&B HTML.

    The sim still uses USD assets. This is only for Three.js visualization,
    where embedded URDF text otherwise has relative mesh paths that W&B cannot
    resolve.
    """

    urdf_rel = _furniturebench_urdf_relpath(urdf_source)
    if urdf_rel is None:
        return urdf_text

    root = ET.fromstring(urdf_text)
    origin_cfg = _furniturebench_viewer_origin(urdf_source)
    if origin_cfg is not None:
        _apply_viewer_origin_to_mesh_elements(root, origin_cfg)

    urdf_dir_rel = posixpath.dirname(urdf_rel)
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith(("http://", "https://", "package://")):
            continue
        if filename.startswith("../mesh/"):
            # FurnitureBench URDFs use paths relative to the furniture asset
            # root, not relative to the URDF file's directory.
            mesh_rel = posixpath.normpath(
                "furniture_bench/assets/furniture/" + filename.removeprefix("../")
            )
        elif filename.startswith("mesh/"):
            mesh_rel = posixpath.normpath("furniture_bench/assets/furniture/" + filename)
        else:
            mesh_rel = posixpath.normpath(posixpath.join(urdf_dir_rel, filename))
        suffix = Path(filename).suffix.lower()
        mesh.attrib["filename"] = f"{FURNITURE_BENCH_RAW_BASE}{mesh_rel}#ext={suffix}"

    return ET.tostring(root, encoding="unicode")


def _viewer_urdf_text(path: str | Path) -> str:
    if _is_url(path):
        with urllib.request.urlopen(str(path), timeout=20) as response:
            text = response.read().decode("utf-8")
        return _rewrite_furniturebench_urdf_mesh_urls(text, path)
    urdf_path = _resolve_local_path(path)
    text = urdf_path.read_text(encoding="utf-8")
    return _rewrite_furniturebench_urdf_mesh_urls(text, urdf_path)


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

    override = getattr(getattr(env.cfg, "assets", None), "object_viewer_urdf_path", "")
    if str(override).strip():
        return _viewer_urdf_text(override)

    urdf_paths = getattr(env, "_object_urdf_paths", None)
    asset_indices = getattr(env, "_object_asset_index_per_env", None)
    if not urdf_paths or asset_indices is None:
        raise RuntimeError(
            "SimToolReal env does not expose object URDF mapping. "
            "Expected _object_urdf_paths and _object_asset_index_per_env."
        )

    asset_index = int(asset_indices[env_id].detach().cpu().item())
    urdf_path = Path(urdf_paths[asset_index])
    if urdf_path.suffix.lower() != ".urdf":
        raise RuntimeError(
            f"Object asset for env {env_id} is {urdf_path}, not a URDF. "
            "Set cfg.assets.object_viewer_urdf_path for interactive viewer HTML."
        )
    return _viewer_urdf_text(urdf_path)


def table_urdf_text_for_env(env, env_id: int) -> str:
    """Return the table URDF text assigned to one env."""

    table_paths = getattr(env, "_table_urdf_paths", None)
    if table_paths:
        return _viewer_urdf_text(table_paths[env_id % len(table_paths)])
    return TABLE_URDF_PATH.read_text(encoding="utf-8")


def fixture_urdf_text_for_env(env, env_id: int) -> str | None:
    """Return optional fixture URDF text for viewer-only visualization."""

    if getattr(env, "fixture", None) is None:
        return None
    override = getattr(getattr(env.cfg, "assets", None), "fixture_viewer_urdf_path", "")
    if not str(override).strip():
        return None
    return _viewer_urdf_text(override)


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
    if getattr(env, "fixture", None) is not None:
        fixture_pos = env.fixture.data.root_pos_w[env_id] - origin
        frame["fixture_pose"] = _pose_xyzw(fixture_pos, env.fixture.data.root_quat_w[env_id])
    if predicted_object_pose_wxyz is not None:
        pred = predicted_object_pose_wxyz[env_id]
        frame["object_pred_pose"] = _pose_xyzw(pred[:3], pred[3:7])
    return frame


def build_pose_viewer_html(
    *,
    frames: list[dict[str, Any]],
    object_urdf_text: str,
    table_urdf_text: str,
    fixture_urdf_text: str | None = None,
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
    has_fixture = fixture_urdf_text is not None and all("fixture_pose" in frame for frame in frames)
    if has_fixture:
        robots.append(
            make_embedded_robot(
                name="fixture",
                urdf_text=fixture_urdf_text,
                color_override=(0.62, 0.62, 0.62),
            )
        )
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
    if has_fixture:
        object_poses["fixture"] = np.stack([frame["fixture_pose"] for frame in frames])
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
        self._fixture_urdf_text = fixture_urdf_text_for_env(inner, self.env_id)

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
            fixture_urdf_text=self._fixture_urdf_text,
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
