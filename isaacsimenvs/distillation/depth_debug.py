"""Depth image debug artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def depth_tensor_to_nchw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 4 and depth.shape[-1] == 1:
        return depth.permute(0, 3, 1, 2)
    if depth.ndim == 3:
        return depth.unsqueeze(1)
    if depth.ndim == 4 and depth.shape[1] == 1:
        return depth
    raise RuntimeError(f"Unsupported depth tensor shape: {tuple(depth.shape)}")


def make_image_grid(images: np.ndarray, *, pad: int = 2) -> np.ndarray:
    images = np.clip(images, 0.0, 1.0)
    n, h, w = images.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.full(
        (rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad),
        255,
        dtype=np.uint8,
    )
    images_u8 = (images * 255.0).round().astype(np.uint8)
    for idx in range(n):
        row = idx // cols
        col = idx % cols
        y0 = row * (h + pad)
        x0 = col * (w + pad)
        grid[y0 : y0 + h, x0 : x0 + w] = images_u8[idx]
    return grid


def _save_png(path: Path, images: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(make_image_grid(images)).save(path)


def _gray_to_panel(image: np.ndarray, *, label: str, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)
    gray = (image * 255.0).round().astype(np.uint8)
    panel = Image.fromarray(gray).resize(size, resample=Image.Resampling.NEAREST).convert("RGB")
    label_h = 16
    canvas = Image.new("RGB", (size[0], size[1] + label_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((3, 2), label, fill=(0, 0, 0))
    canvas.paste(panel, (0, label_h))
    return np.asarray(canvas)


def _make_review_grid(
    *,
    env_ids: list[int],
    raw_window: np.ndarray,
    policy: np.ndarray,
    noisy_window: np.ndarray | None = None,
    policy_full: np.ndarray | None = None,
) -> np.ndarray:
    """Build a labeled RGB grid for quick W&B/video review."""
    panel_size = (160, 90)
    rows: list[np.ndarray] = []
    for i, env_id in enumerate(env_ids):
        panels = [
            _gray_to_panel(raw_window[i], label=f"env{env_id} raw window", size=panel_size),
            _gray_to_panel(
                raw_window[i] if noisy_window is None else noisy_window[i],
                label="noisy window",
                size=panel_size,
            ),
        ]
        if policy_full is not None:
            panels.append(_gray_to_panel(policy_full[i], label="policy full", size=panel_size))
        panels.append(_gray_to_panel(policy[i], label="student crop", size=panel_size))
        rows.append(np.concatenate(panels, axis=1))
    return np.concatenate(rows, axis=0)


def _stats(array: np.ndarray, *, near: float, far: float) -> dict[str, float]:
    finite = np.isfinite(array)
    if not finite.any():
        return {
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "in_window_frac": 0.0,
        }
    valid = array[finite]
    return {
        "min": float(np.min(valid)),
        "p25": float(np.quantile(valid, 0.25)),
        "median": float(np.median(valid)),
        "mean": float(np.mean(valid)),
        "p75": float(np.quantile(valid, 0.75)),
        "max": float(np.max(valid)),
        "in_window_frac": float(((array >= near) & (array <= far) & finite).mean()),
    }


def save_depth_debug(
    *,
    output_dir: Path,
    step: int,
    env_ids: list[int],
    raw_depth: torch.Tensor,
    policy_depth: torch.Tensor,
    near: float,
    far: float,
    noisy_depth: torch.Tensor | None = None,
    policy_full_depth: torch.Tensor | None = None,
) -> dict[str, Path]:
    """Save raw/noisy metric depth, window visualizations, policy depth, and stats."""

    output_dir.mkdir(parents=True, exist_ok=True)
    env_ids = list(env_ids)
    raw = depth_tensor_to_nchw(raw_depth).detach().float().cpu().numpy()[env_ids, 0]
    noisy = None
    if noisy_depth is not None:
        noisy = depth_tensor_to_nchw(noisy_depth).detach().float().cpu().numpy()[env_ids, 0]
    policy = depth_tensor_to_nchw(policy_depth).detach().float().cpu().numpy()[env_ids, 0]
    policy_full = None
    if policy_full_depth is not None:
        policy_full = depth_tensor_to_nchw(policy_full_depth).detach().float().cpu().numpy()[env_ids, 0]
    raw_window = np.nan_to_num((raw - near) / max(far - near, 1e-6), nan=0.0, posinf=1.0, neginf=0.0)
    raw_window = np.clip(raw_window, 0.0, 1.0)
    noisy_window = None
    if noisy is not None:
        noisy_window = np.nan_to_num(
            (noisy - near) / max(far - near, 1e-6),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        noisy_window = np.clip(noisy_window, 0.0, 1.0)

    prefix = output_dir / f"step_{step:08d}"
    arrays = {
        "env_ids": np.asarray(env_ids, dtype=np.int32),
        "raw_depth_m": raw,
        "raw_depth_window": raw_window,
        "policy_depth": policy,
    }
    if policy_full is not None:
        arrays["policy_full_depth"] = policy_full
    if noisy is not None and noisy_window is not None:
        arrays["noisy_depth_m"] = noisy
        arrays["noisy_depth_window"] = noisy_window
        arrays["noisy_minus_raw_m"] = noisy - raw
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    paths: dict[str, Path] = {"npz": prefix.with_suffix(".npz")}
    raw_path = prefix.with_name(prefix.name + "_raw_window.png")
    _save_png(raw_path, raw_window)
    paths["raw_window_png"] = raw_path
    if noisy_window is not None:
        noisy_path = prefix.with_name(prefix.name + "_noisy_window.png")
        _save_png(noisy_path, noisy_window)
        paths["noisy_window_png"] = noisy_path
    if policy_full is not None:
        policy_full_path = prefix.with_name(prefix.name + "_policy_full_depth.png")
        _save_png(policy_full_path, policy_full)
        paths["policy_full_depth_png"] = policy_full_path
    policy_path = prefix.with_name(prefix.name + "_policy_depth.png")
    _save_png(policy_path, policy)
    paths["policy_depth_png"] = policy_path
    review_path = prefix.with_name(prefix.name + "_review_grid.png")
    from PIL import Image

    Image.fromarray(
        _make_review_grid(
            env_ids=env_ids,
            raw_window=raw_window,
            noisy_window=noisy_window,
            policy_full=policy_full,
            policy=policy,
        )
    ).save(review_path)
    paths["review_grid_png"] = review_path

    stats = {
        str(env_id): {
            "raw_depth_m": _stats(raw[i], near=near, far=far),
            **(
                {
                    "noisy_depth_m": _stats(noisy[i], near=near, far=far),
                    "noisy_minus_raw_m": _stats(noisy[i] - raw[i], near=-0.01, far=0.01),
                }
                if noisy is not None
                else {}
            ),
            "policy_depth": _stats(policy[i], near=0.0, far=1.0),
            "policy_nonzero_frac": float((policy[i] > 0.0).mean()),
            "policy_sat_low_frac": float((policy[i] <= 1e-6).mean()),
            "policy_sat_high_frac": float((policy[i] >= 1.0 - 1e-6).mean()),
        }
        for i, env_id in enumerate(env_ids)
    }
    prefix.with_name(prefix.name + "_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["stats_json"] = prefix.with_name(prefix.name + "_stats.json")
    return paths
