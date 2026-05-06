"""Visualize source-mesh vs PhysX-cooked SDF samples in Viser."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = REPO_ROOT / "outputs" / "sdf_comparison" / "canonical_leg_bolt_sdf_grid44.npz"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", nargs="?", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--band", type=float, default=None, help="Meters around zero level set to show.")
    parser.add_argument("--clip", type=float, default=0.004, help="SDF value used for color saturation.")
    parser.add_argument("--diff_clip", type=float, default=0.002, help="Abs diff value used for color saturation.")
    parser.add_argument("--max_points", type=int, default=60000)
    parser.add_argument("--point_size", type=float, default=0.0012)
    parser.add_argument("--panel_spacing", type=float, default=0.075)
    parser.add_argument("--show_all", action="store_true", help="Show all grid points instead of near-surface band.")
    return parser


def _signed_sdf_colors(values: np.ndarray, clip: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    t = np.clip(values / max(clip, 1e-9), -1.0, 1.0)
    colors = np.empty((len(values), 3), dtype=np.uint8)

    inside = t < 0.0
    outside = ~inside
    white = np.array([245.0, 245.0, 245.0])
    blue = np.array([60.0, 115.0, 255.0])
    red = np.array([255.0, 75.0, 55.0])

    a = (-t[inside])[:, None]
    colors[inside] = np.clip((1.0 - a) * white + a * blue, 0, 255).astype(np.uint8)
    a = t[outside][:, None]
    colors[outside] = np.clip((1.0 - a) * white + a * red, 0, 255).astype(np.uint8)
    return colors


def _diff_colors(values: np.ndarray, clip: float) -> np.ndarray:
    t = np.clip(np.asarray(values, dtype=np.float32) / max(clip, 1e-9), 0.0, 1.0)
    lo = np.array([245.0, 245.0, 245.0])
    hi = np.array([130.0, 0.0, 160.0])
    return np.clip((1.0 - t[:, None]) * lo + t[:, None] * hi, 0, 255).astype(np.uint8)


def _subsample(mask: np.ndarray, max_points: int, seed: int = 17) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if max_points > 0 and len(idx) > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
    return idx


def _add_panel(
    server: viser.ViserServer,
    name: str,
    position: tuple[float, float, float],
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    point_size: float,
    mesh_color: tuple[int, int, int],
) -> None:
    server.scene.add_frame(name, position=position, axes_length=0.018, axes_radius=0.0007)
    server.scene.add_mesh_simple(
        f"{name}/source_collision_mesh",
        vertices=mesh_vertices.astype(np.float32),
        faces=mesh_faces.astype(np.uint32),
        color=mesh_color,
        opacity=0.16,
        side="double",
        flat_shading=False,
    )
    server.scene.add_point_cloud(
        f"{name}/sdf_samples",
        points=points.astype(np.float32),
        colors=colors,
        point_size=point_size,
        point_shape="circle",
    )


def _stats(values: np.ndarray) -> str:
    return (
        f"min `{values.min():+.5f}`, p50 `{np.percentile(values, 50):+.5f}`, "
        f"p95 `{np.percentile(values, 95):+.5f}`, max `{values.max():+.5f}`"
    )


def main() -> None:
    args = _build_parser().parse_args()
    npz_path = args.npz.expanduser().resolve()
    data = np.load(npz_path)
    points = data["points"]
    source_sdf = data["source_sdf"]
    physx_sdf = data["physx_sdf"]
    abs_diff = np.abs(source_sdf - physx_sdf)
    mesh_vertices = data["mesh_vertices"]
    mesh_faces = data["mesh_faces"]
    band = float(args.band if args.band is not None else data.get("surface_band", 0.003))

    if args.show_all:
        mask = np.ones(len(points), dtype=bool)
    else:
        mask = (np.abs(source_sdf) <= band) | (np.abs(physx_sdf) <= band)
    idx = _subsample(mask, args.max_points)

    sampled_points = points[idx]
    source_colors = _signed_sdf_colors(source_sdf[idx], args.clip)
    physx_colors = _signed_sdf_colors(physx_sdf[idx], args.clip)
    diff_colors = _diff_colors(abs_diff[idx], args.diff_clip)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    spacing = float(args.panel_spacing)

    _add_panel(
        server,
        "/source_mesh_sdf",
        (0.0, -spacing, 0.0),
        mesh_vertices,
        mesh_faces,
        sampled_points,
        source_colors,
        args.point_size,
        (120, 140, 165),
    )
    _add_panel(
        server,
        "/physx_cooked_sdf",
        (0.0, 0.0, 0.0),
        mesh_vertices,
        mesh_faces,
        sampled_points,
        physx_colors,
        args.point_size,
        (120, 165, 130),
    )
    _add_panel(
        server,
        "/abs_difference",
        (0.0, spacing, 0.0),
        mesh_vertices,
        mesh_faces,
        sampled_points,
        diff_colors,
        args.point_size,
        (165, 135, 170),
    )

    server.gui.add_markdown(
        "# SDF Comparison\n"
        f"`{npz_path}`\n\n"
        "Three panels are offset along world Y:\n"
        "- `/source_mesh_sdf`: independent signed distance to the authored collision mesh.\n"
        "- `/physx_cooked_sdf`: PhysX runtime SDF queried with `create_sdf_shape_view`.\n"
        "- `/abs_difference`: absolute difference at the same query points.\n\n"
        "Signed SDF colors: blue = inside/negative, white = near zero, red = outside/positive. "
        "Difference colors: white = close, purple = larger mismatch.\n\n"
        f"Shown points: `{len(idx)}` / `{len(points)}`. Surface band: `{band:.4f}` m. "
        f"Color clip: `{args.clip:.4f}` m. Difference clip: `{args.diff_clip:.4f}` m.\n\n"
        f"Source SDF: {_stats(source_sdf)}\n\n"
        f"PhysX SDF: {_stats(physx_sdf)}\n\n"
        f"Abs diff: {_stats(abs_diff)}"
    )

    print(f"Serving SDF comparison viewer at http://{args.host}:{args.port}")
    print(f"Loaded: {npz_path}")
    print(f"Showing {len(idx)} / {len(points)} query points")
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
