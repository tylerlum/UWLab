"""Export source-mesh and PhysX-cooked SDF samples for a USD collision mesh.

This script loads a USD into Isaac Sim, creates a PhysX SDF shape view for one
authored SDF collision mesh, queries the cooked SDF on a regular grid, computes
an independent signed distance against the source triangle mesh, and writes the
comparison to an ``.npz`` file for Viser visualization.

The PhysX SDF values come from the runtime cooked collider:

    physics_sim_view.create_sdf_shape_view(...).get_sdf_and_gradients(...)

The source SDF is computed directly from the authored USD mesh with trimesh.
It is a diagnostic reference, not the exact PhysX internal representation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD = (
    REPO_ROOT
    / ".pretrained_checkpoints"
    / "SimToolReal"
    / "omnireset_assets"
    / "FurnitureBench"
    / "SquareLeg"
    / "square_leg_canonical.usd"
)
DEFAULT_OUT = REPO_ROOT / "outputs" / "sdf_comparison" / "canonical_leg_bolt_sdf.npz"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD, help="USD containing the SDF collision mesh.")
    parser.add_argument(
        "--sdf_prim",
        default="/square_leg/collisions/bolt",
        help="Mesh prim path inside the source USD, not the spawned /World path.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grid", type=int, default=42, help="Samples per axis for the query grid.")
    parser.add_argument("--padding", type=float, default=0.004, help="Metric padding around the source mesh bounds.")
    parser.add_argument(
        "--surface-band",
        type=float,
        default=0.003,
        help="Half-width in meters for the surface-band mask saved for visualization.",
    )
    parser.add_argument("--spawn_path", default="/World/Asset")
    parser.add_argument(
        "--graceful_close",
        action="store_true",
        help="Call simulation_app.close() instead of force-exiting after writing the npz.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


args_cli = _build_parser().parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch
import trimesh

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import omni.physics.tensors.impl.api as physx
from isaaclab.sim.utils.stage import get_current_stage_id
from pxr import Gf, Usd, UsdGeom


def _triangulate(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    faces: list[list[int]] = []
    cursor = 0
    for count in face_counts:
        polygon = face_indices[cursor : cursor + int(count)]
        cursor += int(count)
        if len(polygon) < 3:
            continue
        for i in range(1, len(polygon) - 1):
            faces.append([int(polygon[0]), int(polygon[i]), int(polygon[i + 1])])
    return np.asarray(faces, dtype=np.int64)


def _points_to_root_frame(points: np.ndarray, prim: Usd.Prim, root_prim: Usd.Prim) -> np.ndarray:
    xform_cache = UsdGeom.XformCache()
    prim_to_world = xform_cache.GetLocalToWorldTransform(prim)
    root_to_world = xform_cache.GetLocalToWorldTransform(root_prim)
    world_to_root = root_to_world.GetInverse()
    out = np.empty_like(points, dtype=np.float64)
    for i, point in enumerate(points):
        point_world = prim_to_world.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
        point_root = world_to_root.Transform(point_world)
        out[i] = (point_root[0], point_root[1], point_root[2])
    return out


def _load_source_mesh(usd_path: Path, sdf_prim_path: str) -> tuple[np.ndarray, np.ndarray, str]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"{usd_path} has no valid default prim.")
    prim = stage.GetPrimAtPath(sdf_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Could not find SDF prim {sdf_prim_path} in {usd_path}")
    if prim.GetTypeName() != "Mesh":
        raise RuntimeError(f"{sdf_prim_path} is not a Mesh prim.")
    approx = prim.GetAttribute("physics:approximation").Get() if prim.HasAttribute("physics:approximation") else None
    if approx != "sdf":
        raise RuntimeError(f"{sdf_prim_path} has physics:approximation={approx!r}, expected 'sdf'.")

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or face_counts is None or face_indices is None:
        raise RuntimeError(f"{sdf_prim_path} has incomplete mesh topology.")
    vertices = _points_to_root_frame(np.asarray(points, dtype=np.float64), prim, root_prim)
    faces = _triangulate(np.asarray(face_counts, dtype=np.int64), np.asarray(face_indices, dtype=np.int64))
    return vertices, faces, str(root_prim.GetPath())


def _make_grid(vertices: np.ndarray, grid: int, padding: float) -> tuple[np.ndarray, np.ndarray]:
    lo = vertices.min(axis=0) - padding
    hi = vertices.max(axis=0) + padding
    xs = [np.linspace(lo[i], hi[i], grid, dtype=np.float32) for i in range(3)]
    xx, yy, zz = np.meshgrid(xs[0], xs[1], xs[2], indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return points, np.stack([lo, hi], axis=0)


def _source_signed_distance(vertices: np.ndarray, faces: np.ndarray, points: np.ndarray) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # trimesh returns positive inside / negative outside.  PhysX follows the
    # common collision convention used in IsaacLab: negative inside, positive
    # outside.  Flip the sign for direct comparison.
    return -mesh.nearest.signed_distance(points).astype(np.float32)


def _spawn_usd(usd_path: Path, spawn_path: str, device: str) -> sim_utils.SimulationContext:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    prim_utils.create_prim("/World/Light", "DistantLight")
    cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    cfg.func(spawn_path, cfg)
    sim.reset()
    # Step a few frames so cooking/tensor views are ready.
    for _ in range(5):
        sim.step(render=False)
    return sim


def _query_physx_sdf(
    points: np.ndarray,
    spawned_sdf_prim_path: str,
    num_query_points: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    stage_id = get_current_stage_id()
    sim_view = physx.create_simulation_view("torch", stage_id)
    sim_view.set_subspace_roots("/")
    sdf_view = sim_view.create_sdf_shape_view(spawned_sdf_prim_path, num_query_points)
    query = torch.as_tensor(points, dtype=torch.float32, device=device).view(1, num_query_points, 3)
    values_and_gradients = sdf_view.get_sdf_and_gradients(query)
    values_and_gradients = values_and_gradients.detach().cpu().numpy()[0]
    gradients = values_and_gradients[:, :3].astype(np.float32)
    values = values_and_gradients[:, 3].astype(np.float32)
    return values, gradients


def main() -> None:
    usd_path = args_cli.usd.expanduser().resolve()
    output_path = args_cli.output.expanduser().resolve()
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)
    if args_cli.grid < 2:
        raise ValueError("--grid must be >= 2")

    vertices, faces, root_path = _load_source_mesh(usd_path, args_cli.sdf_prim)
    points, bounds = _make_grid(vertices, args_cli.grid, args_cli.padding)
    print(f"USD: {usd_path}")
    print(f"source root: {root_path}")
    print(f"SDF prim: {args_cli.sdf_prim}")
    print(f"source mesh: {len(vertices)} vertices, {len(faces)} triangles")
    print(f"query points: {len(points)} ({args_cli.grid}^3)")
    print(f"query bounds: {bounds[0]} -> {bounds[1]}")

    source_sdf = _source_signed_distance(vertices, faces, points)
    sim = _spawn_usd(usd_path, args_cli.spawn_path, args_cli.device)
    spawned_sdf_prim = f"{args_cli.spawn_path}{args_cli.sdf_prim[len(root_path):]}"
    print(f"spawned SDF prim: {spawned_sdf_prim}")
    physx_sdf, physx_gradients = _query_physx_sdf(points, spawned_sdf_prim, len(points), args_cli.device)

    finite = np.isfinite(physx_sdf) & np.isfinite(source_sdf)
    abs_diff = np.abs(physx_sdf[finite] - source_sdf[finite])
    print(f"source sdf min/max: {source_sdf.min():+.6f} / {source_sdf.max():+.6f}")
    print(f"physx sdf min/max:  {physx_sdf.min():+.6f} / {physx_sdf.max():+.6f}")
    print(f"abs diff p50/p95/max: {np.percentile(abs_diff, 50):.6f} / {np.percentile(abs_diff, 95):.6f} / {abs_diff.max():.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        usd_path=str(usd_path),
        source_root_path=root_path,
        sdf_prim_path=args_cli.sdf_prim,
        spawned_sdf_prim_path=spawned_sdf_prim,
        grid=args_cli.grid,
        padding=args_cli.padding,
        surface_band=args_cli.surface_band,
        bounds=bounds.astype(np.float32),
        mesh_vertices=vertices.astype(np.float32),
        mesh_faces=faces.astype(np.uint32),
        points=points.astype(np.float32),
        source_sdf=source_sdf.astype(np.float32),
        physx_sdf=physx_sdf.astype(np.float32),
        physx_gradients=physx_gradients.astype(np.float32),
    )
    print(f"Wrote {output_path}")
    sim.stop()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        if args_cli.graceful_close:
            simulation_app.close()
        sys.stdout.flush()
        sys.stderr.flush()
        if not args_cli.graceful_close:
            os._exit(exit_code)
