"""Visualize OmniReset FurnitureBench partial assembly poses in Viser.

This script intentionally avoids Isaac Sim. It loads the FurnitureBench tabletop
and leg USD meshes, plus the cached OmniReset ``partial_assemblies`` dataset,
then shows the stored leg root poses relative to the tabletop root.
"""

from __future__ import annotations

import argparse
import colorsys
import time
from pathlib import Path

import numpy as np
import viser

from view_furniturebench_leg_alignment import (
    DEFAULT_ASSET_ROOTS,
    LEG_ASSEMBLED_POS,
    LEG_ASSEMBLED_WXYZ,
    Q_ASSET_POLICY_WXYZ,
    TABLE_ASSEMBLED_POS,
    TABLE_ASSEMBLED_WXYZ,
    _add_parts,
    _compose,
    _find_asset_root,
    _fmt_vec,
    _load_usd_meshes,
    _normalize_wxyz,
    _wxyz_to_rot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_NPZ = (
    REPO_ROOT
    / ".pretrained_checkpoints"
    / "SimToolReal"
    / "omnireset_datasets"
    / "Resets"
    / "SquareLeg__SquareTableTop"
    / "partial_assemblies.npz"
)
DEFAULT_DATASET_PT = DEFAULT_DATASET_NPZ.with_suffix(".pt")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset_root", type=Path, default=None)
    parser.add_argument("--leg_usd", type=Path, default=None, help="Explicit SquareLeg USD to visualize.")
    parser.add_argument("--table_usd", type=Path, default=None, help="Explicit SquareTableTop USD to visualize.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to partial_assemblies.npz or partial_assemblies.pt.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--initial_index", type=int, default=0)
    parser.add_argument("--show_collisions", action="store_true")
    parser.add_argument("--point_size", type=float, default=0.004)
    parser.add_argument(
        "--max_ghost_meshes",
        type=int,
        default=24,
        help="Maximum number of translucent full-leg meshes that can be shown at once.",
    )
    parser.add_argument(
        "--show_all_axes",
        action="store_true",
        help="Start with all 299 stored pose axes visible.",
    )
    return parser


def _default_dataset_path() -> Path:
    if DEFAULT_DATASET_NPZ.exists():
        return DEFAULT_DATASET_NPZ
    if DEFAULT_DATASET_PT.exists():
        return DEFAULT_DATASET_PT
    raise FileNotFoundError(
        "Could not find partial_assemblies.npz or partial_assemblies.pt. "
        f"Expected {DEFAULT_DATASET_NPZ} or {DEFAULT_DATASET_PT}."
    )


def _load_partial_assembly_poses(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = dataset_path.expanduser().resolve()
    if dataset_path.suffix == ".npz":
        data = np.load(dataset_path)
        return (
            np.asarray(data["relative_position"], dtype=np.float64),
            np.asarray(data["relative_orientation"], dtype=np.float64),
        )

    fallback_npz = dataset_path.with_suffix(".npz")
    if fallback_npz.exists():
        data = np.load(fallback_npz)
        return (
            np.asarray(data["relative_position"], dtype=np.float64),
            np.asarray(data["relative_orientation"], dtype=np.float64),
        )

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            f"{dataset_path} is a PyTorch .pt file, but torch is not installed in this Python env. "
            f"Create {fallback_npz.name} once from the .pt file, or run this viewer in an env with torch."
        ) from exc

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    return (
        data["relative_position"].detach().cpu().numpy().astype(np.float64),
        data["relative_orientation"].detach().cpu().numpy().astype(np.float64),
    )


def _index_colors(count: int, value: float = 0.95) -> np.ndarray:
    colors = np.empty((count, 3), dtype=np.uint8)
    denom = max(1, count - 1)
    for idx in range(count):
        rgb = colorsys.hsv_to_rgb(idx / denom, 0.72, value)
        colors[idx] = np.asarray(rgb, dtype=np.float64) * 255.0
    return colors


def _tip_poses(root_pos: np.ndarray, root_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tip_pos = np.empty_like(root_pos)
    tip_wxyz = np.empty_like(root_wxyz)
    for idx, (pos, wxyz) in enumerate(zip(root_pos, root_wxyz, strict=True)):
        tip_pos[idx], tip_wxyz[idx] = _compose(pos, wxyz, LEG_ASSEMBLED_POS, LEG_ASSEMBLED_WXYZ)
    return tip_pos, tip_wxyz


def _line_segments(root_pos: np.ndarray, tip_pos: np.ndarray) -> np.ndarray:
    return np.stack([root_pos, tip_pos], axis=1).astype(np.float32)


def _selected_stats(index: int, root_pos: np.ndarray, root_wxyz: np.ndarray, tip_pos: np.ndarray, tip_wxyz: np.ndarray) -> str:
    rel_xyz = tip_pos[index] - TABLE_ASSEMBLED_POS
    rel_rot = _wxyz_to_rot(TABLE_ASSEMBLED_WXYZ).inv() * _wxyz_to_rot(tip_wxyz[index])
    rel_rpy = rel_rot.as_euler("xyz", degrees=True)
    return (
        "## Selected Partial Pose\n"
        f"- index: `{index}`\n"
        f"- leg root pos: `{_fmt_vec(root_pos[index])}`\n"
        f"- leg root wxyz: `{_fmt_vec(root_wxyz[index])}`\n"
        f"- tip minus hole xyz: `{_fmt_vec(rel_xyz)}` m\n"
        f"- tip position error: `{np.linalg.norm(rel_xyz):.5f}` m\n"
        f"- tip roll/pitch/yaw vs hole: `{_fmt_vec(rel_rpy)}` deg"
    )


def main() -> None:
    args = _build_parser().parse_args()
    asset_root = _find_asset_root(args.asset_root)
    table_usd = args.table_usd if args.table_usd is not None else asset_root / "SquareTableTop" / "square_table_top.usd"
    leg_usd = args.leg_usd if args.leg_usd is not None else asset_root / "SquareLeg" / "square_leg.usd"
    dataset_path = args.dataset if args.dataset is not None else _default_dataset_path()

    table_root, table_parts = _load_usd_meshes(table_usd.expanduser().resolve())
    leg_root, leg_parts = _load_usd_meshes(leg_usd.expanduser().resolve())
    root_pos, root_wxyz = _load_partial_assembly_poses(dataset_path)
    root_wxyz = np.asarray([_normalize_wxyz(quat) for quat in root_wxyz], dtype=np.float64)
    tip_pos, tip_wxyz = _tip_poses(root_pos, root_wxyz)
    count = int(root_pos.shape[0])
    initial_index = int(args.initial_index) % count
    colors = _index_colors(count)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=0.24,
        height=0.24,
        plane="xy",
        cell_size=0.01,
        section_size=0.05,
        position=(0.0, 0.0, TABLE_ASSEMBLED_POS[2]),
    )
    server.scene.add_frame("/world", axes_length=0.04, axes_radius=0.0015)
    server.scene.add_frame("/table", axes_length=0.045, axes_radius=0.0015)
    server.scene.add_frame(
        "/table/assembled_hole",
        position=TABLE_ASSEMBLED_POS,
        wxyz=TABLE_ASSEMBLED_WXYZ,
        axes_length=0.040,
        axes_radius=0.0012,
    )
    table_handles = _add_parts(server, "/table", table_parts, "table", args.show_collisions)

    selected_frame = server.scene.add_frame(
        "/selected_leg",
        position=root_pos[initial_index],
        wxyz=root_wxyz[initial_index],
        axes_length=0.045,
        axes_radius=0.0015,
    )
    server.scene.add_frame(
        "/selected_leg/policy_frame",
        position=(0.0, 0.0, 0.0),
        wxyz=Q_ASSET_POLICY_WXYZ,
        axes_length=0.055,
        axes_radius=0.0014,
    )
    server.scene.add_frame(
        "/selected_leg/assembled_tip",
        position=LEG_ASSEMBLED_POS,
        wxyz=LEG_ASSEMBLED_WXYZ,
        axes_length=0.035,
        axes_radius=0.0012,
    )
    selected_handles = _add_parts(server, "/selected_leg", leg_parts, "leg", args.show_collisions)

    root_cloud = server.scene.add_point_cloud(
        "/partial_pose_roots",
        points=root_pos.astype(np.float32),
        colors=colors,
        point_size=float(args.point_size),
        point_shape="circle",
    )
    tip_cloud = server.scene.add_point_cloud(
        "/partial_pose_tips",
        points=tip_pos.astype(np.float32),
        colors=np.clip(colors.astype(np.float32) * 0.75 + 60.0, 0, 255).astype(np.uint8),
        point_size=float(args.point_size) * 0.8,
        point_shape="circle",
    )
    root_tip_lines = server.scene.add_line_segments(
        "/partial_pose_root_to_tip_lines",
        points=_line_segments(root_pos, tip_pos),
        colors=np.asarray([185, 190, 200], dtype=np.uint8),
        line_width=1.0,
        visible=False,
    )

    pose_frames = []
    for idx in range(count):
        pose_frames.append(
            server.scene.add_frame(
                f"/partial_pose_axes/{idx:03d}",
                position=root_pos[idx],
                wxyz=root_wxyz[idx],
                axes_length=0.014,
                axes_radius=0.00045,
                visible=bool(args.show_all_axes),
            )
        )

    max_ghost_meshes = max(1, int(args.max_ghost_meshes))
    ghost_frames = []
    ghost_mesh_handles: list[list[viser.MeshHandle]] = []
    for slot in range(max_ghost_meshes):
        parent = f"/ghost_mesh_range/slot_{slot:02d}"
        frame = server.scene.add_frame(
            parent,
            axes_length=0.020,
            axes_radius=0.0006,
            visible=False,
        )
        handles = _add_parts(server, parent, leg_parts, "leg", False, ghost=True)
        for handle in handles:
            handle.visible = False
        ghost_frames.append(frame)
        ghost_mesh_handles.append(handles)

    server.gui.add_markdown(
        "# OmniReset Partial Assemblies\n"
        f"Loaded `{count}` FurnitureBench SquareLeg/SquareTableTop partial assembly poses.\n\n"
        "The colored points are stored leg root poses and thread-tip poses relative to the tabletop root. "
        "The selected full mesh shows the exact raw SquareLeg USD pose from the dataset."
    )
    selected_index = server.gui.add_slider("Selected index", 0, count - 1, 1, initial_index)
    previous_button = server.gui.add_button("Previous")
    next_button = server.gui.add_button("Next")
    show_selected_mesh = server.gui.add_checkbox("Show selected mesh", True)
    show_collisions = server.gui.add_checkbox("Show collision meshes", args.show_collisions)
    show_root_points = server.gui.add_checkbox("Show all root points", True)
    show_tip_points = server.gui.add_checkbox("Show all tip points", True)
    show_root_tip_lines = server.gui.add_checkbox("Show root-tip lines", False)
    show_all_axes = server.gui.add_checkbox("Show all pose axes", bool(args.show_all_axes))
    server.gui.add_markdown("## Ghost Mesh Range")
    show_ghost_meshes = server.gui.add_checkbox("Show ghost range meshes", False)
    show_ghost_axes = server.gui.add_checkbox("Show ghost range axes", True)
    ghost_start = server.gui.add_slider("Ghost start index", 0, count - 1, 1, 0)
    ghost_count = server.gui.add_slider("Ghost count", 1, max_ghost_meshes, 1, min(12, max_ghost_meshes))
    ghost_stride = server.gui.add_slider("Ghost stride", 1, max(1, count - 1), 1, 1)
    status = server.gui.add_markdown("")

    def update_selected(index: int, *, sync_slider: bool = True) -> None:
        idx = int(index) % count
        if sync_slider and int(selected_index.value) != idx:
            selected_index.value = idx
        selected_frame.position = tuple(float(x) for x in root_pos[idx])
        selected_frame.wxyz = tuple(float(x) for x in root_wxyz[idx])
        status.content = _selected_stats(idx, root_pos, root_wxyz, tip_pos, tip_wxyz)

    def update_collision_visibility() -> None:
        for handle in table_handles + selected_handles:
            if "/collisions/" in handle.name:
                handle.visible = bool(show_collisions.value)

    def update_selected_visibility() -> None:
        selected_frame.visible = bool(show_selected_mesh.value)
        for handle in selected_handles:
            if "/collisions/" in handle.name:
                handle.visible = bool(show_selected_mesh.value and show_collisions.value)
            else:
                handle.visible = bool(show_selected_mesh.value)

    def update_ghost_range() -> None:
        start = int(ghost_start.value) % count
        stride = max(1, int(ghost_stride.value))
        visible_count = min(max_ghost_meshes, max(1, int(ghost_count.value)))
        for slot in range(max_ghost_meshes):
            show_slot = bool(show_ghost_meshes.value) and slot < visible_count
            idx = (start + slot * stride) % count
            ghost_frames[slot].position = tuple(float(x) for x in root_pos[idx])
            ghost_frames[slot].wxyz = tuple(float(x) for x in root_wxyz[idx])
            ghost_frames[slot].visible = show_slot and bool(show_ghost_axes.value)
            for handle in ghost_mesh_handles[slot]:
                handle.visible = show_slot

    @selected_index.on_update
    def _(_) -> None:
        update_selected(int(selected_index.value), sync_slider=False)

    @previous_button.on_click
    def _(_) -> None:
        update_selected(int(selected_index.value) - 1)

    @next_button.on_click
    def _(_) -> None:
        update_selected(int(selected_index.value) + 1)

    @show_selected_mesh.on_update
    def _(_) -> None:
        update_selected_visibility()

    @show_collisions.on_update
    def _(_) -> None:
        update_collision_visibility()
        update_selected_visibility()

    @show_root_points.on_update
    def _(_) -> None:
        root_cloud.visible = bool(show_root_points.value)

    @show_tip_points.on_update
    def _(_) -> None:
        tip_cloud.visible = bool(show_tip_points.value)

    @show_root_tip_lines.on_update
    def _(_) -> None:
        root_tip_lines.visible = bool(show_root_tip_lines.value)

    @show_all_axes.on_update
    def _(_) -> None:
        for frame in pose_frames:
            frame.visible = bool(show_all_axes.value)

    for handle in (show_ghost_meshes, show_ghost_axes, ghost_start, ghost_count, ghost_stride):
        handle.on_update(lambda _: update_ghost_range())

    update_selected(initial_index)
    update_collision_visibility()
    update_selected_visibility()
    update_ghost_range()

    print(f"Serving OmniReset partial assembly viewer at http://{args.host}:{args.port}")
    print(f"Dataset: {dataset_path.expanduser().resolve()}")
    print(f"Partial assembly poses: {count}")
    print(f"Table USD: {table_usd}")
    print(f"Leg USD: {leg_usd}")
    print(f"Table root: {table_root}")
    print(f"Leg root: {leg_root}")
    try:
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
