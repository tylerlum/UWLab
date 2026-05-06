"""Rewrite the FurnitureBench square-leg USD into the SimToolReal policy frame.

The current UWLab FurnitureBench SquareLeg USD uses a root frame where the
handle-to-thread direction is USD ``-Z``.  SimToolReal wants that long axis to
be object ``+X``.  This script copies the USD and rewrites every mesh point so
that the output USD root frame is the policy/canonical frame:

    canonical +X = old USD -Z = handle -> screw threads
    canonical +Y = old USD +X
    canonical +Z = old USD -Y

It preserves the existing USD prim structure and authored physics attributes,
including the mixed collision setup:

    /square_leg/collisions/leg   physics:approximation = convexHull
    /square_leg/collisions/bolt  physics:approximation = sdf

This is intentionally an asset-authoring example.  Any code that consumes the
canonical output should stop applying the old virtual policy-frame transform,
otherwise the leg will be rotated twice.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / ".pretrained_checkpoints"
    / "SimToolReal"
    / "omnireset_assets"
    / "FurnitureBench"
    / "SquareLeg"
    / "square_leg.usd"
)

# Columns are canonical-frame axes expressed in the old SquareLeg USD root.
R_OLD_CANONICAL = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# Useful reference from OmniReset metadata.  It is not written automatically
# because downstream metadata conventions should be reviewed deliberately.
LEG_ASSEMBLED_POS_OLD = np.array([0.0, 0.0, -0.056658], dtype=np.float64)
LEG_BOTTOM_POS_OLD = np.array([0.0, 0.0, -0.056658], dtype=np.float64)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input leg USD. Defaults to {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output USD. Defaults to '<input stem>_canonical.usd' next to the input.",
    )
    parser.add_argument(
        "--new-origin-shift-in-canonical-frame",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help=(
            "Vector from old root origin to new root origin, expressed in the "
            "canonical frame. Example: '-0.01 0 0' moves the new origin 1 cm "
            "toward canonical -X, i.e. back toward the handle."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output USD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the edits that would be made without writing the output.",
    )
    return parser


def _as_array(points) -> np.ndarray:
    return np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float64)


def _vec3f_array(values: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in values])


def _identity_matrix4() -> Gf.Matrix4d:
    return Gf.Matrix4d(1.0)


def _validate_simple_xforms(stage: Usd.Stage, root_prim: Usd.Prim) -> None:
    """Fail if child Xforms are non-identity.

    The current UWLab SquareLeg USD has identity transforms below the root, so
    rewriting mesh point arrays is enough.  If a future asset has meaningful
    nested Xforms, this script should be extended to bake those transforms
    rather than silently producing a wrong asset.
    """

    for prim in stage.Traverse():
        if prim == root_prim:
            continue
        if not prim.GetPath().HasPrefix(root_prim.GetPath()):
            continue
        if not prim.IsA(UsdGeom.Xformable):
            continue
        xformable = UsdGeom.Xformable(prim)
        local_xform_result = xformable.GetLocalTransformation()
        if isinstance(local_xform_result, tuple):
            local_xform = local_xform_result[0]
            resets_stack = bool(local_xform_result[1]) if len(local_xform_result) > 1 else False
        else:
            local_xform = local_xform_result
            resets_stack = bool(xformable.GetResetXformStack())
        if resets_stack or not Gf.IsClose(local_xform, _identity_matrix4(), 1e-12):
            raise RuntimeError(
                f"Refusing to rewrite {prim.GetPath()} because it has a non-identity local transform. "
                "Bake nested transforms first or extend this script to transform through XformCache."
            )


def _transform_points(points_old: np.ndarray, origin_shift_canonical: np.ndarray) -> np.ndarray:
    # Row-vector form of: p_canonical = R_old_canonical.T @ p_old - origin_shift_canonical.
    return points_old @ R_OLD_CANONICAL - origin_shift_canonical


def _transform_normals(normals_old: np.ndarray) -> np.ndarray:
    normals = normals_old @ R_OLD_CANONICAL
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-12)


def _mesh_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


def _format_vec(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.6f}" for v in values) + "]"


def _rewrite_meshes(stage: Usd.Stage, origin_shift_canonical: np.ndarray, dry_run: bool) -> int:
    count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        points_old = _as_array(points_attr.Get() or [])
        if points_old.size == 0:
            continue

        points_new = _transform_points(points_old, origin_shift_canonical)
        old_lo, old_hi = _mesh_bounds(points_old)
        new_lo, new_hi = _mesh_bounds(points_new)
        approx = prim.GetAttribute("physics:approximation").Get() if prim.HasAttribute("physics:approximation") else None
        collision = (
            prim.GetAttribute("physics:collisionEnabled").Get() if prim.HasAttribute("physics:collisionEnabled") else None
        )
        print(
            f"{prim.GetPath()} collision={collision} approximation={approx}\n"
            f"  old bounds {_format_vec(old_lo)} -> {_format_vec(old_hi)}\n"
            f"  new bounds {_format_vec(new_lo)} -> {_format_vec(new_hi)}"
        )

        if not dry_run:
            points_attr.Set(_vec3f_array(points_new))
            mesh.GetExtentAttr().Set(_vec3f_array(np.vstack([new_lo, new_hi])))

            normals_attr = mesh.GetNormalsAttr()
            normals_old = normals_attr.Get()
            if normals_old:
                normals_new = _transform_normals(_as_array(normals_old))
                normals_attr.Set(_vec3f_array(normals_new))

        count += 1
    return count


def main() -> None:
    args = _build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_canonical{input_path.suffix}")
    )
    origin_shift_canonical = np.asarray(args.new_origin_shift_in_canonical_frame, dtype=np.float64)

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("Canonical frame:")
    print("  +X = old -Z = handle -> screw threads")
    print("  +Y = old +X")
    print("  +Z = old -Y")
    print(f"New-origin shift in canonical frame: {_format_vec(origin_shift_canonical)} m")

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        stage_path = output_path
    else:
        stage_path = input_path

    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {stage_path}")
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"{stage_path} has no valid default prim.")

    print(f"Default prim: {root_prim.GetPath()}")
    _validate_simple_xforms(stage, root_prim)
    mesh_count = _rewrite_meshes(stage, origin_shift_canonical, args.dry_run)

    assembled_new = _transform_points(LEG_ASSEMBLED_POS_OLD.reshape(1, 3), origin_shift_canonical)[0]
    bottom_new = _transform_points(LEG_BOTTOM_POS_OLD.reshape(1, 3), origin_shift_canonical)[0]
    print("\nSuggested position metadata updates for this canonical leg frame:")
    print(f"  assembled_offset.pos: {_format_vec(assembled_new)}")
    print(f"  bottom_offset.pos:    {_format_vec(bottom_new)}")
    print("  Review quaternions and downstream success-frame conventions before replacing metadata.")
    print("  SimToolReal object_policy_frame_quat_wxyz should become identity for this canonical asset.")

    if not args.dry_run:
        stage.GetRootLayer().Save()
        print(f"\nWrote {mesh_count} transformed mesh prim(s) to {output_path}")
    else:
        print(f"\nDry run only; would transform {mesh_count} mesh prim(s).")


if __name__ == "__main__":
    main()
