"""Roll out the pretrained SimToolReal policy through the isaacsimenvs task.

This driver intentionally uses ``Isaacsimenvs-SimToolReal-Direct-v0`` for
observation construction, action scaling, joint target application, PD gains,
materials, and scene setup. It only configures the object pool and scripted
goal poses for deployment-style rollouts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RL_GAMES = REPO_ROOT / "rl_games"
if LOCAL_RL_GAMES.exists():
    sys.path.insert(0, str(LOCAL_RL_GAMES))
sys.path.insert(0, str(REPO_ROOT))

POLICY_DIR = REPO_ROOT / ".pretrained_checkpoints" / "SimToolReal" / "pretrained_policy"
FURNITUREBENCH_ASSET_ROOTS = (
    REPO_ROOT / "assets" / "usd" / "furniturebench",
    REPO_ROOT
    / ".pretrained_checkpoints"
    / "SimToolReal"
    / "omnireset_assets"
    / "FurnitureBench",
)

# Columns are policy-frame axes expressed in the SquareLeg USD root frame:
# policy +x is USD -z, i.e. from the cuboidal handle toward the screw threads.
R_USD_POLICY_LEG = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

FURNITUREBENCH_TABLE_HOLES = np.array(
    [
        [0.0562, 0.0562, 0.0020],
        [-0.0562, 0.0562, 0.0020],
        [-0.0562, -0.0563, 0.0020],
        [0.0562, -0.0563, 0.0020],
    ],
    dtype=np.float32,
)

# OmniReset FurnitureBench metadata assembled offsets.  With the raw SquareLeg
# USD upright at identity, the final assembled leg root is:
# table_root + table_assembled - leg_assembled.
FURNITUREBENCH_TABLE_ASSEMBLED_OFFSET = np.array(
    [0.05625, 0.05625, -0.009435], dtype=np.float32
)
FURNITUREBENCH_LEG_ASSEMBLED_OFFSET = np.array(
    [0.0, 0.0, -0.056658], dtype=np.float32
)
FURNITUREBENCH_LEG_PREINSERT_Z_OFFSET_FROM_FINAL = 0.015427
FURNITUREBENCH_LEG_PREINSERT_YAW_OFFSET_DEG = 20.73494

# Graspable cuboidal handle bbox in policy frame, encoded by SimToolReal as
# metric dimensions multiplied by 25.
FURNITUREBENCH_LEG_OBJECT_SCALE = (
    np.array([[0.06255, 0.03017705, 0.03017705]], dtype=np.float32) * 25.0
)

# Matches isaacgymenvs/utils/observation_action_utils_sharpa.py and
# isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py.
POLICY_KEYPOINT_CORNERS = np.array(
    [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]],
    dtype=np.float32,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("claw_hammer", "furniturebench_leg"),
        default="claw_hammer",
    )
    parser.add_argument("--object_category", default="hammer")
    parser.add_argument("--object_name", default="claw_hammer")
    parser.add_argument("--task_name", default="swing_down")
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--rl_device", default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for rollout-script sampling, including randomized FurnitureBench leg starts.",
    )
    parser.add_argument("--checkpoint", default=str(POLICY_DIR / "model.pth"))
    parser.add_argument("--config", default=str(POLICY_DIR / "config.yaml"))
    parser.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "outputs" / "simtoolreal_isaacsimenvs"),
    )
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--frame_every", type=int, default=2)
    parser.add_argument("--render_each_step", action="store_true")
    goal_viz_group = parser.add_mutually_exclusive_group()
    goal_viz_group.add_argument(
        "--show_goal_viz",
        dest="goal_viz_visible",
        action="store_true",
        default=True,
        help="Render the goal pose object.",
    )
    goal_viz_group.add_argument(
        "--hide_goal_viz",
        dest="goal_viz_visible",
        action="store_false",
        help="Hide the goal pose object from viewer/video while keeping it available for observations.",
    )
    parser.add_argument("--keypoint_tolerance", type=float, default=0.015)
    parser.add_argument("--success_steps", type=int, default=10)
    parser.add_argument(
        "--robot_control_mode",
        choices=("policy", "hold_current", "hold_default"),
        default="policy",
        help="Run policy actions normally, or hold fixed Lab-order joint targets.",
    )
    parser.add_argument(
        "--object_drive_mode",
        choices=("policy", "teleport_trajectory"),
        default="policy",
        help="Run normal policy-driven object motion, or script object poses through the planned trajectory.",
    )
    parser.add_argument("--teleport_interval_s", type=float, default=1.0)
    parser.add_argument("--teleport_waypoints_per_segment", type=int, default=1)
    parser.add_argument(
        "--teleport_write_every_step",
        action="store_true",
        help="Write the current waypoint pose every policy step instead of only at waypoint intervals.",
    )
    parser.add_argument(
        "--teleport_interpolate_between_waypoints",
        action="store_true",
        help="When writing every step, interpolate between adjacent waypoints instead of holding each waypoint.",
    )
    parser.add_argument(
        "--teleport_skip_start_pose",
        action="store_true",
        help="Teleport through goal waypoints only, skipping the scripted initial object pose.",
    )
    parser.add_argument("--teleport_keep_velocity", action="store_true")
    parser.add_argument("--ignore_dones", action="store_true")
    parser.add_argument(
        "--continue_after_all_goals",
        action="store_true",
        help="Keep simulating on the final goal after the sequence is completed.",
    )

    parser.add_argument("--leg_hole_index", type=int, default=0, choices=range(4))
    parser.add_argument(
        "--leg_goal_sequence",
        choices=("dense", "final_only", "preinsert_final", "pingpong"),
        default="dense",
        help="FurnitureBench leg goal sequence to expose to the policy or teleport driver.",
    )
    parser.add_argument("--leg_hover_height", type=float, default=0.12)
    parser.add_argument(
        "--leg_preinsert_height",
        type=float,
        default=None,
        help=(
            "Pre-insert leg root height above the approximate top-hole point. "
            "If unset, use --leg_preinsert_final_z_offset above the final assembled pose."
        ),
    )
    parser.add_argument(
        "--leg_preinsert_final_z_offset",
        type=float,
        default=FURNITUREBENCH_LEG_PREINSERT_Z_OFFSET_FROM_FINAL,
        help="Pre-insert leg root Z offset above the final assembled root pose.",
    )
    parser.add_argument("--leg_insert_height", type=float, default=0.038)
    parser.add_argument(
        "--leg_use_omnireset_final_height",
        action="store_true",
        help="Use OmniReset metadata assembled offsets for the final insert root height.",
    )
    parser.add_argument(
        "--leg_preinsert_yaw_offset_deg",
        type=float,
        default=FURNITUREBENCH_LEG_PREINSERT_YAW_OFFSET_DEG,
        help="Raw upright-leg +Z yaw offset for the pre-insert pose; positive is CCW from above.",
    )
    parser.add_argument(
        "--leg_final_yaw_offset_deg",
        type=float,
        default=0.0,
        help="Raw upright-leg +Z yaw offset for the final insert pose; positive is CCW from above.",
    )
    parser.add_argument(
        "--leg_pingpong_cycles",
        type=int,
        default=20,
        help="Number of preinsert/final repetitions for --leg_goal_sequence pingpong.",
    )
    parser.add_argument("--leg_descend_steps", type=int, default=4)
    parser.add_argument(
        "--leg_spin_mode",
        choices=("interpolate", "after_insert", "helical"),
        default="interpolate",
    )
    parser.add_argument(
        "--leg_physics_profile",
        choices=("simtoolreal", "omnireset"),
        default="simtoolreal",
        help=(
            "Keep the default SimToolReal physics profile, or use the "
            "high-contact OmniReset leg-task PhysX/mass/friction settings."
        ),
    )
    parser.add_argument(
        "--leg_spin_turns",
        type=float,
        default=1.0,
        help="Positive turns spin clockwise when looking down the hole, i.e. about local policy +x.",
    )
    parser.add_argument("--leg_spin_steps", type=int, default=8)
    parser.add_argument("--leg_fixture_clearance", type=float, default=0.002)
    parser.add_argument(
        "--leg_randomize_start",
        action="store_true",
        help="Sample the FurnitureBench leg initial pose instead of using the fixed scripted start.",
    )
    parser.add_argument(
        "--leg_random_start_xy_center",
        nargs=2,
        type=float,
        default=(0.10, 0.08),
        metavar=("X", "Y"),
        help="Center of the randomized leg start XY range in env/world frame.",
    )
    parser.add_argument(
        "--leg_random_start_xy_range",
        nargs=2,
        type=float,
        default=(0.025, 0.025),
        metavar=("DX", "DY"),
        help="Uniform half-widths for randomized leg start XY. Increase to cover more of the table.",
    )
    parser.add_argument(
        "--leg_random_start_z_offset_from_hole",
        type=float,
        default=0.18,
        help="Nominal randomized leg root Z as approximate hole-top Z plus this offset.",
    )
    parser.add_argument(
        "--leg_random_start_z_range",
        type=float,
        default=0.01,
        help="Uniform half-width for randomized leg start Z.",
    )
    parser.add_argument(
        "--leg_random_start_yaw_center_deg",
        type=float,
        default=0.0,
        help="Center yaw for randomized leg starts, about raw/world +Z.",
    )
    parser.add_argument(
        "--leg_random_start_yaw_range_deg",
        type=float,
        default=25.0,
        help="Uniform half-width for randomized leg start yaw in degrees.",
    )
    parser.add_argument(
        "--leg_random_start_roll_pitch_range_deg",
        nargs=2,
        type=float,
        default=(10.0, 10.0),
        metavar=("ROLL", "PITCH"),
        help="Uniform half-widths for randomized roll/pitch in degrees; set 0 0 for upright starts.",
    )
    return parser


def _launch_app():
    from isaaclab.app import AppLauncher

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.record_video:
        args.enable_cameras = True
    app = AppLauncher(args).app
    return app, args


_app, _args = _launch_app()


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return quat_xyzw[[3, 0, 1, 2]]


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_wxyz[[1, 2, 3, 0]]


def pose_xyzw_to_wxyz(pose: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    pose = np.asarray(pose, dtype=np.float32)
    q = xyzw_to_wxyz(pose[3:7])
    return (
        float(pose[0]),
        float(pose[1]),
        float(pose[2]),
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    )


def pose_wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).copy()
    pose[3:7] = wxyz_to_xyzw(pose[3:7])
    return pose


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (R.from_quat(a) * R.from_quat(b)).as_quat().astype(np.float32)


def _quat_matrix_xyzw(matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(matrix).as_quat().astype(np.float32)


def _interpolate_pose_xyzw(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    t = float(np.clip(fraction, 0.0, 1.0))
    pos = (1.0 - t) * a[:3] + t * b[:3]
    slerp = Slerp([0.0, 1.0], R.from_quat(np.stack([a[3:7], b[3:7]], axis=0)))
    quat = slerp([t]).as_quat()[0].astype(np.float32)
    return np.concatenate([pos.astype(np.float32), quat]).astype(np.float32)


def _densify_pose_sequence_xyzw(poses: list[np.ndarray], per_segment: int) -> list[np.ndarray]:
    if not poses:
        return []
    steps = max(1, int(per_segment))
    dense = [np.asarray(poses[0], dtype=np.float32)]
    for start, end in zip(poses[:-1], poses[1:], strict=True):
        for i in range(1, steps + 1):
            dense.append(_interpolate_pose_xyzw(start, end, i / steps))
    return dense


def _policy_to_leg_asset_pose_xyzw(policy_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(policy_pose, dtype=np.float32).copy()
    q_policy_asset = _quat_matrix_xyzw(R_USD_POLICY_LEG.T)
    pose[3:7] = _quat_multiply_xyzw(pose[3:7], q_policy_asset)
    return pose


def _leg_asset_to_policy_pose_xyzw(asset_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(asset_pose, dtype=np.float32).copy()
    q_asset_policy = _quat_matrix_xyzw(R_USD_POLICY_LEG)
    pose[3:7] = _quat_multiply_xyzw(pose[3:7], q_asset_policy)
    return pose


def _leg_policy_pose_from_asset_rotation(pos: np.ndarray, rot: R) -> np.ndarray:
    asset_pose = np.array(
        [float(pos[0]), float(pos[1]), float(pos[2]), *rot.as_quat()],
        dtype=np.float32,
    )
    return _leg_asset_to_policy_pose_xyzw(asset_pose)


def _upright_leg_policy_pose(pos: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Return a policy-frame pose for an upright raw SquareLeg asset.

    Raw SquareLeg ``+Z`` points from screw threads toward the handle, so the
    inserted/upright asset orientation is just a yaw about raw/world ``+Z``.
    """
    return _leg_policy_pose_from_asset_rotation(pos, R.from_euler("z", yaw_rad))


def _keypoints_from_pose_xyzw(
    pose_xyzw: np.ndarray,
    object_scale: np.ndarray,
    object_base_size: float,
    keypoint_scale: float,
) -> np.ndarray:
    """Compute deployment-style SimToolReal object keypoints."""
    pose = np.asarray(pose_xyzw, dtype=np.float32)
    offsets = (
        POLICY_KEYPOINT_CORNERS
        * np.asarray(object_scale, dtype=np.float32)[None]
        * float(object_base_size)
        * float(keypoint_scale)
        * 0.5
    )
    return pose[:3][None] + R.from_quat(pose[3:7]).apply(offsets)


def _keypoint_max_dist_xyzw(
    object_pose_xyzw: np.ndarray,
    goal_pose_xyzw: np.ndarray,
    object_scale: np.ndarray,
    object_base_size: float,
    keypoint_scale: float,
) -> float:
    object_kps = _keypoints_from_pose_xyzw(
        object_pose_xyzw,
        object_scale,
        object_base_size=object_base_size,
        keypoint_scale=keypoint_scale,
    )
    goal_kps = _keypoints_from_pose_xyzw(
        goal_pose_xyzw,
        object_scale,
        object_base_size=object_base_size,
        keypoint_scale=keypoint_scale,
    )
    return float(np.linalg.norm(object_kps - goal_kps, axis=-1).max())


def _sample_furniturebench_leg_start_pose(
    args,
    hole: np.ndarray,
    default_start_pose: np.ndarray,
) -> np.ndarray:
    if not bool(args.leg_randomize_start):
        return default_start_pose

    rng = np.random.default_rng(int(args.seed))
    xy_center = np.asarray(args.leg_random_start_xy_center, dtype=np.float32)
    xy_range = np.asarray(args.leg_random_start_xy_range, dtype=np.float32)
    roll_pitch_range = np.asarray(args.leg_random_start_roll_pitch_range_deg, dtype=np.float32)

    xy = xy_center + rng.uniform(-xy_range, xy_range).astype(np.float32)
    z = (
        float(hole[2])
        + float(args.leg_random_start_z_offset_from_hole)
        + float(rng.uniform(-float(args.leg_random_start_z_range), float(args.leg_random_start_z_range)))
    )
    roll_deg = float(rng.uniform(-roll_pitch_range[0], roll_pitch_range[0]))
    pitch_deg = float(rng.uniform(-roll_pitch_range[1], roll_pitch_range[1]))
    yaw_deg = float(args.leg_random_start_yaw_center_deg) + float(
        rng.uniform(-float(args.leg_random_start_yaw_range_deg), float(args.leg_random_start_yaw_range_deg))
    )

    start_pose = _leg_policy_pose_from_asset_rotation(
        np.array([xy[0], xy[1], z], dtype=np.float32),
        R.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True),
    )
    args._sampled_leg_start_asset_pose_xyzw = _policy_to_leg_asset_pose_xyzw(start_pose)
    args._sampled_leg_start_rpy_deg = np.array([roll_deg, pitch_deg, yaw_deg], dtype=np.float32)
    return start_pose


def _make_furniturebench_leg_trajectory(args) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    table_center_z = 0.38
    table_half_height = 0.15
    fixture_local_bottom = -0.01558619
    fixture_root_z = (
        table_center_z
        + table_half_height
        - fixture_local_bottom
        + float(args.leg_fixture_clearance)
    )
    fixture_root = np.array([0.0, 0.0, fixture_root_z], dtype=np.float32)
    hole = fixture_root + FURNITUREBENCH_TABLE_HOLES[args.leg_hole_index]

    if bool(args.leg_use_omnireset_final_height):
        final_z = (
            fixture_root[2]
            + FURNITUREBENCH_TABLE_ASSEMBLED_OFFSET[2]
            - FURNITUREBENCH_LEG_ASSEMBLED_OFFSET[2]
        )
    else:
        final_z = hole[2] + float(args.leg_insert_height)

    if args.leg_preinsert_height is None:
        preinsert_z = final_z + float(args.leg_preinsert_final_z_offset)
    else:
        preinsert_z = hole[2] + float(args.leg_preinsert_height)

    preinsert_yaw = np.deg2rad(float(args.leg_preinsert_yaw_offset_deg))
    final_yaw = np.deg2rad(float(args.leg_final_yaw_offset_deg))
    hover_pose = _upright_leg_policy_pose(
        np.array([hole[0], hole[1], hole[2] + args.leg_hover_height], dtype=np.float32),
        final_yaw,
    )
    preinsert_pose = _upright_leg_policy_pose(
        np.array([hole[0], hole[1], preinsert_z], dtype=np.float32),
        preinsert_yaw,
    )
    final_pose = _upright_leg_policy_pose(
        np.array([hole[0], hole[1], final_z], dtype=np.float32),
        final_yaw,
    )
    start_pose = _upright_leg_policy_pose(
        np.array([0.10, 0.08, hole[2] + 0.18], dtype=np.float32),
        final_yaw,
    )
    start_pose = _sample_furniturebench_leg_start_pose(args, hole, start_pose)

    if args.leg_goal_sequence == "final_only":
        return start_pose, [final_pose], fixture_root
    if args.leg_goal_sequence == "preinsert_final":
        return start_pose, [preinsert_pose, final_pose], fixture_root
    if args.leg_goal_sequence == "pingpong":
        goals = []
        for _ in range(max(1, int(args.leg_pingpong_cycles))):
            goals.extend([preinsert_pose.copy(), final_pose.copy()])
        return start_pose, goals, fixture_root

    goals = [hover_pose, preinsert_pose]

    has_spin = int(args.leg_spin_steps) > 0 and abs(float(args.leg_spin_turns)) > 1.0e-8
    if args.leg_spin_mode == "interpolate":
        for pose in _densify_pose_sequence([preinsert_pose, final_pose], max(1, int(args.leg_descend_steps)))[1:]:
            goals.append(pose)
        return start_pose, goals, fixture_root

    if has_spin and args.leg_spin_mode == "helical":
        # Positive leg_spin_turns means clockwise from above: raw asset yaw
        # decreases about +Z, equivalent to positive policy-local +X.
        total_spin = 2.0 * np.pi * float(args.leg_spin_turns)
        spin_steps = int(args.leg_spin_steps)
        heights = np.linspace(
            float(preinsert_z),
            float(final_z),
            spin_steps + 1,
            dtype=np.float32,
        )[1:]
        thetas = np.linspace(total_spin / spin_steps, total_spin, spin_steps)
        for height, theta in zip(heights, thetas, strict=True):
            yaw = preinsert_yaw - theta
            goals.append(
                _upright_leg_policy_pose(
                    np.array([hole[0], hole[1], height], dtype=np.float32),
                    yaw,
                )
            )
        return start_pose, goals, fixture_root

    descend_steps = max(1, int(args.leg_descend_steps))
    descend_heights = np.linspace(
        float(preinsert_z),
        float(final_z),
        descend_steps + 1,
        dtype=np.float32,
    )[1:]
    for height in descend_heights:
        goals.append(
            _upright_leg_policy_pose(
                np.array([hole[0], hole[1], height], dtype=np.float32),
                preinsert_yaw,
            )
        )

    if has_spin:
        # Policy +x points down into the hole. Positive turns spin into a
        # right-handed thread: raw asset yaw decreases about +Z.
        total_spin = 2.0 * np.pi * float(args.leg_spin_turns)
        for theta in np.linspace(total_spin / args.leg_spin_steps, total_spin, args.leg_spin_steps):
            goals.append(
                _upright_leg_policy_pose(
                    np.array([hole[0], hole[1], final_z], dtype=np.float32),
                    preinsert_yaw - theta,
                )
            )
    return start_pose, goals, fixture_root


def _furniturebench_asset_root() -> Path:
    for root in FURNITUREBENCH_ASSET_ROOTS:
        if (
            (root / "SquareLeg" / "square_leg.usd").exists()
            and (root / "SquareTableTop" / "square_table_top.usd").exists()
        ):
            return root
    raise FileNotFoundError(
        "Could not find FurnitureBench SquareLeg/SquareTableTop USDs. "
        "Expected them under assets/usd/furniturebench or "
        ".pretrained_checkpoints/SimToolReal/omnireset_assets/FurnitureBench."
    )


def _load_dextoolbench_trajectory(args) -> tuple[np.ndarray, list[np.ndarray], tuple[float, float, float], str]:
    from dextoolbench.objects import NAME_TO_OBJECT

    obj_info = NAME_TO_OBJECT[args.object_name]
    traj_path = (
        REPO_ROOT
        / "dextoolbench"
        / "trajectories"
        / args.object_category
        / args.object_name
        / f"{args.task_name}.json"
    )
    data = json.loads(traj_path.read_text())
    start_pose = np.asarray(data["start_pose"], dtype=np.float32)
    goals = [np.asarray(goal, dtype=np.float32) for goal in data["goals"]]
    return start_pose, goals, tuple(float(x) for x in obj_info.scale), str(obj_info.urdf_path)


def _disable_randomness(cfg) -> None:
    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)

    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_position_noise_z = 0.0
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0

    term = cfg.termination
    term.eval_success_tolerance = 0.0
    term.max_consecutive_successes = 0
    term.auto_goal_reset_on_success = False
    term.force_consecutive_near_goal_steps = True


def _apply_omnireset_leg_physics_profile(cfg) -> None:
    """Use the contact/mass/friction profile from the UWLab OmniReset leg task."""
    cfg.sim.physx.solver_type = 1
    cfg.sim.physx.max_position_iteration_count = 192
    cfg.sim.physx.max_velocity_iteration_count = 1
    cfg.sim.physx.bounce_threshold_velocity = 0.02
    cfg.sim.physx.friction_offset_threshold = 0.01
    cfg.sim.physx.friction_correlation_distance = 0.0005
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
    cfg.sim.physx.gpu_max_rigid_contact_count = 2**23
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**23
    cfg.sim.physx.gpu_collision_stack_size = 2**31

    # OmniReset spawns the insertive object at 1 g, then its inherited startup
    # mass event overwrites the runtime mass with an absolute 20-200 g sample.
    # Use a fixed in-range value here to keep this deterministic rollout
    # representative without copying the full reset-event randomization system.
    cfg.assets.object_mass = 0.05
    cfg.assets.fixture_mass = 0.5
    cfg.assets.object_solver_position_iteration_count = 4
    cfg.assets.object_solver_velocity_iteration_count = 0
    cfg.assets.fixture_solver_position_iteration_count = 4
    cfg.assets.fixture_solver_velocity_iteration_count = 0

    # Use representative values from the OmniReset startup randomization ranges:
    # insertive object (1.0-2.0), receptive object (0.2-0.6), table (0.3-0.6).
    cfg.assets.object_friction = 1.5
    cfg.assets.object_dynamic_friction = 1.4
    cfg.assets.fixture_friction = 0.4
    cfg.assets.fixture_dynamic_friction = 0.325
    cfg.assets.table_friction = 0.45
    cfg.assets.table_dynamic_friction = 0.35


def _make_cfg(args):
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = int(args.num_envs)
    cfg.assets.shuffle_assets = False
    cfg.episode_length_s = max(10.0, float(args.max_steps + 60) / 60.0)
    _disable_randomness(cfg)

    if args.scenario == "claw_hammer":
        start_pose, goals, object_scale, object_urdf = _load_dextoolbench_trajectory(args)
        cfg.assets.object_urdf_paths = (object_urdf,)
        cfg.assets.object_usd_paths = ()
        cfg.assets.object_scales = (object_scale,)
        cfg.reset.fixed_start_pose = pose_xyzw_to_wxyz(start_pose)
        cfg.reset.fixed_goal_pose = pose_xyzw_to_wxyz(goals[0])
        return cfg, start_pose, goals, np.asarray(object_scale, dtype=np.float32)

    start_pose_policy, goals_policy, fixture_root = _make_furniturebench_leg_trajectory(args)
    furniturebench_root = _furniturebench_asset_root()
    square_leg = furniturebench_root / "SquareLeg" / "square_leg.usd"
    square_table = furniturebench_root / "SquareTableTop" / "square_table_top.usd"
    cfg.assets.object_urdf_paths = ()
    cfg.assets.object_usd_paths = (str(square_leg),)
    cfg.assets.object_scales = (tuple(float(x) for x in FURNITUREBENCH_LEG_OBJECT_SCALE[0]),)
    cfg.assets.fixture_usd_path = str(square_table)
    cfg.assets.object_policy_frame_quat_wxyz = tuple(
        float(x) for x in xyzw_to_wxyz(_quat_matrix_xyzw(R_USD_POLICY_LEG))
    )
    if args.leg_physics_profile == "omnireset":
        _apply_omnireset_leg_physics_profile(cfg)

    start_pose_asset = _policy_to_leg_asset_pose_xyzw(start_pose_policy)
    first_goal_asset = _policy_to_leg_asset_pose_xyzw(goals_policy[0])
    cfg.reset.fixed_start_pose = pose_xyzw_to_wxyz(start_pose_asset)
    cfg.reset.fixed_goal_pose = pose_xyzw_to_wxyz(first_goal_asset)
    cfg.reset.fixed_fixture_pose = (
        float(fixture_root[0]),
        float(fixture_root[1]),
        float(fixture_root[2]),
        1.0,
        0.0,
        0.0,
        0.0,
    )
    return cfg, start_pose_policy, goals_policy, FURNITUREBENCH_LEG_OBJECT_SCALE[0].copy()


def _policy_goal_to_asset_goal(args, goal_policy_xyzw: np.ndarray) -> np.ndarray:
    if args.scenario == "furniturebench_leg":
        return _policy_to_leg_asset_pose_xyzw(goal_policy_xyzw)
    return np.asarray(goal_policy_xyzw, dtype=np.float32)


def _asset_pose_wxyz_to_policy_xyzw(args, pose_wxyz: np.ndarray) -> np.ndarray:
    pose_xyzw = pose_wxyz_to_xyzw(pose_wxyz)
    if args.scenario == "furniturebench_leg":
        return _leg_asset_to_policy_pose_xyzw(pose_xyzw)
    return pose_xyzw


def _create_record_camera(inner):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    camera_cfg = CameraCfg(
        prim_path="/World/RecordCamera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -1.0, 1.03),
            rot=(0.8507, 0.5257, 0.0, 0.0),
            convention="opengl",
        ),
    )
    camera = Camera(cfg=camera_cfg)
    inner.sim.reset()
    return camera


def _set_goal_viz_visibility(visible: bool) -> None:
    from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage
    from pxr import UsdGeom

    stage = get_current_stage()
    prim_paths = find_matching_prim_paths("/World/envs/env_.*/GoalViz")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        imageable = UsdGeom.Imageable(prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
    print(
        f"[rollout] goal_viz_visible={visible} prims={len(prim_paths)}",
        flush=True,
    )


def _write_goal(inner, args, goal_policy_xyzw: np.ndarray) -> None:
    import torch

    from isaacsimenvs.tasks.simtoolreal.utils.reset_utils import _clear_goal_trackers

    goal_asset_xyzw = _policy_goal_to_asset_goal(args, goal_policy_xyzw)
    pose = torch.tensor(
        [pose_xyzw_to_wxyz(goal_asset_xyzw)],
        device=inner.device,
        dtype=torch.float32,
    )
    env_ids = torch.arange(inner.num_envs, device=inner.device, dtype=torch.long)
    pose = pose.expand(inner.num_envs, -1).clone()
    pose[:, 0:3] += inner.scene.env_origins
    inner.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
    _clear_goal_trackers(inner, env_ids)


def _write_object_pose(inner, args, object_policy_xyzw: np.ndarray, *, zero_velocity: bool) -> None:
    import torch

    object_asset_xyzw = _policy_goal_to_asset_goal(args, object_policy_xyzw)
    env_ids = torch.arange(inner.num_envs, device=inner.device, dtype=torch.long)
    pose = torch.tensor(
        [pose_xyzw_to_wxyz(object_asset_xyzw)],
        device=inner.device,
        dtype=torch.float32,
    ).expand(inner.num_envs, -1).clone()
    pose[:, 0:3] += inner.scene.env_origins
    inner.object.write_root_pose_to_sim(pose, env_ids=env_ids)
    if zero_velocity:
        inner.object.write_root_velocity_to_sim(
            torch.zeros(inner.num_envs, 6, device=inner.device),
            env_ids=env_ids,
        )


def _teleport_pose_for_step(teleport_poses: list[np.ndarray], step: int, interval_steps: int) -> tuple[int, np.ndarray]:
    segment = min(step // interval_steps, max(0, len(teleport_poses) - 1))
    if segment >= len(teleport_poses) - 1:
        return segment, teleport_poses[-1]
    fraction = (step % interval_steps) / float(interval_steps)
    return segment, _interpolate_pose_xyzw(
        teleport_poses[segment],
        teleport_poses[segment + 1],
        fraction,
    )


def main() -> int:
    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from deployment.rl_player import RlPlayer

    args = _args
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    cfg, start_pose, goals, object_scale = _make_cfg(args)

    env = gym.make("Isaacsimenvs-SimToolReal-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None

    camera = _create_record_camera(inner) if args.record_video else None
    frames: list[np.ndarray] = []

    player = RlPlayer(
        num_observations=140,
        num_actions=cfg.action_space,
        config_path=str(args.config),
        checkpoint_path=str(args.checkpoint),
        device=args.rl_device,
        num_envs=cfg.scene.num_envs,
    )
    player.player.init_rnn()

    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(torch.zeros((cfg.scene.num_envs, cfg.action_space), device=inner.device))
    _write_goal(inner, args, goals[0])
    _set_goal_viz_visibility(bool(args.goal_viz_visible))
    obs = inner._get_observations()

    if args.robot_control_mode == "hold_current":
        inner._replay_target_lab_order = inner.robot.data.joint_pos.detach().clone()
    elif args.robot_control_mode == "hold_default":
        inner._replay_target_lab_order = inner.robot.data.default_joint_pos.detach().clone()

    teleport_poses: list[np.ndarray] = []
    teleport_interval_steps = max(1, int(round(float(args.teleport_interval_s) / float(inner.step_dt))))
    if args.object_drive_mode == "teleport_trajectory":
        teleport_source_poses = (
            [*goals]
            if bool(args.teleport_skip_start_pose)
            else [np.asarray(start_pose, dtype=np.float32), *goals]
        )
        teleport_poses = _densify_pose_sequence_xyzw(
            teleport_source_poses,
            int(args.teleport_waypoints_per_segment),
        )
        _write_object_pose(
            inner,
            args,
            teleport_poses[0],
            zero_velocity=not bool(args.teleport_keep_velocity),
        )
        _write_goal(inner, args, teleport_poses[0])
        obs = inner._get_observations()

    out_dir = Path(args.out_dir) / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_start = getattr(args, "_sampled_leg_start_asset_pose_xyzw", None)
    if sampled_start is not None:
        rpy_deg = getattr(args, "_sampled_leg_start_rpy_deg", np.zeros(3, dtype=np.float32))
        print(
            "[rollout] sampled leg start asset pose xyzw="
            f"{np.asarray(sampled_start, dtype=np.float32).tolist()} "
            f"rpy_deg={np.asarray(rpy_deg, dtype=np.float32).tolist()} seed={int(args.seed)}",
            flush=True,
        )

    obs_log: list[np.ndarray] = []
    action_log: list[np.ndarray] = []
    object_pose_log: list[np.ndarray] = []
    object_pose_asset_log: list[np.ndarray] = []
    goal_pose_log: list[np.ndarray] = []
    kp_dist_log: list[float] = []
    env_kp_dist_log: list[float] = []
    goal_idx_log: list[int] = []
    teleport_pose_log: list[np.ndarray] = []

    current_goal_idx = 0
    teleport_goal_idx = 0
    near_goal_steps = 0
    print(
        f"[rollout] scenario={args.scenario} max_steps={args.max_steps} "
        f"goals={len(goals)} checkpoint={args.checkpoint} "
        f"leg_physics_profile={getattr(args, 'leg_physics_profile', 'n/a')} "
        f"robot_control_mode={args.robot_control_mode} "
        f"object_drive_mode={args.object_drive_mode}",
        flush=True,
    )
    if teleport_poses:
        print(
            f"[rollout] teleport trajectory poses={len(teleport_poses)} "
            f"interval_steps={teleport_interval_steps} "
            f"interval_s={teleport_interval_steps * inner.step_dt:.3f} "
            f"write_every_step={bool(args.teleport_write_every_step)} "
            f"interpolate={bool(args.teleport_interpolate_between_waypoints)} "
            f"skip_start={bool(args.teleport_skip_start_pose)}",
            flush=True,
        )

    for step in range(int(args.max_steps)):
        active_teleport_pose = None
        if teleport_poses:
            if args.teleport_write_every_step:
                if args.teleport_interpolate_between_waypoints:
                    teleport_goal_idx, active_teleport_pose = _teleport_pose_for_step(
                        teleport_poses, step, teleport_interval_steps
                    )
                else:
                    teleport_goal_idx = min(step // teleport_interval_steps, len(teleport_poses) - 1)
                    active_teleport_pose = teleport_poses[teleport_goal_idx]
                _write_object_pose(
                    inner,
                    args,
                    active_teleport_pose,
                    zero_velocity=not bool(args.teleport_keep_velocity),
                )
                if step % teleport_interval_steps == 0:
                    _write_goal(inner, args, active_teleport_pose)
                    obs = inner._get_observations()
            elif step % teleport_interval_steps == 0:
                teleport_goal_idx = min(step // teleport_interval_steps, len(teleport_poses) - 1)
                active_teleport_pose = teleport_poses[teleport_goal_idx]
                _write_object_pose(
                    inner,
                    args,
                    active_teleport_pose,
                    zero_velocity=not bool(args.teleport_keep_velocity),
                )
                _write_goal(inner, args, active_teleport_pose)
                obs = inner._get_observations()
                print(
                    f"[rollout] step={step:4d} teleported object pose "
                    f"{teleport_goal_idx}/{len(teleport_poses) - 1}",
                    flush=True,
                )

        policy_obs = obs["policy"].to(args.rl_device)
        action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        obs, reward, terminated, truncated, info = env.step(action.to(inner.device))

        if args.render_each_step:
            inner.sim.render()
        if camera is not None and step % int(args.frame_every) == 0:
            camera.update(inner.physics_dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frames.append(rgb[0].detach().cpu().numpy()[:, :, :3])

        object_pose_asset_wxyz = torch.cat(
            [
                inner.object.data.root_pos_w[0] - inner.scene.env_origins[0],
                inner.object.data.root_quat_w[0],
            ]
        ).detach().cpu().numpy()
        goal_pose_asset_wxyz = torch.cat(
            [
                inner.goal_viz.data.root_pos_w[0] - inner.scene.env_origins[0],
                inner.goal_viz.data.root_quat_w[0],
            ]
        ).detach().cpu().numpy()
        object_pose_policy_xyzw = _asset_pose_wxyz_to_policy_xyzw(args, object_pose_asset_wxyz)
        goal_pose_policy_xyzw = _asset_pose_wxyz_to_policy_xyzw(args, goal_pose_asset_wxyz)

        active_goal_idx = teleport_goal_idx if teleport_poses else current_goal_idx
        env_kp_dist = float(inner._keypoints_max_dist[0].detach().cpu().item())
        kp_dist = _keypoint_max_dist_xyzw(
            object_pose_policy_xyzw,
            goal_pose_policy_xyzw,
            object_scale=object_scale,
            object_base_size=cfg.reward.object_base_size,
            keypoint_scale=cfg.reward.keypoint_scale,
        )
        if kp_dist < float(args.keypoint_tolerance):
            near_goal_steps += 1
        else:
            near_goal_steps = 0

        obs_log.append(policy_obs[0].detach().cpu().numpy())
        action_log.append(action[0].detach().cpu().numpy())
        object_pose_log.append(object_pose_policy_xyzw)
        object_pose_asset_log.append(pose_wxyz_to_xyzw(object_pose_asset_wxyz))
        goal_pose_log.append(goal_pose_policy_xyzw)
        kp_dist_log.append(kp_dist)
        env_kp_dist_log.append(env_kp_dist)
        goal_idx_log.append(active_goal_idx)
        if active_teleport_pose is None and teleport_poses:
            active_teleport_pose = teleport_poses[teleport_goal_idx]
        if active_teleport_pose is not None:
            teleport_pose_log.append(np.asarray(active_teleport_pose, dtype=np.float32))

        if step % 60 == 0:
            print(
                f"[rollout] step={step:4d} goal={active_goal_idx}/{len(goals)} "
                f"kp_dist={kp_dist:.4f} env_kp={env_kp_dist:.4f} "
                f"reward={float(reward[0].detach().cpu()):+.3f} "
                f"near={near_goal_steps}/{args.success_steps}",
                flush=True,
            )

        if not teleport_poses and near_goal_steps >= int(args.success_steps):
            print(
                f"[rollout] step={step} reached goal {active_goal_idx} "
                f"kp_dist={kp_dist:.4f} env_kp={env_kp_dist:.4f}",
                flush=True,
            )
            current_goal_idx += 1
            near_goal_steps = 0
            if current_goal_idx >= len(goals):
                print(f"[rollout] all goals reached at step {step}", flush=True)
                if not bool(args.continue_after_all_goals):
                    break
                current_goal_idx = len(goals) - 1
            _write_goal(inner, args, goals[current_goal_idx])
            obs = inner._get_observations()

        if (bool(terminated[0].item()) or bool(truncated[0].item())) and not bool(args.ignore_dones):
            print(
                f"[rollout] env ended at step={step} "
                f"terminated={bool(terminated[0].item())} truncated={bool(truncated[0].item())}",
                flush=True,
            )
            break

    npz_path = out_dir / "trajectory.npz"
    np.savez(
        npz_path,
        obs=np.asarray(obs_log, dtype=np.float32),
        actions=np.asarray(action_log, dtype=np.float32),
        object_poses_policy_xyzw=np.asarray(object_pose_log, dtype=np.float32),
        object_poses_asset_xyzw=np.asarray(object_pose_asset_log, dtype=np.float32),
        goal_poses_policy_xyzw=np.asarray(goal_pose_log, dtype=np.float32),
        kp_dists=np.asarray(kp_dist_log, dtype=np.float32),
        env_kp_dists=np.asarray(env_kp_dist_log, dtype=np.float32),
        goal_idxs=np.asarray(goal_idx_log, dtype=np.int32),
        goals_policy_xyzw=np.asarray(goals, dtype=np.float32),
        teleport_poses_policy_xyzw=np.asarray(teleport_poses, dtype=np.float32),
        commanded_teleport_poses_policy_xyzw=np.asarray(teleport_pose_log, dtype=np.float32),
        teleport_interval_steps=np.asarray([teleport_interval_steps], dtype=np.int32),
        start_pose_policy_xyzw=np.asarray(start_pose, dtype=np.float32),
        seed=np.asarray([int(args.seed)], dtype=np.int32),
        leg_randomize_start=np.asarray([bool(args.leg_randomize_start)], dtype=np.bool_),
        object_scale=np.asarray(object_scale, dtype=np.float32),
        scenario=args.scenario,
        robot_control_mode=args.robot_control_mode,
        object_drive_mode=args.object_drive_mode,
        goal_viz_visible=np.asarray([bool(args.goal_viz_visible)], dtype=np.bool_),
    )
    print(f"[rollout] wrote {npz_path}", flush=True)

    if frames:
        import imageio

        video_path = out_dir / "rollout.mp4"
        imageio.mimwrite(str(video_path), frames, fps=int(args.video_fps))
        print(f"[rollout] wrote {video_path} frames={len(frames)}", flush=True)

    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        del _app
        os._exit(code)
