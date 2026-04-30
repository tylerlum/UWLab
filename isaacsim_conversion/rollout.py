"""
Full policy rollout in Isaac Sim.
Phase 9 of the plan: obs → policy → action → physics loop.

Usage:
    source .venv_isaacsim/bin/activate
    PYTHONPATH=. python isaacsim_conversion/rollout.py \
        --assembly beam --part_id 2 --collision_method coacd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RL_GAMES = REPO_ROOT / "rl_games"
if LOCAL_RL_GAMES.exists():
    sys.path.insert(0, str(LOCAL_RL_GAMES))
sys.path.insert(0, str(REPO_ROOT))

from deployment.rl_player import RlPlayer
from isaacsim_conversion.isaacsim_env import _log
from isaacgymenvs.utils.observation_action_utils_sharpa import (
    JOINT_NAMES_ISAACGYM,
    N_OBS,
    _compute_keypoint_positions,
    compute_joint_pos_targets,
    compute_observation,
    create_urdf_object,
)

FURNITUREBENCH_ASSET_ROOT = "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Props/FurnitureBench"
LOCAL_FURNITUREBENCH_ASSET_ROOT = REPO_ROOT / ".pretrained_checkpoints/SimToolReal/omnireset_assets/FurnitureBench"

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
FURNITUREBENCH_TABLE_TRANSLATION = np.array([0.0, 0.0, 0.38], dtype=np.float32)

# The graspable cuboidal handle is roughly x/y/z = 0.03018/0.03018/0.06255 m
# in the SquareLeg USD root frame. The policy frame maps handle-to-thread length
# to +x, and SimToolReal encodes metric bbox sizes by multiplying by 25.
FURNITUREBENCH_LEG_OBJECT_SCALE = np.array([[0.06255, 0.03017705, 0.03017705]], dtype=np.float32) * 25.0


def _prefer_local_asset(local_path: Path, remote_url: str) -> str:
    return str(local_path) if local_path.exists() else remote_url


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (R.from_quat(a) * R.from_quat(b)).as_quat().astype(np.float32)


def _quat_matrix_xyzw(matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(matrix).as_quat().astype(np.float32)


def _policy_to_leg_usd_pose(policy_pose: np.ndarray) -> np.ndarray:
    """Convert virtual SimToolReal leg policy frame pose to the SquareLeg USD root pose."""
    policy_pose = np.asarray(policy_pose, dtype=np.float32).copy()
    q_world_policy = policy_pose[3:7]
    q_policy_usd = _quat_matrix_xyzw(R_USD_POLICY_LEG.T)
    policy_pose[3:7] = _quat_multiply_xyzw(q_world_policy, q_policy_usd)
    return policy_pose


def _leg_usd_to_policy_pose(usd_pose: np.ndarray) -> np.ndarray:
    """Convert SquareLeg USD root pose to the virtual SimToolReal policy frame pose."""
    usd_pose = np.asarray(usd_pose, dtype=np.float32).copy()
    q_world_usd = usd_pose[3:7]
    q_usd_policy = _quat_matrix_xyzw(R_USD_POLICY_LEG)
    usd_pose[3:7] = _quat_multiply_xyzw(q_world_usd, q_usd_policy)
    return usd_pose


def _make_furniturebench_leg_trajectory(args) -> tuple[list[float], list[list[float]]]:
    hole = FURNITUREBENCH_TABLE_TRANSLATION + FURNITUREBENCH_TABLE_HOLES[args.leg_hole_index]
    base_rot = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    base_quat = _quat_matrix_xyzw(base_rot)

    start_pose = np.array([0.10, 0.08, 0.55, *base_quat], dtype=np.float32)
    hover = np.array([hole[0], hole[1], hole[2] + args.leg_hover_height, *base_quat], dtype=np.float32)
    preinsert = np.array([hole[0], hole[1], hole[2] + args.leg_preinsert_height, *base_quat], dtype=np.float32)
    inserted = np.array([hole[0], hole[1], hole[2] + args.leg_insert_height, *base_quat], dtype=np.float32)

    goals = [hover, preinsert, inserted]
    total_spin = -2.0 * np.pi * args.leg_spin_turns
    for theta in np.linspace(total_spin / args.leg_spin_steps, total_spin, args.leg_spin_steps):
        spin_quat = _quat_multiply_xyzw(base_quat, R.from_euler("x", theta).as_quat())
        goals.append(np.array([hole[0], hole[1], hole[2] + args.leg_insert_height, *spin_quat], dtype=np.float32))

    return start_pose.tolist(), [goal.tolist() for goal in goals]


def launch_app():
    """Create AppLauncher FIRST, before any isaaclab/omni imports."""
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_source",
        choices=["fabrica", "dextoolbench", "furniturebench_leg"],
        default="dextoolbench",
        help="Which asset/trajectory layout to use.",
    )
    parser.add_argument("--assembly", default="beam")
    parser.add_argument("--part_id", default="2")
    parser.add_argument("--collision_method", default="coacd")
    parser.add_argument("--object_category", default="hammer")
    parser.add_argument("--object_name", default="claw_hammer")
    parser.add_argument("--task_name", default="swing_down")
    parser.add_argument("--leg_hole_index", type=int, default=0, choices=range(4), help="FurnitureBench tabletop hole index.")
    parser.add_argument("--leg_hover_height", type=float, default=0.12, help="Policy-frame root height above the selected hole.")
    parser.add_argument("--leg_preinsert_height", type=float, default=0.070, help="Pre-insertion policy-frame root height above the selected hole.")
    parser.add_argument("--leg_insert_height", type=float, default=0.038, help="Inserted policy-frame root height above the selected hole.")
    parser.add_argument("--leg_spin_turns", type=float, default=1.0, help="Clockwise local-x turns after insertion.")
    parser.add_argument("--leg_spin_steps", type=int, default=8, help="Number of local-x spin waypoints.")
    parser.add_argument("--max_steps", type=int, default=6000, help="Max sim steps (100s at 60Hz)")
    parser.add_argument("--video_dir", default="outputs/simtoolreal_rollouts", help="Directory to save video")
    parser.add_argument("--video_fps", type=int, default=30, help="Video FPS")
    parser.add_argument(
        "--checkpoint",
        default=".pretrained_checkpoints/SimToolReal/pretrained_policy/model.pth",
        help="Policy checkpoint path",
    )
    parser.add_argument(
        "--config",
        default=".pretrained_checkpoints/SimToolReal/pretrained_policy/config.yaml",
        help="Policy config path",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    return app_launcher.app, args


# Must happen at module level, before any other imports
_app, _args = launch_app()


def main():
    args = _args
    app = _app
    repo_root = REPO_ROOT
    device = "cuda"
    import isaaclab.sim as sim_utils

    # --- Locate assets ---
    robot_urdf = str(repo_root / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf")

    # Object/table/task selection
    obj_info = None
    if args.task_source == "fabrica":
        from dextoolbench.objects import NAME_TO_OBJECT
        table_urdf = str(
            repo_root
            / f"assets/urdf/fabrica/{args.assembly}/environments/{args.part_id}/scene_{args.collision_method}.urdf"
        )
        traj_path = repo_root / f"assets/urdf/fabrica/{args.assembly}/trajectories/{args.part_id}/pick_place.json"
        object_name = f"{args.assembly}_{args.part_id}_{args.collision_method}"
        import fabrica.objects  # noqa: registers fabrica parts
        obj_info = NAME_TO_OBJECT.get(object_name)
        assert obj_info is not None, f"Object '{object_name}' not found in NAME_TO_OBJECT"
        object_urdf = obj_info.urdf_path
    elif args.task_source == "dextoolbench":
        from dextoolbench.objects import NAME_TO_OBJECT
        table_urdf = str(
            repo_root
            / f"assets/urdf/dextoolbench/environments/{args.object_category}/{args.object_name}/{args.task_name}.urdf"
        )
        traj_path = repo_root / f"dextoolbench/trajectories/{args.object_category}/{args.object_name}/{args.task_name}.json"
        object_name = args.object_name
        obj_info = NAME_TO_OBJECT.get(object_name)
        assert obj_info is not None, f"Object '{object_name}' not found in NAME_TO_OBJECT"
        object_urdf = obj_info.urdf_path
    else:
        table_urdf = _prefer_local_asset(
            LOCAL_FURNITUREBENCH_ASSET_ROOT / "SquareTableTop/square_table_top.usd",
            f"{FURNITUREBENCH_ASSET_ROOT}/SquareTableTop/square_table_top.usd",
        )
        object_urdf = _prefer_local_asset(
            LOCAL_FURNITUREBENCH_ASSET_ROOT / "SquareLeg/square_leg.usd",
            f"{FURNITUREBENCH_ASSET_ROOT}/SquareLeg/square_leg.usd",
        )
        traj_path = None
        object_name = "fbleg"

    print(f"Robot URDF:  {robot_urdf}")
    print(f"Table URDF:  {table_urdf}")
    print(f"Object URDF: {object_urdf}")
    print(f"Trajectory:  {traj_path if traj_path is not None else 'generated furniturebench_leg path'}")

    # --- Load trajectory ---
    if args.task_source == "furniturebench_leg":
        start_pose, goals = _make_furniturebench_leg_trajectory(args)
    else:
        with open(traj_path) as f:
            traj = json.load(f)
        goals = traj["goals"]  # list of [x,y,z,qx,qy,qz,qw]
        start_pose = traj["start_pose"]  # [x,y,z,qx,qy,qz,qw]
    print(f"Trajectory: {len(goals)} goals, start_pose={start_pose}")

    # --- Create Isaac Sim environment ---
    from isaacsim_conversion.isaacsim_env import IsaacSimEnv
    env = IsaacSimEnv(
        robot_urdf=robot_urdf,
        table_urdf=table_urdf,
        object_urdf=object_urdf,
        headless=True,
        support_table=args.task_source == "furniturebench_leg",
        app=app,
    )

    start_pos = np.array(start_pose[:3], dtype=np.float32)
    start_quat_xyzw = np.array(start_pose[3:7], dtype=np.float32)

    # --- Load policy ---
    print("\nLoading policy...")
    checkpoint_path = str(repo_root / args.checkpoint) if not args.checkpoint.startswith("/") else args.checkpoint
    config_path = str(repo_root / args.config) if not args.config.startswith("/") else args.config
    _log(f"Loading policy: checkpoint={checkpoint_path}, config={config_path}")
    player = RlPlayer(
        num_observations=140,
        num_actions=29,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        num_envs=1,
    )

    # --- Load URDF for FK ---
    urdf = create_urdf_object("iiwa14_left_sharpa_adjusted_restricted")

    # --- Config from training ---
    obs_list = [
        "joint_pos", "joint_vel", "prev_action_targets", "palm_pos",
        "palm_rot", "object_rot", "fingertip_pos_rel_palm",
        "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
    ]
    import yaml
    with open(config_path) as f:
        policy_cfg = yaml.safe_load(f)
    if args.task_source == "fabrica":
        # Fabrica transfer uses fixedSize from the training config.
        fixed_size = policy_cfg.get("task", {}).get("env", {}).get("fixedSize", [0.141, 0.03025, 0.0271])
        object_scales = np.array([fixed_size], dtype=np.float32)
        _log(f"Object scales (fixedSize): {object_scales[0]}")
    elif args.task_source == "dextoolbench":
        # DexToolBench deployment/eval uses the object grasp bounding box scale from NAME_TO_OBJECT.
        object_scales = np.array([obj_info.scale], dtype=np.float32)
        _log(f"Object scales (NAME_TO_OBJECT.scale): {object_scales[0]}")
    else:
        object_scales = FURNITUREBENCH_LEG_OBJECT_SCALE
        _log(f"Object scales (FurnitureBench leg, meters * 25): {object_scales[0]}")
    hand_moving_average = 0.1
    arm_moving_average = 0.1
    dof_speed_scale = 1.5
    dt = 1 / 60
    success_steps = 10
    keypoint_tolerance = 0.01 * 1.5  # targetSuccessTolerance * keypointScale

    # --- Set up video recording ---
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    if args.task_source == "dextoolbench":
        video_stem = f"rollout_{args.object_category}_{args.object_name}_{args.task_name}"
    elif args.task_source == "furniturebench_leg":
        video_stem = f"rollout_furniturebench_leg_hole{args.leg_hole_index}"
    else:
        video_stem = f"rollout_{args.assembly}_{args.part_id}"
    video_path = video_dir / f"{video_stem}.mp4"
    frames = []

    # Record trajectory data for debugging/visualization
    trajectory_log = []

    # Set up camera for video recording (requires --enable_cameras flag)
    has_camera = False
    if getattr(args, 'enable_cameras', False):
        try:
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
                    # Match Isaac Gym: cam_pos=(0, -1, 1.03), cam_target=(0, 0, 0.53)
                    # opengl convention: -Z=forward, +Y=up
                    pos=(0.0, -1.0, 1.03),
                    rot=(0.8507, 0.5257, 0.0, 0.0),  # wxyz
                    convention="opengl",
                ),
            )
            camera = Camera(cfg=camera_cfg)
            # Camera needs sim reset to initialize internal buffers (_timestamp etc.)
            env.sim.reset()
            has_camera = True
            _log(f"Camera created for video recording to {video_dir}")
        except Exception as e:
            _log(f"Camera setup failed: {e}")
    else:
        _log("Video recording disabled (pass --enable_cameras to enable)")

    # Place object at start pose (must be after camera reset)
    if args.task_source == "furniturebench_leg":
        physical_start_pose = _policy_to_leg_usd_pose(np.concatenate([start_pos, start_quat_xyzw]))
        env.set_object_pose(physical_start_pose[:3], physical_start_pose[3:7])
    else:
        env.set_object_pose(start_pos, start_quat_xyzw)

    # --- Set robot to default pose (match Isaac Gym) ---
    env.reset_robot_to_default_pose(render=False)
    q_init, _ = env.get_robot_state()
    _log(f"Robot default pose set. Arm joints: {q_init[:7]}")
    prev_targets = q_init[None].copy()  # (1, 29)

    current_goal_idx = 0
    goal_pose = np.array(goals[current_goal_idx], dtype=np.float32)[None]  # (1, 7)
    near_goal_steps = 0

    player.player.init_rnn()  # fresh LSTM state

    _log(f"\n=== Starting rollout: {args.max_steps} steps, {len(goals)} goals ===\n")

    # --- Main loop ---
    for step_i in range(args.max_steps):
        # 1. Extract state
        q, qd = env.get_robot_state()
        object_pose_raw = env.get_object_pose_xyzw()
        if args.task_source == "furniturebench_leg":
            object_pose_raw = _leg_usd_to_policy_pose(object_pose_raw)
        object_pose = object_pose_raw[None]  # (1, 7)

        # 2. Compute observation
        obs = compute_observation(
            q=q[None], qd=qd[None], prev_action_targets=prev_targets,
            object_pose=object_pose, goal_object_pose=goal_pose,
            object_scales=object_scales, urdf=urdf, obs_list=obs_list,
        )

        # 3. Policy inference
        obs_tensor = torch.from_numpy(obs).float().to(device)
        action = player.get_normalized_action(obs_tensor, deterministic_actions=True)

        # 4. Compute joint targets
        targets = compute_joint_pos_targets(
            actions=action.cpu().numpy(),
            prev_targets=prev_targets,
            hand_moving_average=hand_moving_average,
            arm_moving_average=arm_moving_average,
            hand_dof_speed_scale=dof_speed_scale,
            dt=dt,
        )
        prev_targets = targets

        # 5. Apply and step
        env.set_joint_position_targets(targets[0])
        env.step(render=True)  # Always render for camera

        # 6. Capture frame (every 2nd step = 30fps video from 60Hz sim)
        if has_camera and step_i % 2 == 0:
            camera.update(dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frames.append(rgb[0].cpu().numpy()[:, :, :3])

        # 7. Goal switching
        object_kps = _compute_keypoint_positions(object_pose, object_scales)
        goal_kps = _compute_keypoint_positions(goal_pose, object_scales)
        keypoints_max_dist = np.max(np.linalg.norm(
            object_kps[0] - goal_kps[0], axis=-1
        ))

        if keypoints_max_dist < keypoint_tolerance:
            near_goal_steps += 1
        else:
            near_goal_steps = 0  # forceConsecutiveNearGoalSteps=True

        if near_goal_steps >= success_steps:
            _log(f"[step {step_i}] Goal {current_goal_idx} REACHED! (dist={keypoints_max_dist:.4f})")
            current_goal_idx += 1
            near_goal_steps = 0
            if current_goal_idx >= len(goals):
                _log(f"\n=== ALL {len(goals)} GOALS REACHED at step {step_i}! ===")
                break
            goal_pose = np.array(goals[current_goal_idx], dtype=np.float32)[None]
            _log(f"  -> Advancing to goal {current_goal_idx}/{len(goals)}")

        # Record trajectory data
        trajectory_log.append({
            "step": step_i,
            "q": q.copy(),
            "object_pose": object_pose[0].copy(),
            "kp_dist": keypoints_max_dist,
            "goal_idx": current_goal_idx,
        })

        # Periodic logging
        if step_i % 60 == 0:
            obj_z = object_pose[0, 2]
            _log(
                f"[step {step_i:5d}] goal={current_goal_idx}/{len(goals)}, "
                f"kp_dist={keypoints_max_dist:.4f}, obj_z={obj_z:.3f}, "
                f"near_goal={near_goal_steps}/{success_steps}"
            )

    _log(f"\nRollout complete. Final goal: {current_goal_idx}/{len(goals)}")

    # Save trajectory data
    if trajectory_log:
        traj_file = video_dir / "trajectory.npz"
        np.savez(
            str(traj_file),
            steps=np.array([t["step"] for t in trajectory_log]),
            q=np.array([t["q"] for t in trajectory_log]),
            object_poses=np.array([t["object_pose"] for t in trajectory_log]),
            kp_dists=np.array([t["kp_dist"] for t in trajectory_log]),
            goal_idxs=np.array([t["goal_idx"] for t in trajectory_log]),
        )
        _log(f"Trajectory saved: {traj_file} ({len(trajectory_log)} steps)")

    # Save video
    if frames:
        import imageio
        _log(f"Saving {len(frames)} frames to {video_path}")
        imageio.mimwrite(str(video_path), frames, fps=args.video_fps)
        _log(f"Video saved: {video_path}")
    else:
        _log("WARNING: No frames captured for video")

    env.close()


if __name__ == "__main__":
    main()
