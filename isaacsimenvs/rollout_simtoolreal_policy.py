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
from scipy.spatial.transform import Rotation as R


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

# Graspable cuboidal handle bbox in policy frame, encoded by SimToolReal as
# metric dimensions multiplied by 25.
FURNITUREBENCH_LEG_OBJECT_SCALE = (
    np.array([[0.06255, 0.03017705, 0.03017705]], dtype=np.float32) * 25.0
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
    parser.add_argument("--keypoint_tolerance", type=float, default=0.015)
    parser.add_argument("--success_steps", type=int, default=10)

    parser.add_argument("--leg_hole_index", type=int, default=0, choices=range(4))
    parser.add_argument("--leg_hover_height", type=float, default=0.12)
    parser.add_argument("--leg_preinsert_height", type=float, default=0.070)
    parser.add_argument("--leg_insert_height", type=float, default=0.038)
    parser.add_argument("--leg_spin_turns", type=float, default=1.0)
    parser.add_argument("--leg_spin_steps", type=int, default=8)
    parser.add_argument("--leg_fixture_clearance", type=float, default=0.002)
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

    base_quat = _quat_matrix_xyzw(R_USD_POLICY_LEG)
    start_pose = np.array([0.10, 0.08, hole[2] + 0.18, *base_quat], dtype=np.float32)
    goals = [
        np.array([hole[0], hole[1], hole[2] + args.leg_hover_height, *base_quat], dtype=np.float32),
        np.array([hole[0], hole[1], hole[2] + args.leg_preinsert_height, *base_quat], dtype=np.float32),
        np.array([hole[0], hole[1], hole[2] + args.leg_insert_height, *base_quat], dtype=np.float32),
    ]

    total_spin = -2.0 * np.pi * float(args.leg_spin_turns)
    for theta in np.linspace(total_spin / args.leg_spin_steps, total_spin, args.leg_spin_steps):
        spin_quat = _quat_multiply_xyzw(base_quat, R.from_euler("x", theta).as_quat())
        goals.append(
            np.array([hole[0], hole[1], hole[2] + args.leg_insert_height, *spin_quat], dtype=np.float32)
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
        return cfg, start_pose, goals

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
    return cfg, start_pose_policy, goals_policy


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


def main() -> int:
    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from deployment.rl_player import RlPlayer

    args = _args
    cfg, _start_pose, goals = _make_cfg(args)

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

    out_dir = Path(args.out_dir) / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_log: list[np.ndarray] = []
    action_log: list[np.ndarray] = []
    object_pose_log: list[np.ndarray] = []
    object_pose_asset_log: list[np.ndarray] = []
    goal_pose_log: list[np.ndarray] = []
    kp_dist_log: list[float] = []
    goal_idx_log: list[int] = []

    current_goal_idx = 0
    near_goal_steps = 0
    print(
        f"[rollout] scenario={args.scenario} max_steps={args.max_steps} "
        f"goals={len(goals)} checkpoint={args.checkpoint}",
        flush=True,
    )

    for step in range(int(args.max_steps)):
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

        kp_dist = float(inner._keypoints_max_dist[0].detach().cpu().item())
        if kp_dist < float(args.keypoint_tolerance):
            near_goal_steps += 1
        else:
            near_goal_steps = 0

        if near_goal_steps >= int(args.success_steps):
            print(
                f"[rollout] step={step} reached goal {current_goal_idx} "
                f"kp_dist={kp_dist:.4f}",
                flush=True,
            )
            current_goal_idx += 1
            near_goal_steps = 0
            if current_goal_idx >= len(goals):
                print(f"[rollout] all goals reached at step {step}", flush=True)
                break
            _write_goal(inner, args, goals[current_goal_idx])

        obs_log.append(policy_obs[0].detach().cpu().numpy())
        action_log.append(action[0].detach().cpu().numpy())
        object_pose_log.append(object_pose_policy_xyzw)
        object_pose_asset_log.append(pose_wxyz_to_xyzw(object_pose_asset_wxyz))
        goal_pose_log.append(goal_pose_policy_xyzw)
        kp_dist_log.append(kp_dist)
        goal_idx_log.append(current_goal_idx)

        if step % 60 == 0:
            print(
                f"[rollout] step={step:4d} goal={current_goal_idx}/{len(goals)} "
                f"kp_dist={kp_dist:.4f} reward={float(reward[0].detach().cpu()):+.3f} "
                f"near={near_goal_steps}/{args.success_steps}",
                flush=True,
            )

        if bool(terminated[0].item()) or bool(truncated[0].item()):
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
        goal_idxs=np.asarray(goal_idx_log, dtype=np.int32),
        goals_policy_xyzw=np.asarray(goals, dtype=np.float32),
        scenario=args.scenario,
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
