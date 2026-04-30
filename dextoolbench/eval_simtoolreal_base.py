"""Run the pretrained policy against the *base* SimToolReal env (legacy
isaacgymenvs path) and dump per-step obs / joint state / action to a .npz.

Historically paired with the retired DirectRLEnv eval script: same policy,
same procedural handle_head pool, but different physics backend (Isaac Gym
vs Isaac Lab/Sim). Diffing those npz files surfaced wiring bugs in the port
(e.g., joint-pos normalization, frame conventions, obs ordering, missing
transforms).

    python dextoolbench/eval_simtoolreal_base.py \\
        --num_steps 240 --output_npz /tmp/simtoolreal_base_obs.npz

Notes:
- Built from dextoolbench/eval.py — strips viser, fabrica object overrides,
  fixedGoalStates, video recording. Keeps the legacy DR-off override pattern
  so initial state and step dynamics are deterministic per seed.
- Uses the same RlPlayer + pretrained_policy/model.pth checkpoint as the
  retired Isaac Lab comparison path, so the action stream is identical given
  identical obs.
- Exits after num_steps regardless of done — we want a fixed-length trace
  even if the legacy env auto-resets.
"""

# isaacgym MUST be imported before torch
# isort: off
from isaacgym import gymapi, gymtorch, gymutil  # noqa: F401
import torch
# isort: on

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from deployment.isaac.isaac_env import create_env
from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.utils import get_repo_root_dir


REPO_ROOT = Path(get_repo_root_dir())


# Keep this in sync with isaacsimenvs canonical ordering.
JOINT_NAMES_CANONICAL = [
    "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
    "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
]


@dataclass
class Args:
    config_path: Path = REPO_ROOT / "pretrained_policy" / "config.yaml"
    """Policy config YAML (also used as env config)."""

    checkpoint_path: Path = REPO_ROOT / "pretrained_policy" / "model.pth"
    """rl_games .pth checkpoint."""

    output_npz: Path = REPO_ROOT / "rollout_videos_eval" / "simtoolreal_base_obs.npz"
    """Where to dump the per-step trace."""

    num_steps: int = 240
    """Number of policy steps to roll out (no early termination)."""

    seed: int = 0
    """RNG seed routed into the env config."""

    deterministic_actions: bool = True
    """If False, sample stochastic actions (matches training; we want True
    for diff-vs-port reproducibility)."""

    hold: bool = False
    """If True, bypass policy and send the action that maps targets to
    current joint positions (arm: 0; hand: inverse of absolute scaling).
    Used to isolate Isaac Lab vs Isaac Gym physics drift from policy variance."""

    hammer_only: bool = False
    """If True, override task.env.handleHeadTypes to ('hammer',) so the
    procedural pool emits only hammer URDFs (paired with the collapsed
    OBJECT_SIZE_DISTRIBUTIONS hammer-cuboid entry on both sides this gives
    byte-equivalent first-asset URDFs across envs)."""

    fixed_goal_pose: Optional[Tuple[float, float, float, float, float, float, float]] = None
    """Pin the goal to a single env-local pose (x, y, z, qx, qy, qz, qw) — xyzw
    quaternion convention (matches isaacgym's root_state_tensor)."""

    video_path: Optional[Path] = None
    """If set, attach a camera sensor matching the sim eval rig (pose
    (0,-1,1.03)→(0,0,0.53), 640×480, ~47.10° FoV) and write an mp4 here.
    Skipped entirely when None — the camera setup is the slow path."""

    video_fps: int = 30
    """Mp4 fps (only meaningful when video_path is set)."""

    frame_every: int = 2
    """Capture one frame every N policy steps (matches sim eval default)."""


def _build_env(args: Args):
    """Create the legacy SimToolReal env with all DR + curricula off so the
    obs trace is determined entirely by (seed, policy)."""
    overrides = {
            # 1 env, no viewer, no video.
            "task.env.numEnvs": 1,
            "task.env.envSpacing": 0.4,
            "task.env.capture_video": False,

            # Reset randomness off (so initial state is reproducible).
            "task.env.resetPositionNoiseX": 0.0,
            "task.env.resetPositionNoiseY": 0.0,
            "task.env.resetPositionNoiseZ": 0.0,
            "task.env.randomizeObjectRotation": False,
            "task.env.resetDofPosRandomIntervalFingers": 0.0,
            "task.env.resetDofPosRandomIntervalArm": 0.0,
            "task.env.resetDofVelRandomInterval": 0.0,
            "task.env.tableResetZRange": 0.0,

            # Action / obs delays + noise off.
            "task.env.useActionDelay": False,
            "task.env.useObsDelay": False,
            "task.env.useObjectStateDelayNoise": False,
            "task.env.objectScaleNoiseMultiplierRange": [1.0, 1.0],

            # Force / torque DR off (zeros; lifted-only flags don't matter).
            "task.env.forceScale": 0.0,
            "task.env.torqueScale": 0.0,
            "task.env.linVelImpulseScale": 0.0,
            "task.env.angVelImpulseScale": 0.0,
            "task.env.forceProbRange": [0.0001, 0.0001],
            "task.env.torqueProbRange": [0.0001, 0.0001],
            "task.env.linVelImpulseProbRange": [0.0001, 0.0001],
            "task.env.angVelImpulseProbRange": [0.0001, 0.0001],

            # Don't drop the episode mid-trace if the object falls.
            "task.env.resetWhenDropped": False,
            # Long episode so num_steps controls termination.
            "task.env.episodeLength": max(args.num_steps + 60, 600),

            # Seed.
            "seed": args.seed,
        }
    if args.hammer_only:
        overrides["task.env.handleHeadTypes"] = ["hammer"]
    if args.fixed_goal_pose is not None:
        # Single-state trajectory: legacy env cycles successes % len, so 1
        # entry pins every reset to the same pose. Quat is xyzw.
        overrides["task.env.useFixedGoalStates"] = True
        overrides["task.env.fixedGoalStates"] = [list(args.fixed_goal_pose)]
        overrides["task.env.fixedGoalStatesJsonPath"] = None
    return create_env(
        config_path=str(args.config_path),
        headless=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        overrides=overrides,
    )


def main():
    args = tyro.cli(Args)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval-base] config={args.config_path}")
    print(f"[eval-base] checkpoint={args.checkpoint_path}")
    print(f"[eval-base] output={args.output_npz}, steps={args.num_steps}, seed={args.seed}")

    env = _build_env(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optional camera sensor matching the historical Isaac Lab comparison rig.
    # Skipped entirely when video_path is None.
    camera_handle = None
    cam_props = None
    frames: list = []
    if args.video_path is not None:
        cam_props = gymapi.CameraProperties()
        cam_props.width = 640
        cam_props.height = 480
        # PinholeCameraCfg(focal=24mm, aperture=20.955mm) ⇒ ~47.10° FoV.
        cam_props.horizontal_fov = math.degrees(
            2.0 * math.atan(20.955 / (2.0 * 24.0))
        )
        env_ptr = env.envs[0]
        camera_handle = env.gym.create_camera_sensor(env_ptr, cam_props)
        env.gym.set_camera_location(
            camera_handle, env_ptr,
            gymapi.Vec3(0.0, -1.0, 1.03),
            gymapi.Vec3(0.0, 0.0, 0.53),
        )

    n_obs = 140
    n_act = 29
    player = RlPlayer(
        num_observations=n_obs,
        num_actions=n_act,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        num_envs=1,
    )
    player.player.init_rnn()

    # Legacy joint-name list straight off the actor handle.
    joint_names = list(env.joint_names) if hasattr(env, "joint_names") else []
    print(f"[eval-base] env.joint_names ({len(joint_names)}):")
    for i, n in enumerate(joint_names):
        marker = " " if (i < len(JOINT_NAMES_CANONICAL) and n == JOINT_NAMES_CANONICAL[i]) else "*"
        print(f"  [{i:2d}]{marker}{n}")

    # Reset (matches dextoolbench/eval.py: zero-action step is the env's
    # reset trigger after construction).
    obs_dict, _, _, _ = env.step(torch.zeros((1, n_act), device=device))
    obs = obs_dict["obs"]

    obs_log = np.zeros((args.num_steps, n_obs), dtype=np.float32)
    action_log = np.zeros((args.num_steps, n_act), dtype=np.float32)
    joint_pos_log = np.zeros((args.num_steps, n_act), dtype=np.float32)
    joint_vel_log = np.zeros((args.num_steps, n_act), dtype=np.float32)
    joint_targets_log = np.zeros((args.num_steps, n_act), dtype=np.float32)
    object_state_log = np.zeros((args.num_steps, 13), dtype=np.float32)
    goal_pose_log = np.zeros((args.num_steps, 7), dtype=np.float32)
    reward_log = np.zeros(args.num_steps, dtype=np.float32)

    for step in range(args.num_steps):
        # Capture obs + raw state BEFORE action (matches what the policy sees).
        obs_log[step] = obs[0].detach().cpu().numpy()
        joint_pos_log[step] = env.arm_hand_dof_pos[0].detach().cpu().numpy()
        joint_vel_log[step] = env.arm_hand_dof_vel[0].detach().cpu().numpy()
        object_state_log[step] = env.object_state[0].detach().cpu().numpy()
        goal_pose_log[step] = env.goal_pose[0].detach().cpu().numpy()

        if args.hold:
            # Hold mode: arm action=0 (velocity-delta keeps arm at
            # prev_target = current pos); hand action = inverse of
            # absolute scaling so target = current hand q.
            jpos = env.arm_hand_dof_pos
            lower = env.arm_hand_dof_lower_limits[:n_act]
            upper = env.arm_hand_dof_upper_limits[:n_act]
            action = torch.zeros_like(jpos)
            action[:, 7:] = (
                2.0 * (jpos[:, 7:] - lower[7:]) / (upper[7:] - lower[7:]) - 1.0
            )
        else:
            action = player.get_normalized_action(
                obs, deterministic_actions=args.deterministic_actions
            )
        action_log[step] = action[0].detach().cpu().numpy()

        obs_dict, reward, _done, _ = env.step(action)
        obs = obs_dict["obs"]
        reward_log[step] = float(reward[0].detach().cpu().item())

        # Capture target AFTER the step so we record what PhysX actually
        # saw on this step (legacy apply_actions sets prev_targets = cur_targets
        # at the end). prev_targets is what `set_dof_position_target_tensor`
        # pushed to PhysX during this step's decimation ticks.
        joint_targets_log[step] = env.prev_targets[0, :n_act].detach().cpu().numpy()

        if camera_handle is not None and step % args.frame_every == 0:
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            rgba = env.gym.get_camera_image(
                env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR
            )
            rgba = rgba.reshape(cam_props.height, cam_props.width, 4)
            frames.append(rgba[:, :, :3])

        if step % 60 == 0:
            print(f"[eval-base] step {step:4d}  reward={reward_log[step]:+.4f}")

    np.savez(
        str(args.output_npz),
        obs=obs_log,
        action=action_log,
        joint_pos=joint_pos_log,
        joint_vel=joint_vel_log,
        joint_targets=joint_targets_log,
        object_state=object_state_log,
        goal_pose=goal_pose_log,
        reward=reward_log,
        joint_names=np.array(joint_names),
    )
    print(f"[eval-base] saved {args.output_npz}")

    if frames:
        import imageio
        args.video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(args.video_path), frames, fps=args.video_fps)
        print(f"[eval-base] saved {args.video_path} — {len(frames)} frames @ {args.video_fps} fps")


if __name__ == "__main__":
    main()
