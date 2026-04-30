"""Phase 2 smoke test: verify full inference pipeline works without any simulator."""

import numpy as np
import torch

from deployment.rl_player import RlPlayer
from isaacgymenvs.utils.observation_action_utils_sharpa import (
    JOINT_NAMES_ISAACGYM,
    N_OBS,
    compute_joint_pos_targets,
    compute_observation,
    create_urdf_object,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 2b: Test observation computation ---
    print("Loading URDF for FK...")
    urdf = create_urdf_object("iiwa14_left_sharpa_adjusted_restricted")
    assert JOINT_NAMES_ISAACGYM == urdf.actuated_joint_names, (
        f"Joint name mismatch!\n"
        f"  Expected: {JOINT_NAMES_ISAACGYM}\n"
        f"  Got:      {urdf.actuated_joint_names}"
    )
    print(f"URDF joint ordering matches JOINT_NAMES_ISAACGYM ({len(JOINT_NAMES_ISAACGYM)} joints)")

    # --- 2c: obs_list from training config ---
    obs_list = [
        "joint_pos", "joint_vel", "prev_action_targets", "palm_pos",
        "palm_rot", "object_rot", "fingertip_pos_rel_palm",
        "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
    ]

    # Dummy state
    N = 1
    q = np.zeros((N, 29), dtype=np.float32)
    qd = np.zeros((N, 29), dtype=np.float32)
    prev_targets = np.zeros((N, 29), dtype=np.float32)
    object_pose = np.array([[0.0, 0.0, 0.7, 0, 0, 0, 1]], dtype=np.float32)
    goal_pose = np.array([[0.0, 0.0, 0.8, 0, 0, 0, 1]], dtype=np.float32)
    object_scales = np.array([[0.141, 0.03025, 0.0271]], dtype=np.float32)

    print("Computing observation from dummy state...")
    obs = compute_observation(
        q=q, qd=qd, prev_action_targets=prev_targets,
        object_pose=object_pose, goal_object_pose=goal_pose,
        object_scales=object_scales, urdf=urdf, obs_list=obs_list,
    )
    assert obs.shape == (1, N_OBS), f"obs.shape={obs.shape}, expected (1, {N_OBS})"
    assert not np.any(np.isnan(obs)), "NaN in observation!"
    assert not np.any(np.isinf(obs)), "Inf in observation!"
    print(f"Observation OK: shape={obs.shape}, min={obs.min():.4f}, max={obs.max():.4f}")

    # Sanity check sub-components
    obs_ranges = {
        "joint_pos":     (0,  29),
        "joint_vel":     (29, 58),
        "prev_targets":  (58, 87),
        "palm_pos":      (87, 90),
        "palm_rot":      (90, 94),
        "object_rot":    (94, 98),
        "fingertip_rel": (98, 113),
        "kp_rel_palm":   (113, 125),
        "kp_rel_goal":   (125, 137),
        "obj_scales":    (137, 140),
    }
    for name, (s, e) in obs_ranges.items():
        v = obs[0, s:e]
        print(f"  {name:20s}: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")

    assert np.allclose(obs[0, 137:140], [0.141, 0.03025, 0.0271]), "object_scales mismatch"
    palm_quat = obs[0, 90:94]
    assert abs(np.linalg.norm(palm_quat) - 1.0) < 0.01, f"palm_rot not unit quat: norm={np.linalg.norm(palm_quat)}"

    # --- Load policy and run full pipeline ---
    print("\nLoading policy...")
    player = RlPlayer(
        num_observations=140, num_actions=29,
        config_path="pretrained_policy/config.yaml",
        checkpoint_path="pretrained_policy/model.pth",
        device=device, num_envs=1,
    )

    obs_tensor = torch.from_numpy(obs).float().to(device)
    action = player.get_normalized_action(obs_tensor, deterministic_actions=True)
    assert action.shape == (1, 29), f"action.shape={action.shape}"
    print(f"Action OK: shape={action.shape}, min={action.min():.4f}, max={action.max():.4f}")

    # Compute joint targets
    targets = compute_joint_pos_targets(
        actions=action.cpu().numpy(),
        prev_targets=prev_targets,
        hand_moving_average=0.1,
        arm_moving_average=0.1,
        hand_dof_speed_scale=1.5,
        dt=1 / 60,
    )
    assert targets.shape == (1, 29), f"targets.shape={targets.shape}"
    print(f"Targets OK: shape={targets.shape}, min={targets.min():.4f}, max={targets.max():.4f}")

    print("\n=== Full inference pipeline OK ===")


if __name__ == "__main__":
    main()
