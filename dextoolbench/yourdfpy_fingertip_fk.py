"""Parser-independent fingertip FK via yourdfpy.

Two modes:

1. Default:      single-shot FK at the env default arm pose (hand=0).
2. --from-dumps: walk the per-step joint_pos arrays in
                 /tmp/simtoolreal_{base,lab}_obs.npz, compute yourdfpy
                 ground-truth fingertip_pos_rel_palm_center at each
                 step's joint configuration, and tabulate the per-step
                 drift vs legacy (Isaac Gym) and lab (Isaac Lab).
                 Tells us whether the 2mm drift is constant in world or
                 varies with arm orientation.

The legacy .npz file is produced by dextoolbench/eval_simtoolreal_base.py.
The Isaac Lab comparison dump was produced by the retired DirectRLEnv eval
script and is kept as a historical debugging input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yourdfpy


REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = (
    REPO_ROOT
    / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
)


# Same default arm pose as scene_utils.ARM_DEFAULT_JOINT_POS.
ARM_DEFAULT = {
    "iiwa14_joint_1": -1.571,
    "iiwa14_joint_2": 1.571,
    "iiwa14_joint_3": 0.0,
    "iiwa14_joint_4": 1.376,
    "iiwa14_joint_5": 0.0,
    "iiwa14_joint_6": 1.485,
    "iiwa14_joint_7": 1.308,
}

# Robot root pose (matches scene_utils robot init_state).
ROBOT_ROOT_POS = np.array([0.0, 0.8, 0.0])

PALM_OFFSET = np.array([-0.0, -0.02, 0.16], dtype=np.float64)
FINGERTIP_OFFSET = np.array([0.02, 0.002, 0.0], dtype=np.float64)

PALM_LINK = "iiwa14_link_7"
FINGERTIP_DP_LINKS = [
    "left_index_DP", "left_middle_DP", "left_ring_DP",
    "left_thumb_DP", "left_pinky_DP",
]
FINGERTIP_ELASTOMER_LINKS = [
    "left_index_elastomer", "left_middle_elastomer", "left_ring_elastomer",
    "left_thumb_elastomer", "left_pinky_elastomer",
]
FINGERTIP_LEAF_LINKS = [
    "left_index_fingertip", "left_middle_fingertip", "left_ring_fingertip",
    "left_thumb_fingertip", "left_pinky_fingertip",
]

# Canonical 29-dim ordering used by the eval dumps.
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

# Obs slice for fingertip_pos_rel_palm in the 140-dim policy obs.
OBS_SLICE_FINGERTIP_REL_PALM = (98, 113)


def fingertip_rel_palm_via_yourdfpy(
    urdf: yourdfpy.URDF, joint_pos_canonical: np.ndarray
) -> np.ndarray:
    """Compute (5, 3) fingertip_pos_rel_palm_center for a 29-dim canonical
    joint vector — pure URDF FK + palm/fingertip offsets."""
    cfg = {
        name: float(joint_pos_canonical[i])
        for i, name in enumerate(JOINT_NAMES_CANONICAL)
    }
    urdf.update_cfg(cfg)

    def link_T(name: str) -> np.ndarray:
        return urdf.get_transform(name)

    palm_T = link_T(PALM_LINK)
    palm_pos = palm_T[:3, 3] + ROBOT_ROOT_POS
    palm_rot = palm_T[:3, :3]
    palm_center = palm_pos + palm_rot @ PALM_OFFSET

    out = np.zeros((5, 3))
    for i, dp in enumerate(FINGERTIP_DP_LINKS):
        T = link_T(dp)
        dp_pos = T[:3, 3] + ROBOT_ROOT_POS
        dp_rot = T[:3, :3]
        ft_offset = dp_pos + dp_rot @ FINGERTIP_OFFSET
        out[i] = ft_offset - palm_center
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-dumps", action="store_true",
                        help="Per-step diff against /tmp/simtoolreal_{base,lab}_obs.npz")
    parser.add_argument("--legacy_npz", default="/tmp/simtoolreal_base_obs.npz")
    parser.add_argument("--lab_npz", default="/tmp/simtoolreal_lab_obs.npz")
    args = parser.parse_args()

    print(f"URDF: {URDF_PATH}")
    urdf = yourdfpy.URDF.load(str(URDF_PATH), build_collision_scene_graph=False)

    if not args.from_dumps:
        # Single-shot at the default arm pose
        default_q = np.zeros(29, dtype=np.float64)
        for i, name in enumerate(JOINT_NAMES_CANONICAL):
            default_q[i] = ARM_DEFAULT.get(name, 0.0)

        ft_rel = fingertip_rel_palm_via_yourdfpy(urdf, default_q)
        print(f"\nDefault arm pose, hand=0 → fingertip_pos_rel_palm_center:")
        for i, dp in enumerate(FINGERTIP_DP_LINKS):
            finger = dp.split("_")[1]
            print(f"  {finger:<8s}  {np.array2string(ft_rel[i], precision=4)}")
        print(f"\n(Reference legacy ft0 obs: [-0.0251, -0.1299,  0.0003])")
        print(f" Reference lab    ft0 obs: [-0.0251, -0.1302,  0.0022])")
        return

    # Per-step mode
    leg = np.load(args.legacy_npz, allow_pickle=True)
    lab = np.load(args.lab_npz, allow_pickle=True)
    n = min(leg["obs"].shape[0], lab["obs"].shape[0])

    lo, hi = OBS_SLICE_FINGERTIP_REL_PALM
    legacy_ft = leg["obs"][:n, lo:hi].reshape(n, 5, 3)
    lab_ft = lab["obs"][:n, lo:hi].reshape(n, 5, 3)
    legacy_q = leg["joint_pos"][:n]  # (n, 29) raw radians
    lab_q = lab["joint_pos"][:n]

    print(f"# Per-step yourdfpy FK vs (legacy, lab) for {n} steps")
    print(f"# Legend: drift = sim_obs - yourdfpy(sim_joint_pos)\n")

    rows = []
    for t in range(n):
        gt_legacy = fingertip_rel_palm_via_yourdfpy(urdf, legacy_q[t])
        gt_lab = fingertip_rel_palm_via_yourdfpy(urdf, lab_q[t])
        leg_drift = legacy_ft[t] - gt_legacy   # (5, 3)
        lab_drift = lab_ft[t] - gt_lab          # (5, 3)
        rows.append((t, leg_drift, lab_drift, legacy_q[t], lab_q[t]))

    # Print first / mid / last step + summary
    for label, t in (("step  0", 0), (f"step {n//2:2d}", n // 2),
                     (f"step {n-1:2d}", n - 1)):
        _, leg_d, lab_d, lq, llq = rows[t]
        print(f"  {label} (legacy joint[7]={lq[7]:+.3f}, lab joint[7]={llq[7]:+.3f})")
        for i in range(5):
            finger = FINGERTIP_DP_LINKS[i].split("_")[1]
            print(
                f"    {finger:<8s}  legacy_drift={np.array2string(leg_d[i], precision=4):<28s}  "
                f"lab_drift={np.array2string(lab_d[i], precision=4)}"
            )

    # Per-finger statistics across all steps
    print(f"\n# Per-finger drift statistics across {n} steps")
    print(f"  {'finger':<8s}  {'legacy mean':<28s}  {'legacy std':<28s}  "
          f"{'lab mean':<28s}  {'lab std':<28s}")
    leg_drifts = np.stack([r[1] for r in rows])  # (n, 5, 3)
    lab_drifts = np.stack([r[2] for r in rows])  # (n, 5, 3)
    for i in range(5):
        finger = FINGERTIP_DP_LINKS[i].split("_")[1]
        print(
            f"  {finger:<8s}  "
            f"{np.array2string(leg_drifts[:, i].mean(0), precision=4):<28s}  "
            f"{np.array2string(leg_drifts[:, i].std(0), precision=4):<28s}  "
            f"{np.array2string(lab_drifts[:, i].mean(0), precision=4):<28s}  "
            f"{np.array2string(lab_drifts[:, i].std(0), precision=4):<28s}"
        )

    # Is the legacy drift constant in world frame? (low std across steps → yes)
    leg_world_std = leg_drifts.std(axis=0).mean()
    lab_world_std = lab_drifts.std(axis=0).mean()
    print(f"\n# Drift std across steps (avg over 5 fingertips × 3 axes):")
    print(f"  legacy: {leg_world_std:.6f} m  ({'constant' if leg_world_std < 1e-3 else 'varies'} in world frame)")
    print(f"  lab:    {lab_world_std:.6f} m  ({'constant' if lab_world_std < 1e-3 else 'varies'} in world frame)")


if __name__ == "__main__":
    main()
