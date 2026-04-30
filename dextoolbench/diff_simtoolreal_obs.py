"""Diff per-step obs traces from the legacy isaacgym SimToolReal env vs the
new isaacsimenvs (Isaac Lab) SimToolReal env.

Inputs are legacy .npz dumps produced by dextoolbench/eval_simtoolreal_base.py
and historical Isaac Lab comparison dumps from the retired DirectRLEnv eval
script.

Both runs use the same pretrained policy and the same DR-off configuration,
so initial-state and step dynamics differences flag *port-level* bugs, not
policy stochasticity.

Per-key report: shape, max abs diff, mean abs diff, first divergent step.

Note: these are independent processes with independent RNG seeds and
different physics backends — *some* divergence accumulates after step 0
even with identical action streams. Pay attention to step-0 magnitudes
and to obs slices that should be deterministic given joint state alone
(e.g., the joint_pos slice 0..29 of the 140-dim obs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# Layout of the 140-dim policy obs (matches obs_list default in both envs):
#   0..29   joint_pos             (29)
#   29..58  joint_vel             (29)
#   58..87  prev_action_targets   (29)
#   87..90  palm_pos              (3)
#   90..94  palm_rot              (4)  wxyz
#   94..98  object_rot            (4)  wxyz
#   98..113 fingertip_pos_rel_palm (5*3 = 15)
#   113..125 keypoints_rel_palm   (4*3 = 12)
#   125..137 keypoints_rel_goal   (4*3 = 12)
#   137..140 object_scales        (3)
OBS_SLICES = [
    ("joint_pos",              0,   29),
    ("joint_vel",              29,  58),
    ("prev_action_targets",    58,  87),
    ("palm_pos",               87,  90),
    ("palm_rot",               90,  94),
    ("object_rot",             94,  98),
    ("fingertip_pos_rel_palm", 98, 113),
    ("keypoints_rel_palm",    113, 125),
    ("keypoints_rel_goal",    125, 137),
    ("object_scales",         137, 140),
]


def _summary(name: str, a: np.ndarray, b: np.ndarray) -> None:
    """Print one row: max|Δ|, mean|Δ|, first divergent step (>1e-3),
    and t=0 values when the diff is large."""
    if a.shape != b.shape:
        print(f"{name:24s}  SHAPE MISMATCH legacy={a.shape} lab={b.shape}")
        return
    diff = np.abs(a - b)
    max_diff = diff.max()
    mean_diff = diff.mean()
    # First step where any element exceeds tolerance
    elem_max = diff.reshape(diff.shape[0], -1).max(axis=1)
    first_div = int(np.argmax(elem_max > 1e-3)) if (elem_max > 1e-3).any() else -1
    print(
        f"{name:24s}  max|Δ|={max_diff:11.4e}  mean|Δ|={mean_diff:11.4e}  "
        f"first_div_step={first_div}"
    )
    if max_diff > 1e-2:
        leg0 = a[0].reshape(-1)
        lab0 = b[0].reshape(-1)
        n = min(8, leg0.size)
        print(f"  legacy[0,:{n}] = {np.array2string(leg0[:n], precision=4)}")
        print(f"  lab   [0,:{n}] = {np.array2string(lab0[:n], precision=4)}")
        print(f"  diff  [0,:{n}] = {np.array2string((leg0 - lab0)[:n], precision=4)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", default="/tmp/simtoolreal_base_obs.npz",
                        help="Path to legacy (isaacgym) obs dump")
    parser.add_argument("--lab", default="/tmp/simtoolreal_lab_obs.npz",
                        help="Path to new (Isaac Lab) obs dump")
    args = parser.parse_args()

    leg = np.load(args.legacy, allow_pickle=True)
    lab = np.load(args.lab, allow_pickle=True)

    n = min(leg["obs"].shape[0], lab["obs"].shape[0])
    print(f"# Comparing first {n} steps: legacy={args.legacy}, lab={args.lab}\n")

    # Joint name comparison
    leg_names = [str(x) for x in leg["joint_names"]]
    lab_names = [str(x) for x in lab["joint_names"]]
    if leg_names != lab_names:
        print("WARN: joint_names differ between dumps (canonical reorder may be off).")
        for i, (a, b) in enumerate(zip(leg_names, lab_names)):
            tag = "  " if a == b else "**"
            print(f"  [{i:2d}]{tag} legacy={a:30s} lab={b}")
        print()

    # Top-level tensor diffs (raw state we logged outside the obs vector)
    print("# Aux tensors (raw state at each step)")
    for k in ("joint_pos", "joint_vel", "object_state", "goal_pose", "action", "reward"):
        if k in leg.files and k in lab.files:
            _summary(k, leg[k][:n], lab[k][:n])
    print()

    # Per-slice obs diffs
    print("# Policy obs slices (140-dim)")
    leg_obs = leg["obs"][:n]
    lab_obs = lab["obs"][:n]
    _summary("OBS_TOTAL", leg_obs, lab_obs)
    for name, lo, hi in OBS_SLICES:
        _summary(name, leg_obs[:, lo:hi], lab_obs[:, lo:hi])


if __name__ == "__main__":
    main()
