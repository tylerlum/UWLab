#!/usr/bin/env bash
# Launch a FurnitureBench square-leg success-metric/contact ablation matrix.
#
# Default mode is a dry run:
#   bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_success_metric_ablation.sh
#
# Submit from a Slurm login node:
#   DRY_RUN=0 bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_success_metric_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/furniturebench_leg_sapg_finetune.sub}"

DRY_RUN="${DRY_RUN:-1}"
SUBMIT_CLUSTER="${SUBMIT_CLUSTER:-1}"

WANDB_GROUP="${WANDB_GROUP:-2026-05-06_leg_success_metric_ablation01}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000000}"
HORIZON_LENGTH="${HORIZON_LENGTH:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"
SBATCH_MEM="${SBATCH_MEM:-96000}"

A5000_TARGET="${A5000_TARGET:---nodelist=move3}"
L40S_TARGET="${L40S_TARGET:---nodelist=move4}"
RTX6000_TARGET="${RTX6000_TARGET:---nodelist=move5}"

COMMON_EXPORTS=(
    "INIT_MODE=upright_fixed"
    "FORCE_LIFTED_FOR_KEYPOINT_REWARD=false"
    "HOVER_HEIGHT=0.4"
    "RANDOM_START_XY_CENTER=[0.20,0.08]"
    "RANDOM_START_XY_RANGE=[0.0,0.0]"
    "RANDOM_START_Z_RANGE=0.0"
    "RANDOM_START_YAW_RANGE_DEG=0.0"
    "RANDOM_START_ROLL_PITCH_RANGE_DEG=[0.0,0.0]"
    "LIFTING_REW_SCALE=20.0"
    "LIFTING_BONUS=300.0"
    "FORCE_SCALE=0.0"
    "TORQUE_SCALE=0.0"
    "OBJECT_SCALE_NOISE_RANGE=[1.0,1.0]"
    "ENABLE_RETRACT=false"
    "FORCE_CONSECUTIVE_NEAR_GOAL=true"
    "SUCCESS_STEPS=10"
    "CAPTURE_VIDEO=false"
    "CAPTURE_VIEWER_LEN=600"
    "CAPTURE_VIEWER_INTERVAL=6000"
)

run_sbatch() {
    local label="$1"
    local target_args_text="$2"
    shift 2
    local exports=(
        "REPO_ROOT=$REPO_ROOT"
        "WANDB_GROUP=$WANDB_GROUP"
        "MAX_ITERATIONS=$MAX_ITERATIONS"
    )
    local env_kv
    for env_kv in "${COMMON_EXPORTS[@]}" "$@"; do
        exports+=("$env_kv")
    done
    local target_args=()
    if [[ -n "$target_args_text" ]]; then
        read -r -a target_args <<< "$target_args_text"
    fi
    local cmd=(env "${exports[@]}" sbatch --job-name="$label" "${target_args[@]}" --mem="$SBATCH_MEM" --export=ALL "$SBATCH_SCRIPT")

    printf '\n# %s\n' "$label"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "$DRY_RUN" == "0" && "$SUBMIT_CLUSTER" == "1" ]]; then
        "${cmd[@]}"
    fi
}

submit_job() {
    local gpu_kind="$1"
    local target_args="$2"
    local num_envs="$3"
    local block_size="$4"
    local goal_mode="$5"
    local success_mode="$6"
    local fixture_xy="$7"
    local success_tolerance="$8"
    local omni_pos_tol="$9"
    local omni_ori_tol="${10}"
    local seed="${11}"
    local label_suffix="${12}"
    local run_tag="fbleg_${gpu_kind}_${label_suffix}_${goal_mode}_${success_mode}_seed${seed}"
    local minibatch_size
    minibatch_size=$((num_envs * HORIZON_LENGTH))

    run_sbatch "$run_tag" "$target_args" \
        "GPU_KIND=$gpu_kind" \
        "RUN_TAG=$run_tag" \
        "GOAL_MODE=$goal_mode" \
        "SUCCESS_MODE=$success_mode" \
        "FIXTURE_XY_OFFSET=$fixture_xy" \
        "GOAL_XY_OFFSET=[0.0,0.0]" \
        "SUCCESS_TOLERANCE=$success_tolerance" \
        "TARGET_SUCCESS_TOLERANCE=$success_tolerance" \
        "OMNIRESET_POSITION_SUCCESS_THRESHOLD=$omni_pos_tol" \
        "OMNIRESET_ORIENTATION_SUCCESS_THRESHOLD=$omni_ori_tol" \
        "NUM_ENVS=$num_envs" \
        "EXPL_COEF_BLOCK_SIZE=$block_size" \
        "HORIZON_LENGTH=$HORIZON_LENGTH" \
        "SEQ_LENGTH=$SEQ_LENGTH" \
        "MINIBATCH_SIZE=$minibatch_size" \
        "SEED=$seed"
}

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "Missing Slurm script: $SBATCH_SCRIPT" >&2
    exit 1
fi

if [[ "$DRY_RUN" == "0" && "$SUBMIT_CLUSTER" == "1" ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch is not available. Run this from a Slurm login node, or leave DRY_RUN=1." >&2
    exit 1
fi

echo "Repo root:    $REPO_ROOT"
echo "Slurm script: $SBATCH_SCRIPT"
echo "W&B group:    $WANDB_GROUP"
echo "Dry run:      $DRY_RUN"

# Positive controls: threaded fixture is shifted away, goals stay at the
# canonical hole target, and success is the strict SimToolReal keypoint metric.
submit_job "a5000" "$A5000_TARGET" 1536 256 "highHover" "simtoolreal_keypoints" "[-0.25,0.0]" 0.01 0.03 0.15 60 "ctrl_fixture_away"
submit_job "a5000" "$A5000_TARGET" 1536 256 "highHoverAndFinal" "simtoolreal_keypoints" "[-0.25,0.0]" 0.01 0.03 0.15 61 "ctrl_fixture_away"
submit_job "a5000" "$A5000_TARGET" 1536 256 "dense" "simtoolreal_keypoints" "[-0.25,0.0]" 0.01 0.03 0.15 62 "ctrl_fixture_away"

# Real fixture, relaxed OmniReset final alignment. Non-final goals still use
# strict 0.01 SimToolReal keypoint success, so dense waypoints cannot be skipped
# by the old loose 0.075 tolerance.
submit_job "l40s" "$L40S_TARGET" 3072 512 "highHoverAndFinal" "omnireset_alignment" "[0.0,0.0]" 0.01 0.03 0.15 63 "real_fixture_relaxed_omni"
submit_job "l40s" "$L40S_TARGET" 3072 512 "dense" "omnireset_alignment" "[0.0,0.0]" 0.01 0.03 0.15 64 "real_fixture_relaxed_omni"
submit_job "l40s" "$L40S_TARGET" 3072 512 "preInsertAndFinal" "omnireset_alignment" "[0.0,0.0]" 0.01 0.03 0.15 65 "real_fixture_relaxed_omni"

# Real fixture, strict SimToolReal keypoint success. OmniReset alignment
# diagnostics are still logged for apples-to-apples comparison.
submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 "highHoverAndFinal" "simtoolreal_keypoints" "[0.0,0.0]" 0.01 0.03 0.15 66 "real_fixture_keypoint"
submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 "dense" "simtoolreal_keypoints" "[0.0,0.0]" 0.01 0.03 0.15 67 "real_fixture_keypoint"
submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 "preInsertAndFinal" "simtoolreal_keypoints" "[0.0,0.0]" 0.01 0.03 0.15 68 "real_fixture_keypoint"
