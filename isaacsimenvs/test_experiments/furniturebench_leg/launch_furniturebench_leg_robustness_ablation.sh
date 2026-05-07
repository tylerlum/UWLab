#!/usr/bin/env bash
# Launch FurnitureBench square-leg robustness/contact ablations.
#
# Default mode is a dry run:
#   bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_robustness_ablation.sh
#
# Submit from a Slurm login node:
#   DRY_RUN=0 bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_robustness_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/furniturebench_leg_sapg_finetune.sub}"

DRY_RUN="${DRY_RUN:-1}"
SUBMIT_CLUSTER="${SUBMIT_CLUSTER:-1}"

WANDB_GROUP="${WANDB_GROUP:-2026-05-07_leg_robustness_ablation01}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400000}"
HORIZON_LENGTH="${HORIZON_LENGTH:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"
SBATCH_MEM="${SBATCH_MEM:-96000}"

A5000_TARGET="${A5000_TARGET:---nodelist=move3}"
RTX6000_TARGET="${RTX6000_TARGET:---nodelist=move5}"

COMMON_EXPORTS=(
    "INIT_MODE=upright_fixed"
    "SUCCESS_MODE=simtoolreal_keypoints"
    "FORCE_LIFTED_FOR_KEYPOINT_REWARD=false"
    "HOVER_HEIGHT=0.4"
    "FIXTURE_XY_OFFSET=[0.0,0.0]"
    "FIXTURE_RANDOM_XY_RANGE=[0.0,0.0]"
    "GOAL_XY_OFFSET=[0.0,0.0]"
    "RANDOM_START_XY_CENTER=[0.20,0.08]"
    "RANDOM_START_XY_RANGE=[0.0,0.0]"
    "RANDOM_START_Z_RANGE=0.0"
    "RANDOM_START_YAW_RANGE_DEG=0.0"
    "RANDOM_START_ROLL_PITCH_RANGE_DEG=[0.0,0.0]"
    "SUCCESS_TOLERANCE=0.01"
    "TARGET_SUCCESS_TOLERANCE=0.01"
    "OMNIRESET_POSITION_SUCCESS_THRESHOLD=0.01"
    "OMNIRESET_ORIENTATION_SUCCESS_THRESHOLD=0.05"
    "SUCCESS_STEPS=10"
    "DENSE_DESCEND_STEPS=10"
    "DENSE_SCREW_TURNS=1.0"
    "SCREW_METRIC_MIN_TURNS_FOR_INSERT=0.5"
    "SCREW_METRIC_HOLE_RADIUS=0.02"
    "FINAL_SUCCESS_REQUIRES_SCREW_INSERT_LIKE=true"
    "LIFTING_REW_SCALE=20.0"
    "LIFTING_BONUS=300.0"
    "FORCE_SCALE=0.0"
    "TORQUE_SCALE=0.0"
    "OBJECT_SCALE_NOISE_RANGE=[1.0,1.0]"
    "ENABLE_RETRACT=false"
    "FORCE_CONSECUTIVE_NEAR_GOAL=true"
    "CAPTURE_VIDEO=false"
    "CAPTURE_VIEWER_LEN=7200"
    "CAPTURE_VIEWER_INTERVAL=12000"
    "CAPTURE_VIEWER_FULL_EPISODES=true"
    "CAPTURE_VIEWER_EPISODES=1"
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
    local seed="$5"
    local label_suffix="$6"
    shift 6
    local run_tag="fbleg_${gpu_kind}_${label_suffix}_seed${seed}"
    local minibatch_size
    minibatch_size=$((num_envs * HORIZON_LENGTH))

    run_sbatch "$run_tag" "$target_args" \
        "GPU_KIND=$gpu_kind" \
        "RUN_TAG=$run_tag" \
        "NUM_ENVS=$num_envs" \
        "EXPL_COEF_BLOCK_SIZE=$block_size" \
        "HORIZON_LENGTH=$HORIZON_LENGTH" \
        "SEQ_LENGTH=$SEQ_LENGTH" \
        "MINIBATCH_SIZE=$minibatch_size" \
        "SEED=$seed" \
        "$@"
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

# Remove the final screw gate and the high-hover scaffold. If this still gets
# high screw_insert_like, the dense helical goals alone are doing the teaching.
submit_job "a5000" "$A5000_TARGET" 3072 512 90 "nogate_preDense10_noHigh" \
    "GOAL_MODE=preInsertDenseFinal" \
    "FINAL_SUCCESS_REQUIRES_SCREW_INSERT_LIKE=false"

# Same no-high-hover setting, but keep the final screw gate.
submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 91 "gate_preDense10_noHigh" \
    "GOAL_MODE=preInsertDenseFinal"

# Reduce helical supervision density.
submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 92 "gate_dense5" \
    "GOAL_MODE=dense" \
    "DENSE_DESCEND_STEPS=5"
submit_job "a5000" "$A5000_TARGET" 3072 512 93 "gate_dense3" \
    "GOAL_MODE=dense" \
    "DENSE_DESCEND_STEPS=3"
submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 94 "gate_preFinal_sparse" \
    "GOAL_MODE=preInsertAndFinal"

# Moderate and strong relative object-start randomization.
submit_job "a5000" "$A5000_TARGET" 3072 512 95 "gate_objRand_mod" \
    "GOAL_MODE=dense" \
    "INIT_MODE=random_table" \
    "RANDOM_START_XY_RANGE=[0.04,0.04]" \
    "RANDOM_START_Z_RANGE=0.02" \
    "RANDOM_START_YAW_RANGE_DEG=45.0" \
    "RANDOM_START_ROLL_PITCH_RANGE_DEG=[20.0,20.0]"
submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 96 "gate_objRand_strong" \
    "GOAL_MODE=dense" \
    "INIT_MODE=random_table" \
    "RANDOM_START_XY_RANGE=[0.08,0.08]" \
    "RANDOM_START_Z_RANGE=0.03" \
    "RANDOM_START_YAW_RANGE_DEG=180.0" \
    "RANDOM_START_ROLL_PITCH_RANGE_DEG=[60.0,60.0]"

# Shift the whole fixture/goal/task frame per episode. This tests workspace
# randomization without adding relative start difficulty.
submit_job "a5000" "$A5000_TARGET" 3072 512 97 "gate_fixtureRand_mod" \
    "GOAL_MODE=dense" \
    "FIXTURE_RANDOM_XY_RANGE=[0.03,0.03]"

# Combine moderate relative object-start randomization with per-episode fixture
# XY randomization.
submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 98 "gate_fixtureObjRand_mod" \
    "GOAL_MODE=dense" \
    "INIT_MODE=random_table" \
    "FIXTURE_RANDOM_XY_RANGE=[0.03,0.03]" \
    "RANDOM_START_XY_RANGE=[0.04,0.04]" \
    "RANDOM_START_Z_RANGE=0.02" \
    "RANDOM_START_YAW_RANGE_DEG=45.0" \
    "RANDOM_START_ROLL_PITCH_RANGE_DEG=[20.0,20.0]"

# Lower high-hover control: still dense, but the first target is less far above
# the hole.
submit_job "a5000" "$A5000_TARGET" 3072 512 99 "gate_lowHover20cm" \
    "GOAL_MODE=dense" \
    "HOVER_HEIGHT=0.2"
