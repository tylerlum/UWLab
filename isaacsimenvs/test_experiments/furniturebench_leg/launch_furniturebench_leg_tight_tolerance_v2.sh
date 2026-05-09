#!/usr/bin/env bash
# Launch the 2026-05-09 FurnitureBench leg tight-tolerance rerun matrix.
#
# Dry run by default:
#   bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_tight_tolerance_v2.sh
#
# Submit from a Slurm login node:
#   DRY_RUN=0 bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_tight_tolerance_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${REPO_ROOT}/isaacsimenvs/test_experiments/furniturebench_leg/furniturebench_leg_sapg_finetune.sub}"

DRY_RUN="${DRY_RUN:-1}"
SUBMIT_CLUSTER="${SUBMIT_CLUSTER:-1}"
WANDB_GROUP="${WANDB_GROUP:-2026-05-09_leg_tight_tolerance_v2}"
MAX_ITERATIONS="${MAX_ITERATIONS:-400000}"
HORIZON_LENGTH="${HORIZON_LENGTH:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"

MOVE_TARGET="${MOVE_TARGET:---partition=move --account=move --nodelist=move3}"
JUNO_TARGET="${JUNO_TARGET:---partition=juno --account=juno --nodelist=juno2}"
MOVE_NODE="${MOVE_NODE:-move3}"
JUNO_NODE="${JUNO_NODE:-juno2}"

fallback_mem_for_node() {
    local node="$1"
    if [[ "$node" == "$JUNO_NODE" ]]; then
        echo "${SBATCH_MEM_JUNO_FALLBACK:-96000}"
    else
        echo "${SBATCH_MEM_MOVE_FALLBACK:-96000}"
    fi
}

mem_for_node() {
    local node="$1"
    if ! command -v scontrol >/dev/null 2>&1; then
        fallback_mem_for_node "$node"
        return
    fi

    local info real_mem gpu_count mem
    info="$(scontrol show node "$node" 2>/dev/null || true)"
    real_mem="$(grep -o 'RealMemory=[0-9]*' <<< "$info" | head -1 | cut -d= -f2)"
    gpu_count="$(grep -o 'gres/gpu=[0-9]*' <<< "$info" | head -1 | cut -d= -f2)"
    if [[ -z "$gpu_count" ]]; then
        gpu_count="$(grep -o 'gpu:[^, ]*:[0-9]*' <<< "$info" | head -1 | awk -F: '{print $NF}')"
    fi
    if [[ -z "$real_mem" || -z "$gpu_count" || "$gpu_count" -le 0 ]]; then
        fallback_mem_for_node "$node"
        return
    fi

    mem=$(( real_mem * 85 / 100 / gpu_count ))
    if (( mem < 64000 )); then
        mem=64000
    fi
    echo "$mem"
}

MOVE_MEM="${SBATCH_MEM_MOVE:-$(mem_for_node "$MOVE_NODE")}"
JUNO_MEM="${SBATCH_MEM_JUNO:-$(mem_for_node "$JUNO_NODE")}"

COMMON_EXPORTS=(
    "REPO_ROOT=$REPO_ROOT"
    "WANDB_GROUP=$WANDB_GROUP"
    "MAX_ITERATIONS=$MAX_ITERATIONS"
    "INIT_MODE=upright_fixed"
    "GOAL_MODE=preInsertDenseFinal"
    "SUCCESS_MODE=simtoolreal_keypoints"
    "ENABLE_RETRACT=false"
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
    "FORCE_CONSECUTIVE_NEAR_GOAL=true"
    "SUCCESS_STEPS=10"
    "DENSE_SCREW_TURNS=1.0"
    "SCREW_METRIC_MIN_TURNS_FOR_INSERT=0.5"
    "SCREW_METRIC_HOLE_RADIUS=0.02"
    "FINAL_SUCCESS_REQUIRES_SCREW_INSERT_LIKE=false"
    "LIFTING_REW_SCALE=20.0"
    "LIFTING_BONUS=300.0"
    "FORCE_SCALE=0.0"
    "TORQUE_SCALE=0.0"
    "OBJECT_SCALE_NOISE_RANGE=[1.0,1.0]"
    "CAPTURE_VIDEO=false"
    "CAPTURE_VIEWER_LEN=7200"
    "CAPTURE_VIEWER_INTERVAL=12000"
    "CAPTURE_VIEWER_FULL_EPISODES=true"
    "CAPTURE_VIEWER_EPISODES=1"
    "NUM_ENVS=3072"
    "EXPL_COEF_BLOCK_SIZE=512"
    "HORIZON_LENGTH=$HORIZON_LENGTH"
    "SEQ_LENGTH=$SEQ_LENGTH"
    "MINIBATCH_SIZE=$((3072 * HORIZON_LENGTH))"
)

run_sbatch() {
    local label="$1"
    local target_args_text="$2"
    local mem_mb="$3"
    shift 3
    local exports=()
    local env_kv
    for env_kv in "${COMMON_EXPORTS[@]}" "$@"; do
        exports+=("$env_kv")
    done
    local target_args=()
    if [[ -n "$target_args_text" ]]; then
        read -r -a target_args <<< "$target_args_text"
    fi
    local cmd=(env "${exports[@]}" sbatch --job-name="$label" "${target_args[@]}" --mem="$mem_mb" --export=ALL "$SBATCH_SCRIPT")

    printf '\n# %s\n' "$label"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [[ "$DRY_RUN" == "0" && "$SUBMIT_CLUSTER" == "1" ]]; then
        "${cmd[@]}"
    fi
}

submit_job() {
    local target_args="$1"
    local mem_mb="$2"
    local gpu_kind="$3"
    local dense_steps="$4"
    local tolerance="$5"
    local final_success_steps="$6"
    local seed="$7"
    local suffix="$8"
    local run_tag="fbleg_${gpu_kind}_${suffix}_pd${dense_steps}_tol${tolerance//./}_nogate_seed${seed}"

    run_sbatch "$run_tag" "$target_args" "$mem_mb" \
        "GPU_KIND=$gpu_kind" \
        "RUN_TAG=$run_tag" \
        "DENSE_DESCEND_STEPS=$dense_steps" \
        "SUCCESS_TOLERANCE=$tolerance" \
        "TARGET_SUCCESS_TOLERANCE=$tolerance" \
        "FINAL_SUCCESS_STEPS=$final_success_steps" \
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

echo "Repo root:     $REPO_ROOT"
echo "Slurm script:  $SBATCH_SCRIPT"
echo "W&B group:     $WANDB_GROUP"
echo "Dry run:       $DRY_RUN"
echo "Move target:   $MOVE_TARGET mem=${MOVE_MEM}MB"
echo "Juno target:   $JUNO_TARGET mem=${JUNO_MEM}MB"

submit_job "$MOVE_TARGET" "$MOVE_MEM" "a5000" 10 "0.002" "null" 210 "tight"
submit_job "$MOVE_TARGET" "$MOVE_MEM" "a5000" 10 "0.003" "null" 211 "tight"
submit_job "$MOVE_TARGET" "$MOVE_MEM" "a5000" 10 "0.005" "null" 212 "tight"
submit_job "$MOVE_TARGET" "$MOVE_MEM" "a5000" 3 "0.002" "null" 213 "tight"
submit_job "$MOVE_TARGET" "$MOVE_MEM" "a5000" 3 "0.003" "null" 214 "tight"
submit_job "$MOVE_TARGET" "$MOVE_MEM" "a5000" 3 "0.005" "null" 215 "tight"

submit_job "$JUNO_TARGET" "$JUNO_MEM" "juno2" 10 "0.003" "60" 216 "finalhold60"
submit_job "$JUNO_TARGET" "$JUNO_MEM" "juno2" 3 "0.003" "60" 217 "finalhold60"
