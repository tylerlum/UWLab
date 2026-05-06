#!/usr/bin/env bash
# Launch matrix for the first FurnitureBench square-leg SimToolReal finetunes.
#
# Default mode is a dry run:
#   bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_10run_matrix.sh
#
# Submit the 9 cluster jobs from a Slurm login node:
#   DRY_RUN=0 bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_10run_matrix.sh
#
# The 10th run is local-only and is printed as a command because it should be
# started deliberately on the workstation GPU.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/furniturebench_leg_sapg_finetune.sub}"

DRY_RUN="${DRY_RUN:-1}"
SUBMIT_CLUSTER="${SUBMIT_CLUSTER:-1}"
PRINT_LOCAL="${PRINT_LOCAL:-1}"

WANDB_GROUP="${WANDB_GROUP:-2026-05-06_leg_finetune_matrix01}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000000}"
HORIZON_LENGTH="${HORIZON_LENGTH:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"
SBATCH_MEM="${SBATCH_MEM:-128000}"
SUCCESS_MODE="${SUCCESS_MODE:-omnireset_alignment}"
ENABLE_RETRACT="${ENABLE_RETRACT:-false}"
FORCE_CONSECUTIVE_NEAR_GOAL="${FORCE_CONSECUTIVE_NEAR_GOAL:-false}"

# Override these if the cluster node layout changes.  The current Slurm
# features are coarse (for example `24G,turing`), so node pinning is the
# clearest way to request exactly the intended GPU type.
A5000_TARGET="${A5000_TARGET:---nodelist=move3}"
L40S_TARGET="${L40S_TARGET:---nodelist=move4}"
RTX6000_TARGET="${RTX6000_TARGET:---nodelist=move5}"

run_sbatch() {
    local label="$1"
    local target_args_text="$2"
    shift 2
    local exports=(
        "ALL"
        "REPO_ROOT=$REPO_ROOT"
        "WANDB_GROUP=$WANDB_GROUP"
        "MAX_ITERATIONS=$MAX_ITERATIONS"
    )
    local env_kv
    for env_kv in "$@"; do
        exports+=("$env_kv")
    done
    local export_arg
    export_arg="$(IFS=,; echo "${exports[*]}")"
    local target_args=()
    if [[ -n "$target_args_text" ]]; then
        read -r -a target_args <<< "$target_args_text"
    fi
    local cmd=(sbatch --job-name="$label" "${target_args[@]}" --mem="$SBATCH_MEM" "--export=$export_arg" "$SBATCH_SCRIPT")

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
    local init_mode="$6"
    local seed="$7"
    local run_tag="fbleg_${gpu_kind}_${goal_mode}_${init_mode}_seed${seed}"
    local minibatch_size
    minibatch_size=$((num_envs * HORIZON_LENGTH))

    run_sbatch "$run_tag" "$target_args" \
        GPU_KIND="$gpu_kind" \
        RUN_TAG="$run_tag" \
        GOAL_MODE="$goal_mode" \
        INIT_MODE="$init_mode" \
        SUCCESS_MODE="$SUCCESS_MODE" \
        ENABLE_RETRACT="$ENABLE_RETRACT" \
        FORCE_CONSECUTIVE_NEAR_GOAL="$FORCE_CONSECUTIVE_NEAR_GOAL" \
        NUM_ENVS="$num_envs" \
        EXPL_COEF_BLOCK_SIZE="$block_size" \
        HORIZON_LENGTH="$HORIZON_LENGTH" \
        SEQ_LENGTH="$SEQ_LENGTH" \
        MINIBATCH_SIZE="$minibatch_size" \
        SEED="$seed"
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

# A5000 jobs: lower VRAM, keep the easier partial-assembly init and sweep goals.
submit_job "a5000" "$A5000_TARGET" 3072 512 "finalGoalOnly" "omnireset_partial_assemblies" 42
submit_job "a5000" "$A5000_TARGET" 3072 512 "preInsertAndFinal" "omnireset_partial_assemblies" 43
submit_job "a5000" "$A5000_TARGET" 3072 512 "dense" "omnireset_partial_assemblies" 44

# L40S jobs: same goal sweep from upright fixed starts.
submit_job "l40s" "$L40S_TARGET" 6144 1024 "finalGoalOnly" "upright_fixed" 45
submit_job "l40s" "$L40S_TARGET" 6144 1024 "preInsertAndFinal" "upright_fixed" 46
submit_job "l40s" "$L40S_TARGET" 6144 1024 "dense" "upright_fixed" 47

# RTX PRO 6000 jobs: hardest random-table starts get the largest batch.
submit_job "rtx6000" "$RTX6000_TARGET" 12288 2048 "finalGoalOnly" "random_table" 48
submit_job "rtx6000" "$RTX6000_TARGET" 12288 2048 "preInsertAndFinal" "random_table" 49
submit_job "rtx6000" "$RTX6000_TARGET" 12288 2048 "dense" "random_table" 50

if [[ "$PRINT_LOCAL" == "1" ]]; then
    cat <<EOF

# Local workstation run, start manually when the GPU is free.
OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 \\
env_uwlab/bin/python -u isaacsimenvs/train.py \\
  --task Isaacsimenvs-FurnitureBenchLeg-Direct-v0 \\
  --agent rl_games_sapg_cfg_entry_point \\
  --checkpoint .pretrained_checkpoints/SimToolReal/pretrained_policy/model.pth \\
  --checkpoint_load_mode weights \\
  --headless \\
  --capture_viewer \\
  --capture_viewer_len 600 \\
  --capture_viewer_interval 6000 \\
  --wandb_activate \\
  --wandb_project UWLab-SimToolReal-FurnitureBenchLeg \\
  --wandb_group "$WANDB_GROUP" \\
  --wandb_name 0_fbleg_local_dense_partial_seed51 \\
  env.scene.num_envs=3072 \\
  env.furniturebench_leg.goal_mode=dense \\
  env.furniturebench_leg.initialization_mode=omnireset_partial_assemblies \\
  env.furniturebench_leg.success_mode="$SUCCESS_MODE" \\
  env.furniturebench_leg.enable_retract="$ENABLE_RETRACT" \\
  env.termination.force_consecutive_near_goal_steps="$FORCE_CONSECUTIVE_NEAR_GOAL" \\
  agent.params.config.max_epochs="$MAX_ITERATIONS" \\
  agent.params.config.horizon_length="$HORIZON_LENGTH" \\
  agent.params.config.seq_length="$SEQ_LENGTH" \\
  agent.params.config.minibatch_size=$((3072 * HORIZON_LENGTH)) \\
  agent.params.config.central_value_config.minibatch_size=$((3072 * HORIZON_LENGTH)) \\
  agent.params.config.expl_coef_block_size=512 \\
  agent.params.config.name=0_fbleg_local_dense_partial_seed51 \\
  agent.params.seed=51 \\
  hydra.run.dir=outputs/train_fbleg_local_dense_partial_seed51
EOF
fi
