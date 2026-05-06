#!/usr/bin/env bash
# Launch the second FurnitureBench square-leg finetune matrix.
#
# This matrix starts from the SimToolReal pretrained policy with easy upright,
# deterministic object starts, high-hover support, and no forced-lift shortcut.
# It includes fixture-away keypoint positive controls and fixture-present
# OmniReset-threaded runs.
#
# Dry run:
#   bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_high_hover_controls_matrix.sh
#
# Submit from a Slurm login node:
#   DRY_RUN=0 bash isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_high_hover_controls_matrix.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/furniturebench_leg_sapg_finetune.sub}"

DRY_RUN="${DRY_RUN:-1}"
SUBMIT_CLUSTER="${SUBMIT_CLUSTER:-1}"
PRINT_LOCAL="${PRINT_LOCAL:-1}"

WANDB_GROUP="${WANDB_GROUP:-2026-05-06_leg_high_hover_controls_matrix03_relaxed_omnireset}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000000}"
HORIZON_LENGTH="${HORIZON_LENGTH:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-16}"
SBATCH_MEM="${SBATCH_MEM:-220000}"

A5000_TARGET="${A5000_TARGET:---nodelist=move3}"
L40S_TARGET="${L40S_TARGET:---nodelist=move4}"
RTX6000_TARGET="${RTX6000_TARGET:---nodelist=move5}"

COMMON_EXPORTS=(
    "INIT_MODE=upright_fixed"
    "FORCE_LIFTED_FOR_KEYPOINT_REWARD=false"
    "HOVER_HEIGHT=0.4"
    "SUCCESS_TOLERANCE=0.075"
    "TARGET_SUCCESS_TOLERANCE=0.01"
    "OMNIRESET_POSITION_SUCCESS_THRESHOLD=0.02"
    "OMNIRESET_ORIENTATION_SUCCESS_THRESHOLD=0.10"
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
    "CAPTURE_VIDEO=false"
    "VIDEO_INTERVAL=6000"
    "VIDEO_CAPTURE_FRAMES=600"
    "VIDEO_FPS=30"
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
    local goal_xy="$8"
    local seed="$9"
    local label_suffix="${10}"
    local run_tag="fbleg_${gpu_kind}_${label_suffix}_${goal_mode}_seed${seed}"
    local minibatch_size
    minibatch_size=$((num_envs * HORIZON_LENGTH))

    run_sbatch "$run_tag" "$target_args" \
        "GPU_KIND=$gpu_kind" \
        "RUN_TAG=$run_tag" \
        "GOAL_MODE=$goal_mode" \
        "SUCCESS_MODE=$success_mode" \
        "FIXTURE_XY_OFFSET=$fixture_xy" \
        "GOAL_XY_OFFSET=$goal_xy" \
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

# Fixture-away controls: goal remains at the normal hole target, physical
# threaded tabletop is shifted aside, success is SimToolReal keypoint success.
# Keep A5000 controls smaller: 3 x 3072-env rendering jobs exceeded the
# 125 GB cgroup limit during Isaac scene startup in matrix01.
submit_job "a5000" "$A5000_TARGET" 1536 256 "highHover" "simtoolreal_keypoints" "[-0.25,0.0]" "[0.0,0.0]" 42 "ctrl_fixture_away"
submit_job "a5000" "$A5000_TARGET" 1536 256 "highHoverAndFinal" "simtoolreal_keypoints" "[-0.25,0.0]" "[0.0,0.0]" 43 "ctrl_fixture_away"
submit_job "a5000" "$A5000_TARGET" 1536 256 "finalGoalOnly" "simtoolreal_keypoints" "[-0.25,0.0]" "[0.0,0.0]" 44 "ctrl_fixture_away"

# Fixture-present runs: normal threaded tabletop, final goal uses OmniReset's
# assembled-frame position + roll/pitch alignment. These are the main training
# probes; env counts are also conservative to avoid startup cgroup OOM.
submit_job "l40s" "$L40S_TARGET" 3072 512 "highHoverAndFinal" "omnireset_alignment" "[0.0,0.0]" "[0.0,0.0]" 45 "real_fixture"
submit_job "l40s" "$L40S_TARGET" 3072 512 "dense" "omnireset_alignment" "[0.0,0.0]" "[0.0,0.0]" 46 "real_fixture"

submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 "preInsertAndFinal" "omnireset_alignment" "[0.0,0.0]" "[0.0,0.0]" 47 "real_fixture"
submit_job "rtx6000" "$RTX6000_TARGET" 3072 512 "dense" "omnireset_alignment" "[0.0,0.0]" "[0.0,0.0]" 48 "real_fixture"

if [[ "$PRINT_LOCAL" == "1" ]]; then
    cat <<EOF

# Local workstation canary. Keep this smaller than cluster jobs; 1536 envs
# with MP4 capture can OOM a 24 GB 4090 during Isaac Sim startup/training.
OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 \\
env_uwlab/bin/python -u isaacsimenvs/train.py \\
  --task Isaacsimenvs-FurnitureBenchLeg-Direct-v0 \\
  --agent rl_games_sapg_cfg_entry_point \\
  --checkpoint .pretrained_checkpoints/SimToolReal/pretrained_policy/model.pth \\
  --checkpoint_load_mode weights \\
  --headless \\
  --capture_viewer \\
  --capture_viewer_len 600 \\
  --capture_viewer_interval 3000 \\
  --wandb_activate \\
  --wandb_project UWLab-SimToolReal-FurnitureBenchLeg \\
  --wandb_group "$WANDB_GROUP" \\
  --wandb_name 0_fbleg_local384_real_fixture_highHoverAndFinal_seed51 \\
  env.scene.num_envs=384 \\
  env.furniturebench_leg.goal_mode=highHoverAndFinal \\
  env.furniturebench_leg.initialization_mode=upright_fixed \\
  env.furniturebench_leg.success_mode=omnireset_alignment \\
  env.furniturebench_leg.force_lifted_for_keypoint_reward=false \\
  env.furniturebench_leg.hover_height=0.4 \\
  env.furniturebench_leg.omnireset_position_success_threshold=0.02 \\
  env.furniturebench_leg.omnireset_orientation_success_threshold=0.10 \\
  env.furniturebench_leg.random_start_xy_center=[0.20,0.08] \\
  env.furniturebench_leg.random_start_xy_range=[0.0,0.0] \\
  env.furniturebench_leg.random_start_z_range=0.0 \\
  env.furniturebench_leg.random_start_yaw_range_deg=0.0 \\
  env.furniturebench_leg.random_start_roll_pitch_range_deg=[0.0,0.0] \\
  env.reward.lifting_rew_scale=20.0 \\
  env.reward.lifting_bonus=300.0 \\
  env.domain_randomization.force_scale=0.0 \\
  env.domain_randomization.torque_scale=0.0 \\
  env.domain_randomization.object_scale_noise_multiplier_range=[1.0,1.0] \\
  agent.params.config.max_epochs="$MAX_ITERATIONS" \\
  agent.params.config.horizon_length="$HORIZON_LENGTH" \\
  agent.params.config.seq_length="$SEQ_LENGTH" \\
  agent.params.config.minibatch_size=$((384 * HORIZON_LENGTH)) \\
  agent.params.config.central_value_config.minibatch_size=$((384 * HORIZON_LENGTH)) \\
  agent.params.config.expl_coef_block_size=64 \\
  agent.params.config.name=0_fbleg_local384_real_fixture_highHoverAndFinal_seed51 \\
  agent.params.seed=51 \\
  hydra.run.dir=outputs/train_fbleg_local384_real_fixture_highHoverAndFinal_seed51
EOF
fi
