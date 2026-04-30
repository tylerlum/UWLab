# Isaac Sim Distillation Runbook

This file captures the current distillation status, the known-good commands,
and the cluster workflow for continuing from the current state.

## Current Branch And Scope

- Primary branch: `2026-04-15_distillation`
- Stable large-scale path:
  - `teacher_obs` student input
  - `4096` envs
  - direct simulator observations, not FK
- Not yet validated end-to-end:
  - camera-image student distillation at scale

## Current Findings

### Teacher / rollout status

- `isaacsim_conversion/rollout.py` is still the known-good viewer baseline.
- `distill_eval.py --mode teacher_eval` now matches the old rollout much more
  closely after:
  - fixing joint-order permutation parity with Isaac Gym
  - matching fingertip/contact material settings
  - using direct simulator observations instead of FK for scale

### Multi-env camera status

- Real-camera pose now uses the raw `T_W_R @ T_R_C` transform.
- The real-camera path is currently treated as a ROS optical camera pose:
  - `+Z` forward
  - `+Y` down
  - `+X` right
- Multi-env camera capture works and headless rendering now explicitly forces
  sensor updates so saved frames are not stale.
- ZED-like intrinsics are configured for:
  - width `960`
  - height `540`
  - `K = [[694.03, 0, 480.10], [0, 694.03, 282.77], [0, 0, 1]]`

### Distillation status

The easiest currently working training path is teacher-observation distillation.

Known experiment variants already added:

- `hammer_distill_teacher_obs_4096_hold_then_decay.yaml`
- `hammer_distill_teacher_obs_4096_hold_then_decay_long.yaml`
- `hammer_distill_teacher_obs_4096_fixed_slowbeta.yaml`
- `hammer_distill_teacher_obs_4096_always_student.yaml`
- `hammer_distill_teacher_obs_4096_mlp_hold.yaml`

Teacher baseline that was observed with large-batch direct-sim evaluation:

- `4096` envs, fixed start, `700` steps
- `goal_idx ~= 8/37`
- `goal_completion_ratio ~= 0.214`

Important interpretation note:

- Mixed-policy train metrics are **not** the same as pure student metrics.
- Early train episodes can look good while pure `student_eval` still fails.

### Known caveat

`distill.py` currently contains an experimental in-process eval path driven by
`eval_num_envs > 0`. That path creates a second env in the same app/stage and is
not the recommended pattern for cluster runs.

For long runs, prefer:

- `eval_num_envs: 0`
- checkpoint during training
- separate `student_eval` jobs from saved checkpoints

## Known-Good Commands

### Viewer baseline

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/rollout.py \
  --task_source dextoolbench \
  --object_category hammer \
  --object_name claw_hammer \
  --task_name swing_down \
  --max_steps 700 \
  --checkpoint pretrained_policy/model.pth \
  --config pretrained_policy/config.yaml
```

### Teacher eval

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill_eval.py \
  --mode teacher_eval \
  --headless \
  --max_steps 700 \
  --num_envs 4096 \
  --env_spacing 4.0 \
  --object_start_mode fixed \
  --teacher_checkpoint pretrained_policy/model.pth \
  --teacher_config pretrained_policy/config.yaml \
  --camera_config isaacsim_conversion/configs/hammer_camera_depth_160x90.yaml
```

### Teacher-observation training

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
  --mode train \
  --headless \
  --student_input teacher_obs \
  --student_arch mlp_recurrent \
  --distill_config isaacsim_conversion/configs/hammer_distill_teacher_obs_4096_hold_then_decay_long.yaml \
  --camera_config isaacsim_conversion/configs/hammer_camera_depth_160x90.yaml \
  --teacher_checkpoint pretrained_policy/model.pth \
  --teacher_config pretrained_policy/config.yaml
```

### Student eval

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill_eval.py \
  --mode student_eval \
  --headless \
  --student_input teacher_obs \
  --student_arch mlp_recurrent \
  --student_checkpoint distillation_runs/<run>/checkpoints/student_best.pt \
  --max_steps 700 \
  --num_envs 512 \
  --env_spacing 4.0 \
  --object_start_mode fixed \
  --teacher_checkpoint pretrained_policy/model.pth \
  --teacher_config pretrained_policy/config.yaml \
  --camera_config isaacsim_conversion/configs/hammer_camera_depth_160x90.yaml
```

## Cluster Workflow

Shared storage path:

```text
/move/u/$USER/github_repos/depthbasedRL
```

Recommended cache path on compute nodes:

```text
/tmp/$USER_ov_cache
```

Use `/tmp` by default because it is known to exist. If `/scratch` exists on the
allocated node, it can be used later as an override.

### Bring-up sequence

1. Push this branch.
2. Get an interactive `L40S` allocation on `move-interactive`.
3. Clone/update the repo in `/move/u/$USER/github_repos/depthbasedRL`.
4. Build `.venv-isaacsim-py311`.
5. Download the pretrained policy.
6. Run the smoke tests in order:
   - `test_inference.py`
   - short `teacher_eval`
   - then long `sbatch` jobs

### Recommended first long run

- GPU: `L40S`
- account: `move`
- training mode: `teacher_obs`
- schedule:
  - `beta_hold_episodes: 5`
  - `beta_decay: 0.05`
  - `num_episodes: 150`
- checkpoint every `5` episodes
- external eval only

## Cluster Commands

Interactive bring-up:

```bash
ssh tylerlum@sc

srun --account move -p move-interactive \
  --nodelist=move4 \
  --time=6:00:00 \
  --gres=gpu:1 \
  --mem=64G \
  --cpus-per-task=8 \
  --pty bash -i
```

Then:

```bash
cd /move/u/$USER/github_repos
git clone -b 2026-04-15_distillation https://github.com/kushal2000/depthbasedRL.git
cd depthbasedRL

export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH=/tmp/$USER_ov_cache
mkdir -p "$OMNI_KIT_CACHE_PATH"

./scripts/setup_isaacsim_uv_env.sh
./scripts/run_in_isaacsim_env.sh python download_pretrained_policy.py
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/test_inference.py
```

Batch train:

```bash
sbatch scripts/cluster/sbatch_distill_teacher_obs_l40s.sh
```

Batch eval:

```bash
sbatch --export=CHECKPOINT=distillation_runs/<run>/checkpoints/student_best.pt \
  scripts/cluster/sbatch_student_eval_l40s.sh
```
