# RTX6000 Cluster Bring-Up Notes

This note captures the current understanding of the `rtxpro6000` path on the
cluster and how to test it without disturbing the working `L40S` setup.

## What failed and why

Two different issues showed up in the recent cluster batch:

### 1. GroundPlane collision on some teacher-obs configs

This is **not** caused by the new monitor-env feature.

The new monitor logic keeps a subset of the **existing** vectorized env batch
student-only. It does **not** create a second simulator env or a second stage.

The actual collision came from the older in-process eval path in
`isaacsim_conversion/distill.py`, which still runs when:

- `eval_interval` is active
- and `eval_num_envs > 0`

That path creates a second `IsaacSimDistillEnv` inside the same app/stage, which
recreates:

- `/World/GroundPlane`
- and other scene content

This is why configs like these are currently unsafe for long cluster jobs:

- `hammer_distill_teacher_obs_4096_hold_then_decay.yaml`
- `hammer_distill_teacher_obs_4096_mlp_hold.yaml`
- `hammer_distill_teacher_obs_4096_always_student.yaml`

because they still contain:

- `eval_interval: 1`
- `eval_num_envs: 128`

The safe pattern is:

- `eval_num_envs: 0`
- use the new `student_monitor` rows during training
- run separate `student_eval` jobs from checkpoints when needed

### 2. RTX6000 CUDA kernel-image failure

The `move5` jobs failed with:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

This is most likely a software-stack / binary compatibility issue for the
current `rtxpro6000` nodes, not a simple naming or cache issue.

So for `rtxpro6000`, the safest approach is to test in a **separate shared clone
and separate env**, rather than touching the working `L40S` setup.

## Recommended isolation strategy

Use a separate shared repo clone and env:

```text
/move/u/$USER/github_repos/depthbasedRL_rtx6000
/move/u/$USER/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-py311
```

Reason:

- keeps the working `L40S` setup intact
- lets us try different package versions or installer behavior
- avoids confusion about which environment works on which GPU family

Also use a per-job cache path:

```bash
export OMNI_KIT_CACHE_PATH=/tmp/$USER_ov_cache_$SLURM_JOB_ID
```

## Bring-up sequence for RTX6000

### 1. Get an interactive `move5` allocation

```bash
ssh tylerlum@sc

srun --account move -p move-interactive \
  --nodelist=move5 \
  --time=6:00:00 \
  --gres=gpu:1 \
  --mem=64G \
  --cpus-per-task=8 \
  --pty bash -i
```

### 2. Verify the node and GPU

```bash
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h / /move /tmp
```

### 3. Create an isolated shared clone

Because GitHub access from compute nodes was unreliable, the safest approach is
the same as the working `L40S` path:

- sync the local repo into shared storage from the local machine, or
- clone on login/shared side first, then use it from the compute node

Target:

```text
/move/u/$USER/github_repos/depthbasedRL_rtx6000
```

### 4. Build a separate env

From that clone:

```bash
export PATH=$HOME/.local/bin:$PATH
export OMNI_KIT_ACCEPT_EULA=YES
export OMNI_KIT_CACHE_PATH=/tmp/$USER_ov_cache_rtx6000
mkdir -p "$OMNI_KIT_CACHE_PATH"

cd /move/u/$USER/github_repos/depthbasedRL_rtx6000
./scripts/setup_isaacsim_uv_env.sh .venv-isaacsim-rtx6000-py311
```

### 5. Smoke tests

Run in order:

```bash
ISAACSIM_ENV_DIR=/move/u/$USER/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-py311 \
  ./scripts/run_in_isaacsim_env.sh python download_pretrained_policy.py

ISAACSIM_ENV_DIR=/move/u/$USER/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-py311 \
  ./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/test_inference.py

ISAACSIM_ENV_DIR=/move/u/$USER/github_repos/depthbasedRL_rtx6000/.venv-isaacsim-rtx6000-py311 \
  ./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill_eval.py \
    --mode teacher_eval \
    --headless \
    --max_steps 10 \
    --num_envs 1 \
    --teacher_checkpoint pretrained_policy/model.pth \
    --teacher_config pretrained_policy/config.yaml \
    --camera_config isaacsim_conversion/configs/hammer_camera_depth_160x90.yaml
```

If this still fails with the same CUDA kernel-image error, then the current
package stack is not usable on `move5` and we should stop there until we choose
alternate package versions.

## Current recommendation

- Keep using `L40S` for productive runs.
- Disable the old in-process eval path in configs used for cluster training:
  - set `eval_num_envs: 0`
- Use the new `student_monitor` rows as the live student-only signal.
- Treat `RTX6000` as a separate environment-debugging track with its own clone
  and env, not as a drop-in replacement for the working `L40S` setup.
