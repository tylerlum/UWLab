# Isaac Sim Policy Rollout

Roll out pretrained Isaac Gym policies in Isaac Sim (via Isaac Lab) for visualization/evaluation.

**Status**: All 12 goals of the beam assembly task completed successfully in Isaac Sim using the FGT curriculum policy trained in Isaac Gym.

## Prerequisites

- Python 3.11 (Isaac Sim 5.x / Isaac Lab 2.3.x requirement)
- NVIDIA GPU with driver >= 525.60 (we have 570.124.06)
- CUDA 12+ (we have 12.8)
- `uv` for package management

## Installation

All commands assume you're in the repo root.

This setup intentionally uses a separate environment from the Isaac Gym
training environment in [docs/installation.md](../docs/installation.md).

- Isaac Gym / training env: Python 3.8, usually `.venv`
- Isaac Sim conversion env: Python 3.11, `.venv-isaacsim-py311`

Do not delete or repurpose your Isaac Gym environment for this flow.

To suppress the Omniverse EULA prompt in non-interactive runs:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

The helper script `./scripts/run_in_isaacsim_env.sh` sets this automatically if
you have not already set it.

It also adds Isaac Lab's bundled source tree to `PYTHONPATH`, which is needed
for imports such as `isaaclab.sim` in this pip-installed layout.

### 1. Create Python 3.11 venv

```bash
uv venv .venv-isaacsim-py311 --python 3.11
source .venv-isaacsim-py311/bin/activate
```

If Python 3.11 is not available: `uv python install 3.11`

Or use the repo helper:

```bash
./scripts/setup_isaacsim_uv_env.sh
```

### 2. Install PyTorch (CUDA 12.1)

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

### 3. Install vendored rl_games and inference deps

```bash
uv pip install -e ./rl_games/
uv pip install omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro
```

**Do NOT** run `uv pip install -e .` — the project's pyproject.toml pins `numpy==1.23.0`, `isaacgym-stubs`, and `warp-lang==0.10.1` which conflict with Python 3.11 / Isaac Sim.

### 4. Install Isaac Lab (bundles Isaac Sim)

```bash
uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
```

This is a ~15GB download. The `[isaacsim,all]` extra bundles Isaac Sim + all extensions + Isaac Lab modules.

**First launch notes**:
- Isaac Sim will prompt to accept the NVIDIA EULA on first run.
- First launch takes ~2-5 minutes (shader compilation). Subsequent launches are ~16-30s.
- Shader cache lives in `~/.cache/ov/`. On NFS this is slow — see step 5 for speedup.

### 5. (Optional) Speed up startup with local cache

Isaac Sim startup is slow (~2 min) when caches are on NFS (`~/.cache/ov/` is on `portal-nfs-01`). On compute nodes with local SSD (`/scratch`):

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p $OMNI_KIT_CACHE_PATH
```

Note: `/scratch` is node-local and won't persist across different SLURM nodes.

### 6. Verify installation

```bash
# 1. Policy inference smoke test (no simulator needed)
PYTHONPATH=. python isaacsim_conversion/test_inference.py

# 2. Isaac Sim + Isaac Lab launch test (~2 min first time)
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
import isaaclab.sim as sim_utils
print('Isaac Sim + Isaac Lab OK')
app.close()
"
```

## Quick reference: full install from scratch

```bash
uv venv .venv-isaacsim-py311 --python 3.11
source .venv-isaacsim-py311/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv pip install -e ./rl_games/
uv pip install omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro
uv pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
```

Helper-script equivalent:

```bash
./scripts/setup_isaacsim_uv_env.sh
```

## Policy checkpoints and data

The environment setup does not automatically download policy or dataset assets.

- If you want the repo's released pretrained policy, run:

```bash
./scripts/run_in_isaacsim_env.sh python download_pretrained_policy.py
```

This creates:

```text
pretrained_policy/config.yaml
pretrained_policy/model.pth
```

- If you want to run a trained Fabrica policy, point `--checkpoint` at
  `last/model.pth` or `best/model.pth` from your training run and `--config` at
  the corresponding `config.yaml`.
- DexToolBench data is documented in the main [README.md](../README.md), but it
  is not required for the base Fabrica Isaac Sim rollout flow shown below.

## Usage

### Smoke test (inference only, no simulator)

```bash
source .venv-isaacsim-py311/bin/activate
PYTHONPATH=. python isaacsim_conversion/test_inference.py
```

Or without activating:

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/test_inference.py
```

### Full rollout (without video)

```bash
source .venv-isaacsim-py311/bin/activate
PYTHONPATH=. python isaacsim_conversion/rollout.py --headless --max_steps 700 \
    --checkpoint path/to/last/model.pth \
    --config path/to/config.yaml
```

### DexToolBench rollout example

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

## Distillation

The repo now includes a first-pass Isaac Sim distillation path for the fixed
DexToolBench hammer task:

- task: `hammer / claw_hammer / swing_down`
- teacher: existing low-dimensional Isaac Gym policy
- student default: mono transformer + recurrence
- student modality default: `depth`
- auxiliary head default: `object_pos`

### Frame conventions and DEXTRAH reference

This distillation path now follows the same important env-local convention used
by DEXTRAH for auxiliary object position:

- DEXTRAH env computes object position as:
  - `object_pos = object.data.root_pos_w - scene.env_origins`
- DEXTRAH then exposes that through:
  - `aux_info["object_pos"]`
- DEXTRAH's DAgger code consumes that env output directly as the auxiliary
  supervision target.

This repo mirrors that convention:

- `object_pos` auxiliary target is **env-local world position**
- it is **not** camera-relative
- it is **not** robot-relative

For camera placement in cloned multi-env scenes:

- camera rotation is shared across envs
- camera translation is authored **env-locally** under each cloned env namespace
- env origins provide the world translation offset automatically

For the real-camera `T_W_C` path, the current implementation uses a ROS optical
camera convention:

- `convention="ros"`
- right: `+X`
- down: `+Y`
- forward: `+Z`

The source of truth is the raw calibrated transform in
`isaacsim_conversion/task_utils.py`.

### Camera sanity check

This uses the real-camera-inspired `T_W_R @ T_R_C` pose from the planning work
and saves RGB/depth snapshots under `distillation_runs/.../camera_debug/`.

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/camera_debug.py \
    --mode camera_debug \
    --headless \
    --camera_config isaacsim_conversion/configs/hammer_camera.yaml
```

### Teacher-only baseline

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill_eval.py \
    --mode teacher_eval \
    --headless \
    --teacher_checkpoint pretrained_policy/model.pth \
    --teacher_config pretrained_policy/config.yaml
```

### Student training

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
    --mode train \
    --headless \
    --teacher_checkpoint pretrained_policy/model.pth \
    --teacher_config pretrained_policy/config.yaml \
    --camera_config isaacsim_conversion/configs/hammer_camera.yaml \
    --distill_config isaacsim_conversion/configs/hammer_distill.yaml \
    --wandb
```

By default this writes to:

```text
distillation_runs/<timestamped_run>/
  resolved_distill_config.yaml
  metrics.csv
  checkpoints/student_latest.pt
  checkpoints/student_best.pt
  camera_debug/
```

### Cluster usage

For cluster runs on shared storage, use:

- shared repo clone under `/move/u/$USER/github_repos/depthbasedRL`
- repo-local Isaac Sim env `.venv-isaacsim-py311`
- `OMNI_KIT_CACHE_PATH=/tmp/$USER_ov_cache`

Bootstrap on an allocated node with:

```bash
./scripts/cluster/bootstrap_cluster_isaacsim.sh
```

Then launch the long teacher-observation distillation run with:

```bash
sbatch scripts/cluster/sbatch_distill_teacher_obs_l40s.sh
```

And evaluate a saved student checkpoint with:

```bash
sbatch --export=CHECKPOINT=distillation_runs/<run>/checkpoints/student_best.pt \
  scripts/cluster/sbatch_student_eval_l40s.sh
```

Training defaults in `hammer_distill.yaml` now use:

- depth images
- `mono_transformer_recurrent`
- `beta_mode: fixed_decay`
- `beta_decay: 0.1`
- randomized object starts

During training, the script now also:

- logs a one-time `teacher_eval_baseline`
- runs periodic `student_eval` every `eval_interval` episodes
- writes all of those rows into `metrics.csv`
- optionally logs them to Weights & Biases with `--wandb`

### Student evaluation

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill_eval.py \
    --mode student_eval \
    --headless \
    --student_checkpoint distillation_runs/<run>/checkpoints/student_best.pt \
    --teacher_checkpoint pretrained_policy/model.pth \
    --teacher_config pretrained_policy/config.yaml
```

You can override the beta schedule from the CLI without editing YAML:

```bash
--beta_mode fixed_decay --beta_start 1.0 --beta_end 0.0 --beta_decay 0.1
```

### Quick ablations

Disable auxiliary object-position loss:

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
    --mode train \
    --headless \
    --distill_config isaacsim_conversion/configs/hammer_distill.yaml \
    --student_modality depth
```

Then set `aux_object_pos_weight: 0.0` in
`isaacsim_conversion/configs/hammer_distill.yaml`.

Try RGB or RGB-D instead of depth:

```bash
--student_modality rgb
--student_modality rgbd
```

### Full rollout with video recording

```bash
source .venv-isaacsim-py311/bin/activate
# Kill any lingering Isaac Sim processes first!
ps aux | grep "python.*isaacsim\|python.*rollout" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null

PYTHONPATH=. python isaacsim_conversion/rollout.py --headless --enable_cameras --max_steps 700 \
    --checkpoint path/to/last/model.pth \
    --config path/to/config.yaml \
    --video_dir rollout_videos/my_experiment
```

### Example: beam assembly with FGT curriculum policy

```bash
FGT_BASE="train_dir/FABRICA_TRAINING/fabrica/no_dr_fgt_curriculum_beam_2_coacd_2026-03-18_01-34-01/runs/00_no_dr_fgt_curriculum_beam_2_coacd_2026-03-18_01-34-01"

PYTHONPATH=. python isaacsim_conversion/rollout.py --headless --enable_cameras --max_steps 700 \
    --checkpoint "$FGT_BASE/last/model.pth" \
    --config "$FGT_BASE/config.yaml" \
    --video_dir rollout_videos/fgt_beam
```

## Critical transfer details: Isaac Gym → Isaac Sim

These are the key findings that made the policy transfer work. Getting any of these wrong causes the policy to fail silently (robot moves but doesn't complete the task).

### 1. Drive gain compensation (180/pi)

Isaac Lab's `UrdfConverter` multiplies revolute joint stiffness/damping by `pi/180` internally (rad→deg conversion for USD). But our values from Isaac Gym are already in PhysX units. We pre-multiply by `180/pi` to cancel this out:

```python
JOINT_STIFFNESSES_COMPENSATED = {k: v * 180.0 / math.pi for k, v in JOINT_STIFFNESSES.items()}
```

Without this, drives are ~57x too weak and the robot barely moves.

### 2. contact_offset = 0.002

Training uses `contact_offset=0.002` (in `SimToolReal.yaml`). Isaac Sim defaults to `0.02` (10x larger). Must be set explicitly via `CollisionPropertiesCfg`:

```python
collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0)
```

### 3. Robot gravity disabled

Isaac Gym disables gravity on the robot (`asset_options.disable_gravity = True`). Must match in Isaac Sim:

```python
rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
```

### 4. Self-collision off

```python
articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)
```

### 5. Robot default arm pose

The robot arm starts at a specific configuration, NOT all zeros:

```python
default_arm_pos = [-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 1.308]  # SharPa mount
```

Hand joints start at all zeros.

### 6. Object scales = fixedSize from training config

The `object_scales` observation uses `fixedSize` from the training config (actual bounding box in meters), NOT `Object.scale` from the registry (which is the mesh scale factor):

```python
# CORRECT: from training config
fixedSize: [0.141, 0.03025, 0.0271]

# WRONG: from Object.scale (mesh scale factor, ~25x larger)
Object.scale: [1.905, 0.300, 0.288]
```

### 7. Joint ordering permutation

Isaac Gym uses depth-first ordering (all thumb joints, then all index joints...), Isaac Sim uses breadth-first (all first-level joints across fingers, then second-level...). A permutation array maps between them automatically.

### 8. Quaternion convention

Isaac Sim uses wxyz, Isaac Gym/observation code uses xyzw. Convert at the sim boundary:

```python
quat_xyzw = quat_wxyz[[1, 2, 3, 0]]  # wxyz → xyzw
```

### 9. Use `last/model.pth`, not early checkpoints

The `nn/` checkpoint is saved early in training and barely trained. Always use `last/model.pth` or `best/model.pth`.

### 10. merge_fixed_joints = True

Match Isaac Gym's `collapse_fixed_joints=True`. Without this, extra fixed-joint bodies cause collision issues.

## Video recording details

Video recording uses the Isaac Lab `Camera` sensor with `--enable_cameras` flag.

**Camera setup:**
- Camera sensor must be created AFTER `IsaacSimEnv.__init__()`, followed by `sim.reset()` to initialize internal buffers (`_timestamp`).
- Object poses set before `sim.reset()` get wiped — always place objects AFTER camera init.
- Use `convention="opengl"` for the `CameraCfg.OffsetCfg` (standard camera: -Z forward, +Y up).
- Camera position matching Isaac Gym viewer: `pos=(0, -1, 1.03)` looking at `target=(0, 0, 0.53)`.
- First run with `--enable_cameras` takes ~5-15 min for RTX shader compilation. Subsequent runs are cached.
- **Kill all old Isaac Sim processes before starting** — multiple processes on one GPU will hang.

## Architecture

The inference pipeline is simulator-agnostic:

```
Isaac Sim → q, qd, object_pose → compute_observation() → RlPlayer → compute_joint_pos_targets() → Isaac Sim
                                    (numpy, yourdfpy FK)     (rl_games)      (numpy)
```

Only 3 values come from the simulator per step:
- Joint positions (29)
- Joint velocities (29)
- Object world pose (7: xyz + quaternion xyzw)

Everything else (palm pose, fingertips, keypoints) is computed via forward kinematics in `observation_action_utils_sharpa.py`.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `test_inference.py` | Smoke test — full obs→policy→action pipeline without simulator |
| `test_camera.py` | Standalone camera sensor test |
| `isaacsim_env.py` | Isaac Lab scene setup, URDF import, state extraction, physics config |
| `rollout.py` | Main entry point — policy rollout with goal switching and video recording |
| `README.md` | This file |
