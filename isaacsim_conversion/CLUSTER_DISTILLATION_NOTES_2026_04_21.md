# Cluster Distillation Notes

This is a handoff note for running Isaac Sim distillation jobs on the Stanford
Slurm cluster from the current image-distillation branch.

## Current Repo State

- Local repo: `/home/tylerlum/github_repos/depthbasedRL`
- Cluster normal clone: `/move/u/tylerlum/github_repos/depthbasedRL`
- Cluster RTX 6000 clone: `/move/u/tylerlum/github_repos/depthbasedRL_rtx6000`
- Active branch: `2026-04-20_dextrah_style_distillation`
- Standard cluster env: `.venv-isaacsim-py311`
- RTX 6000 env: `.venv-isaacsim-rtx6000-cu128-py311`

Before launching cluster jobs, push locally and fast-forward the cluster clone:

```bash
ssh tylerlum@sc
cd /move/u/tylerlum/github_repos/depthbasedRL
git fetch origin 2026-04-20_dextrah_style_distillation
git checkout 2026-04-20_dextrah_style_distillation
git pull --ff-only
```

## Useful Status Commands

```bash
ssh tylerlum@sc
squeue -u tylerlum -o "%.18i %.9P %.30j %.8u %.2t %.10M %.10l %.12R"
/juno/u/tylerlum/sgpu_tyler.py -p move,move-interactive,humanoid,humanoid-interactive
```

For job details:

```bash
sacct -j <JOBID> --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList -P
tail -f /move/u/tylerlum/github_repos/depthbasedRL/slurm_logs/<job_name>_<JOBID>.out
tail -f /move/u/tylerlum/github_repos/depthbasedRL/slurm_logs/<job_name>_<JOBID>_nvidia_smi.csv
```

## GPU Selection

- Prefer `L40S` for Isaac Sim image training when available.
- `RTX PRO 6000` can work with the separate RTX clone/env, but the normal env
  has had CUDA architecture compatibility issues on that hardware.
- `A5000` works but is slower and has less VRAM; do not assume an unconstrained
  `#SBATCH --partition=move` job will land on L40S.
- If a run must use L40S, pin a known L40S node such as `move4` or `humanoid1`
  with `#SBATCH --nodelist=<node>`. If none are free, the job should pend rather
  than silently take an A5000.

## Slurm Gotchas

- `sbatch` can submit successfully and still exit immediately if the script was
  corrupted by shell quoting. Always check `sacct` and the output log.
- Avoid complex nested heredocs over `ssh`; write scripts locally and `scp`, or
  keep tracked scripts under `scripts/cluster/`.
- Use job-local caches under `/tmp/${USER}/...` to avoid shared Omniverse cache
  lock conflicts between Isaac Sim jobs.
- Always set `OMNI_KIT_ACCEPT_EULA=YES`.
- W&B API key is available at `/juno/u/tylerlum/.wandb_api_key`.
- Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for large image runs.

## Current Image Job Pattern

Most successful image jobs use:

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
  --mode train_online --headless --task_source dextoolbench \
  --object_category hammer --object_name claw_hammer --task_name swing_down \
  --student_input camera --student_arch mono_transformer_recurrent \
  --num_envs <N> --env_spacing 4.0 --ground_plane_size 500 --camera_backend tiled \
  --distill_config <distill_config.yaml> \
  --camera_config <camera_config.yaml> \
  --teacher_checkpoint pretrained_policy/model.pth \
  --teacher_config pretrained_policy/config.yaml \
  --wandb --wandb_entity tylerlum --wandb_project depthbasedRL-isaacsim-distill \
  --capture_viewer --capture_viewer_video --capture_viewer_len 600 \
  --capture_viewer_env_id 0 \
  --capture_viewer_wandb_key interactive_viewer \
  --capture_viewer_video_wandb_key policy_input_video \
  --capture_viewer_video_fps 60
```

Use tracked scripts when possible:

```bash
sbatch scripts/cluster/sbatch_img80_wrist_depth_l40s.sh
```

## Rendering And Camera Gotchas

- Use `TiledCameraCfg` for both third-person and wrist image policies.
- Wrist tiled cameras should be spawned under the wrist link with the configured
  local offset. Do not move wrist tiled cameras each step with `set_world_poses`.
- Third-person tiled cameras are spawned at their per-env local pose because
  dynamic pose writes were unreliable for render products.
- Set `ground_plane_size=500`; Isaac Lab's default ground plane can be too small
  for large env grids with `env_spacing=4.0`, causing far envs to see a white
  background instead of the table/ground appearance.
- Saved policy videos are low-res by design for these 80/160/320 ablations.

## Current Depth Windows

- Third-person depth/RGBD used `window_normalize` with roughly `0.55m..1.60m`.
- Wrist depth should use a closer window. Current first choice:
  `depth_min_m=0.0`, `depth_max_m=1.0`.
- The W&B video for depth-only runs is a visualization fallback from the depth
  buffer, not an RGB render.

## Active Jobs To Recognize

Job names used in this series:

- `img80_rgb_third_u1`
- `img80_depth_third_u1`
- `img80_rgbd_third_u1`
- `img80_rgb_wrist_v9_u1`
- `img80_depth_wrist_v9_u1`

Check W&B run links in the Slurm log via:

```bash
grep -E "View run|wandb: Syncing run|online iter|Traceback|RuntimeError|CUDA out|ERROR" \
  /move/u/tylerlum/github_repos/depthbasedRL/slurm_logs/<job_name>_<JOBID>.out
```
