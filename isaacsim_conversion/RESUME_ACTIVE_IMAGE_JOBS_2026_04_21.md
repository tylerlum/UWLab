# Resume Active Image Jobs

This note prepares resuming the currently running image-based distillation jobs
as new Slurm jobs with new WandB runs. It does not execute anything.

## Current Active Jobs

- `15209711` `img80_rgb_third_u1`
- `15209712` `img80_depth_third_u1`
- `15209713` `img80_rgbd_third_u1`
- `15210520` `img80_rgb_wrist_v9_u1`

## Current Latest Checkpoints

Normal clone:

- `/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/img80_rgb_third_u1_15209711_2048env/checkpoints/student_latest.pt`
- `/move/u/tylerlum/github_repos/depthbasedRL/distillation_runs/img80_rgb_wrist_v9_u1_tiled_2048_15210520/checkpoints/student_latest.pt`

RTX 6000 clone:

- `/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/distillation_runs/img80_depth_third_u1_15209712_2048env/checkpoints/student_latest.pt`
- `/move/u/tylerlum/github_repos/depthbasedRL_rtx6000/distillation_runs/img80_rgbd_third_u1_15209713_1536env/checkpoints/student_latest.pt`

## Important Resume Caveat

`distill.py` resumes:

- student weights
- optimizer state

But it does **not** currently restore the global online iteration counter into
the training loop or resume the same WandB run. Therefore resume means:

- new Slurm job
- new WandB run
- new run directory
- training continues from the saved model state
- logged `episode` / iteration numbers restart from zero in the new run

So this is a useful practical continuation, but not a perfect same-run resume.

## Prepared Resume Scripts

Normal clone:

- `scripts/cluster/sbatch_resume_img80_rgb_third_l40s.sh`
- `scripts/cluster/sbatch_resume_img80_rgb_wrist_l40s.sh`

RTX 6000 clone:

- `scripts/cluster/sbatch_resume_img80_depth_third_rtx6000.sh`
- `scripts/cluster/sbatch_resume_img80_rgbd_third_rtx6000.sh`

Each script:

- points to the current `student_latest.pt`
- uses a fresh `run_dir`
- uses a fresh `wandb_name`
- enables periodic viewer/video logging with
  `--capture_viewer_interval 1000`

## If You Want To Resume Later

Cancel the old job first, then submit the prepared replacement.

Normal clone:

```bash
ssh tylerlum@sc
cd /move/u/tylerlum/github_repos/depthbasedRL
git fetch origin 2026-04-20_dextrah_style_distillation
git checkout 2026-04-20_dextrah_style_distillation
git pull --ff-only

scancel 15209711
sbatch scripts/cluster/sbatch_resume_img80_rgb_third_l40s.sh

scancel 15210520
sbatch scripts/cluster/sbatch_resume_img80_rgb_wrist_l40s.sh
```

RTX 6000 clone:

```bash
ssh tylerlum@sc
cd /move/u/tylerlum/github_repos/depthbasedRL_rtx6000
git fetch origin 2026-04-20_dextrah_style_distillation
git checkout 2026-04-20_dextrah_style_distillation
git pull --ff-only

scancel 15209712
sbatch scripts/cluster/sbatch_resume_img80_depth_third_rtx6000.sh

scancel 15209713
sbatch scripts/cluster/sbatch_resume_img80_rgbd_third_rtx6000.sh
```

## Why New WandB Runs

This is intentional. It avoids:

- overwriting or confusing the existing run history
- mismatched step numbers in a single run
- ambiguity about where the interruption happened

The new runs will still load the saved weights and optimizer state.
