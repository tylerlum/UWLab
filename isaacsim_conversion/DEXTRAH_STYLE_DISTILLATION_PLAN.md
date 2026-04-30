# DEXTRAH-Style Distillation Notes

## What Changed

This branch adds a DEXTRAH-style online distillation mode:

```bash
--mode train_online
```

The loop is continuous rather than episode-batched:

1. Reset vectorized envs once at startup.
2. At each simulator step, compute privileged teacher action at the current student-visited state.
3. Compute student action from camera or teacher-observation input.
4. Optimize the student against teacher action and auxiliary object-position target.
5. Step the simulator with the student action only (`beta = 0`).
6. Reset only envs that satisfy done/reset conditions.

The older `--mode train` path remains available for fixed-decay and curriculum comparisons.

## Reset Conditions

IsaacSim distillation now tracks per-env reset state similar to `isaacgymenvs/tasks/simtoolreal/env.py`.

Enabled reset reasons:

- `object_z_low`: object env-local z is below `0.1`.
- `time_limit`: `progress_buf >= episode_length - 1`, default `600`.
- `max_goals`: env has reached all task goals.
- `hand_far`: max fingertip-to-object distance is above `1.5 m`.
- `dropped_after_lift`: optional, default `false` to match `SimToolReal.yaml`.

Reset reason counts are written to `metrics.csv` and W&B as:

- `reset_object_z_low`
- `reset_time_limit`
- `reset_max_goals`
- `reset_hand_far`
- `reset_dropped_after_lift`

Goal metrics use the max goal index reached since the last full vectorized reset, so successful envs are not scored as zero immediately after a per-env reset.

## Starter Commands

Teacher-observation sanity baseline:

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
  --mode train_online \
  --headless \
  --student_input teacher_obs \
  --student_arch mlp_recurrent \
  --distill_config isaacsim_conversion/configs/hammer_distill_teacher_obs_online_dagger_4096.yaml \
  --camera_config isaacsim_conversion/configs/hammer_camera_depth_160x90.yaml \
  --teacher_checkpoint pretrained_policy/model.pth \
  --teacher_config pretrained_policy/config.yaml
```

RGB 320x180 online DAgger:

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
  --mode train_online \
  --headless \
  --student_input camera \
  --student_arch mono_transformer_recurrent \
  --student_modality rgb \
  --distill_config isaacsim_conversion/configs/hammer_distill_rgb_320x180_online_dagger_512.yaml \
  --camera_config isaacsim_conversion/configs/hammer_camera_rgb_320x180.yaml \
  --teacher_checkpoint pretrained_policy/model.pth \
  --teacher_config pretrained_policy/config.yaml
```

RGB 160x90 online DAgger:

```bash
./scripts/run_in_isaacsim_env.sh python isaacsim_conversion/distill.py \
  --mode train_online \
  --headless \
  --student_input camera \
  --student_arch mono_transformer_recurrent \
  --student_modality rgb \
  --distill_config isaacsim_conversion/configs/hammer_distill_rgb_160x90_online_dagger_512.yaml \
  --camera_config isaacsim_conversion/configs/hammer_camera_rgb_160x90.yaml \
  --teacher_checkpoint pretrained_policy/model.pth \
  --teacher_config pretrained_policy/config.yaml
```

## Experiment Priority

Short-term:

1. Validate `teacher_obs` with `train_online`.
2. Reproduce the old `RGB 320x180` student result using `train_online`.
3. Compare `RGB 160x90` at the same env count to isolate input-detail effects.
4. Compare full-resolution render plus downsample only after the above baselines are stable.

Later:

1. Depth with DEXTRAH-style depth clipping and noise.
2. RGB-D at the best RGB resolution/env count.
3. Wrist camera only.
4. Third-person plus wrist camera fusion.
5. Multi-GPU scaling toward DEXTRAH's 4-GPU L40S setup.
