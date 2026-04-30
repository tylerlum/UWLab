# UWLab SimToolReal Rollouts

This branch adds a contained SimToolReal rollout path without modifying the OmniReset task path.

## Hammer Sanity Check

```bash
OMNI_KIT_ACCEPT_EULA=YES OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA=1 PYTHONNOUSERSITE=1 \
  env_uwlab/bin/python isaacsim_conversion/rollout.py \
  --task_source dextoolbench \
  --object_category hammer \
  --object_name claw_hammer \
  --task_name swing_down \
  --max_steps 120 \
  --video_dir outputs/simtoolreal_hammer_smoke \
  --headless \
  --enable_cameras
```

Expected local checkpoint/config paths:

- `.pretrained_checkpoints/SimToolReal/pretrained_policy/model.pth`
- `.pretrained_checkpoints/SimToolReal/pretrained_policy/config.yaml`

The checkpoint directory is intentionally ignored by Git.

## FurnitureBench Leg Prototype

```bash
OMNI_KIT_ACCEPT_EULA=YES OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA=1 PYTHONNOUSERSITE=1 \
  env_uwlab/bin/python isaacsim_conversion/rollout.py \
  --task_source furniturebench_leg \
  --max_steps 300 \
  --video_dir outputs/simtoolreal_furniturebench_leg \
  --headless \
  --enable_cameras
```

The leg prototype uses the OmniReset FurnitureBench SquareLeg and SquareTableTop USDs. If ignored local copies exist under `.pretrained_checkpoints/SimToolReal/omnireset_assets/FurnitureBench`, the rollout uses those; otherwise it falls back to the UWLab Hugging Face asset URLs.

Frame convention for the leg task:

- SimToolReal policy `+x` maps to SquareLeg USD `-z`.
- That points from the cuboidal handle toward the screw threads.
- The SquareLeg USD origin is already near the cuboidal handle center, so no extra positional origin offset is currently applied.
- Generated goals place the policy-frame origin over a selected tabletop hole, descend, then rotate clockwise about local `+x`.

The leg scene adds a simple gray support work surface under the SquareTableTop USD so the FurnitureBench part is visible in video. Current status: the leg path spawns and runs through the policy loop, but the policy drops the leg in the short smoke test. The next useful step is a reset that starts the leg in-hand or near a stable grasp before evaluating insertion/spin behavior.
