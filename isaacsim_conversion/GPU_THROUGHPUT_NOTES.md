# GPU Throughput Notes

Rough comparison from April 19, 2026 for Isaac Sim distillation rollouts.

Benchmark workload:
- `distill.py --mode teacher_eval`
- `256` envs
- `200` steps
- tiled RGB camera enabled at `320x180`
- teacher policy rollout

| GPU | Wall time | Relative to local RTX 4090 | Approx env-steps/s |
| --- | ---: | ---: | ---: |
| RTX 4090 local | `36.57s` | `1.00x` | `1400` |
| RTX PRO 6000 Blackwell | `50.61s` | `1.38x slower` | `1012` |
| L40S | `63.55s` | `1.74x slower` | `805` |
| RTX A5000 | `79.85s` | `2.18x slower` | `641` |

Capacity lower bounds observed with tiled RGB cameras:
- RTX 4090 local: `256` envs at `320x180`.
- L40S: `768` envs at `320x180`.
- RTX PRO 6000 Blackwell: at least `512` envs in short training smoke; larger image-training env counts still need preflight.

Notes:
- The first A5000 benchmark was contaminated by an Omniverse key-value DB lock and took `250.93s`; the clean rerun was `79.85s`.
- RTX PRO 6000 Blackwell needs the cu128 Isaac Sim environment because of `sm_120` support.
