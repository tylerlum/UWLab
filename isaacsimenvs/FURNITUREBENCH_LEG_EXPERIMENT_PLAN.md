# FurnitureBench Leg SimToolReal Experiment Plan

Last updated: 2026-05-07

This file tracks what we currently believe about finetuning the pretrained
SimToolReal policy on the FurnitureBench square-leg screw task in UWLab.  It is
intended to complement `isaacsimenvs/SIMTOOLREAL_FURNITUREBENCH_LEG_NOTES.md`,
which has the lower-level frame, asset, SDF, and physics details.

## Operational Guardrails

- Do not start, stop, cancel, or restart cluster jobs without explicit approval.
- It is fine to inspect W&B, Slurm logs, TensorBoard event files, local code, and
  local output artifacts without approval.
- Before proposing a new cluster batch, write the exact run matrix, W&B group,
  GPU target assumptions, and what question each run answers.
- Treat early W&B metrics as directional only. For this task, always inspect the
  pose viewer or rollout video before calling a run successful.

## Source of Truth Files

Core task and MDP:

- `isaacsimenvs/tasks/furniturebench_leg/furniturebench_leg_env.py`
- `isaacsimenvs/tasks/furniturebench_leg/furniturebench_leg_env_cfg.py`
- `isaacsimenvs/tasks/simtoolreal/utils/obs_utils.py`
- `isaacsimenvs/tasks/simtoolreal/utils/action_utils.py`
- `isaacsimenvs/tasks/simtoolreal/utils/reward_utils.py`
- `isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py`

Training and cluster launch:

- `isaacsimenvs/train.py`
- `isaacsimenvs/cfg/train/SimToolRealSAPG.yaml`
- `isaacsimenvs/test_experiments/furniturebench_leg/furniturebench_leg_sapg_finetune.sub`
- `isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_robustness_ablation.sh`
- `isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_screw_diagnostics.sh`
- `isaacsimenvs/test_experiments/furniturebench_leg/launch_furniturebench_leg_high_hover_controls_matrix.sh`

Evaluation and visualization:

- `isaacsimenvs/eval_furniturebench_leg_policy.py`
- `isaacsimenvs/rollout_simtoolreal_policy.py`
- `isaacsimenvs/tasks/simtoolreal/pose_viewer.py`
- `scripts/tools/view_furniturebench_leg_alignment.py`
- `scripts/tools/view_omnireset_partial_assemblies.py`

## Current MDP Shape

The leg task inherits the trusted SimToolReal Isaac Sim MDP and replaces the
asset/goal side with the FurnitureBench leg/fixture:

- Robot: Kuka iiwa + Sharpa hand.
- Dynamic object: OmniReset/FurnitureBench `SquareLeg` USD.
- Kinematic fixture: OmniReset/FurnitureBench `SquareTableTop` USD.
- Physics profile: OmniReset-like SDF/contact settings for the threaded bolt
  and receiving hole.
- Policy observations: 140-dim SimToolReal observations.
- Policy actions: 29 normalized actions, then SimToolReal action delay,
  canonical-to-Lab joint mapping, arm delta integration, hand absolute target
  mapping, smoothing, clamp, and Isaac Lab joint-position targets.
- Control: 60 Hz policy over 120 Hz physics.

The object and goal observations use the SimToolReal policy frame convention:
policy `+x` points from the leg handle toward the screw threads.  The USD asset
frame is not rewritten during training; the conversion happens at observation
and scripted-goal boundaries.

## Goal Modes

All goal modes end at the same final inserted pose unless otherwise noted:
the OmniReset assembled pose for the chosen tabletop hole.

- `finalGoalOnly`: only the final inserted pose.
- `preInsertAndFinal`: manual pre-insert pose, then final inserted pose.
- `preInsertDenseFinal`: manual pre-insert pose, then dense helical screw
  interpolation down to the final pose. No high hover goal.
- `dense`: high hover, manual pre-insert pose, then dense helical screw
  interpolation down to the final pose.
- `highHover`: only the high hover pose.
- `highHoverAndFinal`: high hover, then final inserted pose.

Current important geometric values:

- Manual pre-insert pose: final pose plus `0.015427 m` in local/world z and
  `+20.73494 deg` yaw.
- Final yaw offset: `0 deg`.
- Dense screw interpolation: pre-insert yaw to
  `final_yaw - pre_yaw - 2*pi*dense_screw_turns`.
- Current default dense screw turns: `1.0`.
- Current high hover default: `0.4 m`, which is probably too high for the next
  serious set of tests.

Interpretation:

- `preInsertDenseFinal` isolates whether the helical insertion segment is
  enough, without the possibly over-scaffolded high hover.
- `dense` includes the approach scaffold and is easier, but may teach behavior
  that is more staged than necessary.
- Sparse modes answer whether the pretrained SimToolReal policy can infer
  screw motion from final/pre-insert goals without dense yaw/depth waypoints.

## Success, Rewards, and Screw Metrics

There are two success modes:

- `simtoolreal_keypoints`: keypoint-distance success in the SimToolReal style.
- `omnireset_alignment`: OmniReset assembled-frame position plus roll/pitch
  alignment, yaw ignored.

The current training batches mostly use `simtoolreal_keypoints` with:

- `SUCCESS_TOLERANCE=0.01`
- `TARGET_SUCCESS_TOLERANCE=0.01`
- `SUCCESS_STEPS=10`
- `FORCE_CONSECUTIVE_NEAR_GOAL=true`
- `FORCE_LIFTED_FOR_KEYPOINT_REWARD=false`

`FORCE_LIFTED_FOR_KEYPOINT_REWARD=false` matters because otherwise the agent can
be denied useful approach reward when it is visually close to the goal but has
not satisfied the inherited lifted-object condition.

The optional final screw gate is:

```text
final_success_requires_screw_insert_like = true
```

This does not directly reward spinning every step. It changes the final subgoal
success condition so final success requires:

- the leg has entered the hole;
- maximum clockwise turns after entry is at least
  `screw_metric_min_turns_for_insert`;
- maximum depth after entry reaches the required insertion depth;
- no pushthrough flag was triggered.

Current default:

```text
screw_metric_min_turns_for_insert = 0.5
screw_metric_hole_radius = 0.02 m
```

The policy learns to spin primarily because the dense helical subgoals move both
z and yaw between the manual pre-insert pose and final pose. The screw gate
prevents straight-down final-goal shortcuts from being counted as success, but
the dense waypoints are the main source of intermediate spin supervision.

Important W&B metrics:

- `episode_final/screw_insert_like`: best single sort metric for "inserted by
  rotating enough and not pushthrough" at episode end.
- `screw_insert_like_ratio`: live per-step ratio; useful but more volatile.
- `episode_final/all_goals_hit`: whether the episode finished the configured
  goal sequence.
- `all_goals_hit_ratio`: live all-goals metric for previous completed episodes.
- `episode_final/screw_cw_turns_after_entry`: how much clockwise rotation
  happened after entering the hole.
- `episode_final/screw_depth_after_entry_m`: how far the leg progressed in
  depth after entry.
- `episode_final/screw_pushthrough`: should stay low; high values mean the
  policy/contact setup is exploiting or breaking the threaded geometry.
- `omnireset_pos_align_error` and `omnireset_xy_rot_align_error`: useful when
  comparing keypoint success vs OmniReset-style assembled-frame success.

Do not use raw `global_step` to reason about physical progress. It is now
epoch-scale for W&B/TensorBoard sanity; real simulator frames are logged as
`info/frames`.

## Current Domain Randomization State

Inherited SimToolReal defaults still include:

- policy observation delay: enabled, max 3 policy steps;
- action delay: enabled, max 3 policy steps;
- object-state delay/noise: enabled, max 10 policy steps;
- object observed xyz noise: `0.01 m`;
- object observed rotation noise: `5 deg`;
- joint velocity observation noise: `0.1`;
- object scale noise multiplier knob exists but is currently launched as
  `[1.0, 1.0]`.

Current FurnitureBench cluster launches intentionally disable random external
wrenches:

```text
FORCE_SCALE=0.0
TORQUE_SCALE=0.0
```

Reason: first establish that the task can learn correct threaded insertion with
the fixed SDF fixture. Add force/torque disturbances only after the basic
success and screw metrics look reliable.

Physics-param randomization is not currently ported for this v1 task:

- no object mass randomization during training;
- no friction randomization during training;
- no DOF gain/friction/armature randomization during training.

This is intentional for now. Mass/friction randomization is a later robustness
axis after the core MDP is validated.

## What Looks Validated So Far

These are working assumptions based on the videos, local rollouts, and W&B runs
seen so far. They should be revised as more full-episode viewers come in.

- The claw-hammer SimToolReal regression works and remains the first sanity
  check for the observation/action path.
- Green goal visualization is working after increasing visibility; earlier
  invisible-goal videos were a viewer/material opacity issue, not an MDP issue.
- The W&B/Three.js interactive pose viewer now records useful full-episode
  captures. The old 0/1-frame reset-boundary bug is fixed.
- TensorBoard/W&B scalar logging is now throttled and uses epoch-scale
  `global_step`; older groups with frame-scale steps are polluted and should not
  be used for loading-performance conclusions.
- The OmniReset USD assets with mixed SDF/convex collisions are the right sim
  assets for training. FurnitureBench URDF/OBJ paths are only used for
  lightweight Three.js visualization.
- The fixed kinematic fixture is the right default. A free fixture would hide
  insertion failures by letting the target move.
- The 0.05 kg leg mass is more reasonable than the misleading 1 g variant
  factory default and is within OmniReset's runtime mass randomization range.
- Teleport tests and OmniReset-policy rollouts suggest the threaded SDF contact
  is stable enough to train against when using the OmniReset-like physics
  profile.
- Dense helical goals plus the final screw gate are the strongest current
  explanation for the first runs that showed clear clockwise spinning behavior.

## What Looks Weak or Not Yet Trusted

- The pretrained SimToolReal policy alone is not expected to solve the leg task
  zero-shot. It can provide a good approach prior, but not reliable screwing.
- `finalGoalOnly` is not yet trusted. If it succeeds, inspect videos carefully
  for straight pushthrough or tolerance artifacts.
- `preInsertAndFinal` is not yet trusted as sufficient. Sparse pre/final goals
  may not teach the continuous yaw-depth coupling needed for screw threads.
- Very strict final criteria such as 5 mm keypoint tolerance or OmniReset's
  2.5 mm / 0.025 rad thresholds looked too hard in early experiments.
- A 0.4 m high hover is probably too high. It may help solve the task, but it
  likely over-scaffolds the trajectory relative to the real intended behavior.
- High `all_goals_hit` alone is not enough. A good run must also show
  `screw_insert_like`, clockwise turns after entry, depth after entry, low
  pushthrough, and believable viewer/video contact.
- If fixture randomization is enabled, ensure the goal, fixture, and visual
  frames all move together. The fixture should remain fixed within an episode.
- One known thing to watch: inherited `table_reset_z_range` can move the table
  height. If future fixture/goal height behavior looks inconsistent, either
  set that range to zero for this task or make the fixture/goal height follow
  the sampled table z exactly.

## Main Experimental Axes

### 1. Object Initial Pose

Reasonable settings:

- `upright_fixed`: easiest positive-control setup. Use this for diagnosing
  rewards, goals, and contact.
- `random_table` low: small xy/yaw/roll-pitch variation around a nearby start.
- `random_table` moderate: e.g. xy range `0.04 m`, z range `0.02 m`,
  yaw `45 deg`, roll/pitch `20 deg`.
- `random_table` strong: e.g. xy range `0.08 m`, z range `0.03 m`,
  yaw `180 deg`, roll/pitch `60 deg`.
- `omnireset_partial_assemblies`: close-to-insert starts sampled from
  OmniReset partial assemblies. Useful for curriculum and for studying whether
  the policy can finish from already-aligned threaded states.

Recommended progression:

1. Validate spinning from `upright_fixed`.
2. Add moderate object randomization.
3. Add strong object randomization only after moderate works.
4. Use partial assemblies for a separate "finish the screw" curriculum/control.

### 2. Fixture Pose

Reasonable settings:

- fixed default fixture: primary training setup.
- fixture moved away, e.g. `FIXTURE_XY_OFFSET=[-0.25,0.0]`: positive control
  for reward/goal reaching without threaded contact.
- fixture XY randomization low/moderate, e.g. `[0.01,0.01]` to `[0.03,0.03]`.
- fixture plus object randomization: hardest realistic setup.

Do not randomize fixture pose until the fixed-fixture task is reliable. For real
deployment, the tabletop/fixture should be calibrated and fixed; fixture
randomization is for robustness, not the base task definition.

### 3. Goal Schedule

Reasonable goal schedules:

- final only: minimal, but likely too sparse.
- pre-insert plus final: sparse two-stage insertion.
- pre-insert plus dense final: most important current mode. It isolates the
  helical screw phase without the high hover.
- high hover plus pre-insert plus dense final: easiest scaffolded dense mode.
- low hover plus dense final: same scaffold, but with more realistic approach
  heights.

Reasonable hover heights to test:

- no hover;
- `0.05 m`;
- `0.10 m`;
- `0.20 m`;
- avoid making `0.40 m` the default unless it clearly improves learning without
  making behavior unnatural.

Reasonable dense screw waypoint counts:

- 1 intermediate/step: tests whether almost-sparse yaw-depth hint is enough.
- 2 or 3: low waypoint count, still teaches yaw-depth coupling.
- 5: medium.
- 10: current strong scaffold.

The highest-priority sparse-goal question:

```text
Does preInsertDenseFinal still work with 1, 2, 3, or 5 screw steps?
```

If it works with 1-3 steps, the behavior is not overly dependent on dense
waypoint shaping. If it only works with 10, we should treat it as a curriculum
that needs to be annealed toward sparser goals later.

### 4. Success and Reward Criteria

Reasonable settings:

- keypoint success, 1 cm tolerance: current practical default.
- keypoint success, 5 mm tolerance: stricter but may be too hard initially.
- OmniReset alignment, relaxed: position `1 cm`, orientation `0.05 rad`.
- OmniReset alignment, strict: metadata position `2.5 mm`, orientation
  `0.025 rad`.
- final screw gate off: tells whether dense goals alone are enough.
- final screw gate on: prevents straight pushthrough from being counted as
  success.

Open question:

```text
Should final success be SimToolReal keypoints, OmniReset alignment, or both?
```

Current recommendation:

- Train initially with keypoint success at 1 cm plus the final screw gate.
- Log OmniReset alignment in parallel.
- Once spinning works robustly, compare a stricter OmniReset-alignment variant
  as an evaluation metric and possibly as a late curriculum target.

### 5. Screw Supervision

Reasonable settings:

- no screw gate: ablation.
- screw gate only at final goal: current practical safeguard.
- dense helical goals: current strongest spin teacher.
- explicit intermediate screw reward: not currently implemented; consider only
  if dense goals are insufficient or too brittle.
- curriculum: start with 10 dense screw steps, then reduce to 5, 3, 2, 1.

Current interpretation:

- The screw gate is a validity filter, not a dense spin reward.
- Dense helical goals are what tell the policy how to spin.
- If someone asks why spinning started working, the short answer is:
  "we stopped accepting straight-down final-pose shortcuts and added a
  yaw-depth helical goal curriculum between the manually aligned pre-insert
  pose and the OmniReset final assembled pose."

### 6. Physics and Contact

Reasonable settings:

- OmniReset physics profile: default for leg training.
- fixed `0.05 kg` leg: current default.
- no force/torque randomization: current default.
- force/torque randomization: later robustness after base success.
- mass/friction randomization: later robustness, not v1 default.

Do not switch threaded collisions to convex decomposition for training unless
the goal is a negative control. The useful behavior comes from the SDF threaded
bolt and SDF receiving-hole collision meshes.

### 7. Logging and Evaluation

Use W&B sort:

```text
episode_final/screw_insert_like
```

Then inspect:

```text
episode_final/all_goals_hit
episode_final/screw_cw_turns_after_entry
episode_final/screw_depth_after_entry_m
episode_final/screw_pushthrough
all_goals_hit_ratio
screw_insert_like_ratio
```

Viewer expectations:

- `CAPTURE_VIEWER_FULL_EPISODES=true`
- `CAPTURE_VIEWER_EPISODES=1` or `2`
- `CAPTURE_VIEWER_LEN=7200` for long captures
- `CAPTURE_VIEWER_INTERVAL=12000` or larger to avoid excessive media logging

TensorBoard/W&B summaries should remain:

```text
SUMMARIES_STEP_MODE=epoch
SUMMARIES_INTERVAL_ENABLED=true
SUMMARIES_DEFER_SEC=30
SUMMARIES_INTERVAL_SEC_MIN=60
SUMMARIES_INTERVAL_SEC_MAX=600
```

This keeps W&B usable and prevents frame-scale global steps from reaching 1B+.

## Immediate Next Questions

These are the next questions to answer before scaling to more diverse
randomization. They should be run only after explicit approval.

1. Does `preInsertDenseFinal` work without high hover?
   - Compare dense screw steps `10`, `5`, `3`, `2`, `1`.
   - Keep object and fixture fixed.
   - Keep screw gate on.

2. Is the screw gate necessary once dense helical goals exist?
   - Same `preInsertDenseFinal` setup.
   - Compare final screw gate on vs off.
   - Use videos and `episode_final/screw_insert_like`, not just reward.

3. How low can the hover be if we include a hover?
   - Compare no hover, `0.05 m`, `0.10 m`, `0.20 m`.
   - Do not make `0.40 m` the default unless the lower hovers fail badly.

4. Does the trained policy still spin under moderate object-start randomization?
   - Start from the best fixed-upright setting.
   - Add moderate `random_table`.
   - Check that the policy still rotates after hole entry rather than pushing
     through.

5. Does fixture randomization work after the fixed task works?
   - Start with fixture XY range `[0.01,0.01]` or `[0.03,0.03]`.
   - Keep object init easy at first.
   - Add object randomization only after fixture-only randomization works.

6. Does OmniReset alignment success agree with keypoint success?
   - Evaluate the same policies under both metrics.
   - Do not switch training success to strict OmniReset metadata thresholds until
     the relaxed/keypoint version reliably spins and inserts.

## Future Curriculum Ideas

- Start with `preInsertDenseFinal` and 10 screw steps, then reduce the number of
  screw steps over training.
- Start with 1 cm keypoint success and screw gate, then tighten keypoint or add
  OmniReset-alignment evaluation.
- Start from upright fixed object, then moderate object randomization, then
  strong object randomization.
- Use OmniReset partial assemblies as a "finish the screw" phase, especially if
  full pick-up plus approach makes the task too broad.
- Add force/torque disturbances only after base insertion works.
- Add mass/friction randomization after the contact behavior is trusted.
- Eventually add a retract/move-away phase, analogous to SimToolReal peg-in-hole,
  but only after insertion itself is not ambiguous.

## Current Batch Interpretation

The current robustness-ablation style batch is designed to answer:

- whether the final screw gate is necessary;
- whether high hover is necessary;
- how many dense screw waypoints are needed;
- whether moderate/strong object randomization breaks the behavior;
- whether fixture XY randomization breaks the behavior;
- whether lower hover is enough.

Do not over-interpret these runs until they have enough training time and at
least one full-episode viewer/video per promising condition. A run with high
reward but low clockwise-turn depth behavior is not a success for this task.

## Proposed Approval Template for Next Cluster Batch

Before launching a new batch, write something like:

```text
W&B group:
  2026-05-XX_leg_<purpose>

Jobs:
  1. <name>: <gpu>, <goal mode>, <init mode>, <success/gate>, <question>
  2. ...

Keep fixed:
  object mass, physics profile, fixture pose, obs/action delays, force/torque off

Primary metrics:
  episode_final/screw_insert_like
  episode_final/screw_cw_turns_after_entry
  episode_final/screw_depth_after_entry_m
  episode_final/screw_pushthrough
  episode_final/all_goals_hit

Approval needed before:
  sbatch/scancel/restart
```

