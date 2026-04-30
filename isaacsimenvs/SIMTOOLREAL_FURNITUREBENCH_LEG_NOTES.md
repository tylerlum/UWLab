# SimToolReal FurnitureBench Leg Notes

These notes capture the working assumptions and non-obvious implementation details for running the pretrained SimToolReal policy on the FurnitureBench square leg task in UWLab/Isaac Sim.

## Trusted Code Path

The preferred rollout entrypoint is:

```bash
isaacsimenvs/rollout_simtoolreal_policy.py
```

This uses `Isaacsimenvs-SimToolReal-Direct-v0` for observation construction, action scaling, joint target application, PD gains, materials, and scene setup. The older `isaacsim_conversion/rollout.py` path was useful for early experiments, but the current policy/physics matching work should be based on `isaacsimenvs`.

The SimToolReal policy interface is:

- 140 policy observations.
- 29 normalized actions: 7 Kuka iiwa arm joints plus 22 Sharpa hand joints.
- `deployment/rl_player.py` appends the SAPG scalar internally before calling `rl_games`.
- Action outputs are normalized. The env action pipeline converts canonical policy order to Isaac Lab joint order, applies optional action delay, integrates arm actions as velocity-like deltas into joint-position targets, maps hand actions as absolute joint-position targets, smooths both with moving-average filters, clamps to joint limits, and finally calls `set_joint_position_target`.

The default control rate is 60 Hz policy control over 120 Hz physics:

- `sim.dt = 1 / 120`
- `decimation = 2`
- policy step `dt = 1 / 60`

## Scene Setup

The FurnitureBench leg scenario in `rollout_simtoolreal_policy.py` configures:

- Robot: Kuka iiwa 14 plus Sharpa hand from `assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf`.
- Work surface: `assets/urdf/table_narrow.urdf`, converted to USD and made kinematic.
- Dynamic grasp object: FurnitureBench `SquareLeg/square_leg.usd`.
- Fixed mating fixture: FurnitureBench `SquareTableTop/square_table_top.usd`, spawned as a separate kinematic rigid object.
- Optional `GoalViz` object: collision disabled; visibility can be toggled with `--hide_goal_viz` or `--show_goal_viz`.

Keep the tabletop/mating part kinematic. This matches OmniReset and is the most useful real-world assumption: in real use the tabletop or fixture should be clamped/bolted and calibrated. Letting the tabletop move can hide bad insertion physics because the target frame moves with the object being pushed.

The robot is baked with gravity disabled. The dynamic leg has gravity enabled. The table, fixture, and goal viz are kinematic with gravity disabled.

## Decisions and Gotchas From Debugging

These are the practical clarifications that mattered while getting the task to run.

### Kinematic tabletop / receptive object

OmniReset makes the receptive object kinematic. In the UWLab OmniReset RL-state config:

- `insertive_object`: `kinematic_enabled=False`
- `receptive_object`: `kinematic_enabled=True`
- physical support table: `kinematic_enabled=True`

This means the square leg is the only free object in the assembly interaction. The FurnitureBench tabletop/mating part is fixed in the simulation.

We should do the same for SimToolReal leg rollouts:

- Keep `SquareLeg` dynamic.
- Keep `SquareTableTop` kinematic.
- Keep `table_narrow` kinematic.

This is easier than baking the tabletop into the table URDF, easier to swap/debug, closer to OmniReset, and closer to a real deployment where the fixture/tabletop should be clamped or bolted. A free tabletop can hide failures because the policy may push the receptive object instead of inserting the leg.

### The 1 g mass confusion

OmniReset's variant factory sets the insertive-object spawn mass to `0.001 kg`, which looks suspicious for the FurnitureBench leg. That is not the full story for RL-state training/eval configs: the startup randomization event overwrites the insertive object mass with an absolute sample in `[0.02, 0.2] kg`.

For deterministic SimToolReal leg rollouts, use an explicit fixed mass in that range. Current choice:

```text
SquareLeg mass = 0.05 kg
```

This avoids the misleading 1 g default while staying representative of OmniReset's actual runtime mass range.

### Policy rate and action path

The pretrained SimToolReal policy should still run at 60 Hz:

```text
physics dt = 1 / 120
decimation = 2
policy dt = 1 / 60
```

The important action details are:

- The policy outputs 29 normalized actions in canonical SimToolReal order.
- `action_utils.py` permutes canonical order to Isaac Lab joint order.
- Arm actions are integrated as velocity-like deltas into joint-position targets.
- Hand actions are absolute normalized-to-joint-limit position targets.
- Both arm and hand targets are smoothed with moving average `0.1`.
- Targets are clamped to joint limits before being sent to Isaac Lab.

When performance is bad, check this path before changing physics. Observation/action order, quaternion convention, and target rescaling are more likely to silently degrade behavior than a small friction tweak.

### Quaternion conventions

Use the conventions explicitly:

- SciPy/deployment helper math uses `xyzw`.
- Isaac Lab root states and config poses use `wxyz`.
- The rollout uses helper functions `xyzw_to_wxyz()` and `wxyz_to_xyzw()` at the boundaries.

For the leg policy frame, the most important transform is:

```text
q_world_policy = q_world_asset * q_asset_policy
q_world_asset = q_world_policy * inverse(q_asset_policy)
```

Mixing the order or the `xyzw`/`wxyz` convention makes the leg appear roughly plausible but point the screw axis incorrectly.

### Goal visualization

The goal object is useful for debugging but can block the view of the hole/thread interaction. Use:

```bash
--hide_goal_viz
```

This only changes USD visibility for `GoalViz`. It should not change observations, goal poses, or physics because goal-viz collision is disabled.

### Viewer/video timing

Short videos were misleading because a few hundred policy steps only showed a few seconds. For visual checks, prefer about 10-20 seconds:

```text
600 policy steps = 10 seconds at 60 Hz
1200 policy steps = 20 seconds at 60 Hz
```

For non-headless viewing, make sure the rollout is not terminating early on success or time. The leg debugging path added options such as `--ignore_dones` and longer `--max_steps` so the viewer keeps updating long enough to inspect contact.

### Teleport sanity mode

The policy is not expected to solve the new leg task perfectly. To isolate physics from robot control, use the fixed-robot scripted-object mode:

```bash
--robot_control_mode hold_default \
--object_drive_mode teleport_trajectory
```

Useful variants:

- Periodic teleport every `--teleport_interval_s 1.0`: good for checking whether the object blasts away after contact.
- `--teleport_write_every_step`: good for dense screw/path visualization, but less strict as a physics stress test because the state is overwritten every step.
- Hide goal viz while doing this if it blocks the hole.

The harder future sanity check would be to drive the object with forces/torques instead of teleporting, but teleport mode is the simplest first-pass contact diagnostic.

### OmniReset spin count observation

The state-based OmniReset leg expert reached success quickly in the rerun. With the analysis thresholds used there:

- first hole-entry estimate: about `4.3 s`
- first success: about `6.2 s`
- entry-to-success rotation: about `0.45` clockwise turns

Treat this as an empirical observation, not a hard task definition. OmniReset success ignores yaw, so the exact number of spins is not part of the reward/success condition.

## Leg Frame Convention

SimToolReal expects the object frame `+x` axis to be the long grasped-object axis. For the FurnitureBench square leg, `+x` should point from the cuboidal handle toward the screw threads.

The SquareLeg USD root frame does not already match that policy frame. The rollout uses:

```python
R_USD_POLICY_LEG = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
```

The columns are policy-frame axes expressed in the SquareLeg USD root frame:

- policy `+x` = USD `-z` = handle toward screw threads.
- policy `+y` = USD `+x`.
- policy `+z` = USD `-y`.

Observation conversion uses:

```text
q_world_policy = q_world_asset * q_asset_policy
policy_pos = asset_pos + rotate(q_world_asset, object_policy_frame_pos_offset)
```

For the current square leg, `object_policy_frame_pos_offset = (0, 0, 0)` because the USD origin is already near the cuboidal handle center. If a later asset origin is at the full object centroid instead, shift this offset toward `-x` in policy frame so the policy origin remains near the handle/grasp area.

When writing scripted policy-frame goals back to the USD asset, use the inverse rotation:

```text
q_world_asset = q_world_policy * inverse(q_asset_policy)
```

This is implemented by `_policy_to_leg_asset_pose_xyzw()`.

## Goal Placement

The rollout currently uses the first FurnitureBench tabletop hole by default:

```python
FURNITUREBENCH_TABLE_HOLES = [
    [ 0.0562,  0.0562, 0.0020],
    [-0.0562,  0.0562, 0.0020],
    [-0.0562, -0.0563, 0.0020],
    [ 0.0562, -0.0563, 0.0020],
]
```

The fixed fixture pose is placed on top of `table_narrow` using the SquareTableTop bottom metadata:

```text
fixture_root_z = table_center_z + table_half_height - fixture_local_bottom + clearance
```

with current values:

- `table_center_z = 0.38`
- `table_half_height = 0.15`
- `fixture_local_bottom = -0.01558619`
- default clearance = `0.002`

The policy-frame goal orientation points the leg long axis down, so the threaded end points toward the hole. Clockwise screwing is represented as negative rotation about local policy `+x`.

The scripted leg trajectory has two useful modes:

- `--leg_spin_mode after_insert`: descend first, then spin at fixed depth.
- `--leg_spin_mode helical`: descend and spin together.

The teleport sanity mode can hold the robot fixed while moving the object through the scripted trajectory:

```bash
--robot_control_mode hold_default \
--object_drive_mode teleport_trajectory \
--teleport_interval_s 1.0
```

Use `--teleport_write_every_step` for dense pose writes. Without it, poses are written only at the configured waypoint interval, which is better for checking whether contacts produce obvious jumps or blasts after each placement.

## OmniReset Goal and Success Semantics

OmniReset uses metadata-defined assembled frames rather than raw object root frames.

FurnitureBench square leg metadata:

```yaml
assembled_offset:
  pos: [0.0, 0.0, -0.056658]
  quat: [1.0, 0.0, 0.0, 0.0]
bottom_offset:
  pos: [0.0, 0.0, -0.056658]
  quat: [1.0, 0.0, 0.0, 0.0]
```

FurnitureBench square tabletop metadata:

```yaml
assembled_offset:
  pos: [0.05625, 0.05625, -0.009435]
  quat: [1.0, 0.0, 0.0, 0.0]
bottom_offset:
  pos: [0.0, 0.0, -0.015586]
  quat: [1.0, 0.0, 0.0, 0.0]
success_thresholds:
  position: 0.0025
  orientation: 0.025
```

OmniReset success computes:

```text
T_table_hole_to_leg_assembled =
    inverse(T_world_table_assembled) * T_world_leg_assembled
```

Then:

- `xyz_distance = norm(relative_xyz)`
- `euler_xy_distance = abs(wrap(roll)) + abs(wrap(pitch))`
- position success: `xyz_distance < 0.0025`
- orientation success: `euler_xy_distance < 0.025`
- task success: `position_success and orientation_success`
- yaw is intentionally ignored, i.e. `yaw` from the relative quaternion is discarded.

This means the reward/success checks that the leg is deep enough and upright enough, but it does not enforce a unique screw yaw phase at a given depth. The simulator contact model must provide the screw behavior. If a policy pushes straight down and the leg penetrates without spinning, OmniReset-style success alone would not catch that unless the penetration produces wrong position/tilt or instability.

### Exact OmniReset success condition

The precise boolean used by `ProgressContext` is:

```python
position_aligned = xyz_distance < success_position_threshold
orientation_aligned = euler_xy_distance < success_orientation_threshold
success = orientation_aligned & position_aligned
```

For `fbtabletop`, the thresholds come from metadata:

```text
success_position_threshold = 0.0025 m
success_orientation_threshold = 0.025 rad
```

Important interpretation:

- The 3D norm includes depth. The leg must be at the right final depth, not just centered over the hole.
- The orientation condition only checks that the insertion axis is upright/aligned. It does not care about spin phase around that axis.
- This is why yaw can be non-unique while success is still well-defined: success is an assembled-frame position plus roll/pitch condition, not a full 6-DoF pose match.
- The code keeps a `continuous_success_counter`, but in the RL-state config success itself is not a termination. The listed terminations are timeout and abnormal robot state. Evaluation/play scripts may stop or log when they see success, but the task config does not terminate immediately just because success is true.

### OmniReset dense and sparse reward terms

The relevant state-task rewards are:

```text
0.1 * progress_context
0.1 * ee_asset_distance_tanh
0.1 * dense_success_reward
1.0 * success_reward
-1e-4 * action_l2_clamped
-1e-3 * action_rate_l2_clamped
-1e-2 * joint_vel_l2_clamped
-100.0 * abnormal_robot_state
```

`progress_context` mostly computes and stores the success metrics; its call returns zeros, so the `0.1` weight does not add a direct dense reward.

`ee_asset_distance_tanh` is a reach/grasp-style shaping term:

```text
1 - tanh(distance(end_effector_gripper_offset, insertive_object) / std)
std = 1.0
```

`dense_success_reward` is:

```text
angle_term = exp(-euler_xy_distance / std)
pos_term = exp(-xyz_distance / std)
dense_success_reward = 0.5 * (angle_term + pos_term)
std = 1.0
```

`success_reward` is the sparse binary reward:

```text
1.0 if orientation_aligned and position_aligned else 0.0
```

### Recommended SimToolReal leg success/reward

For our own reward, keep the OmniReset-style terminal metric for comparability:

```text
assembled-frame position error < 2.5 mm
assembled-frame roll/pitch error < 0.025 rad
yaw ignored for terminal success
require 3-5 consecutive policy steps for robust evaluation
```

Then add separate diagnostics or shaping for thread sanity:

- Track downward progress after the leg is radially near the hole.
- Track clockwise yaw progress about local policy `+x`.
- Penalize depth progress without enough yaw progress if we specifically want to discourage thread penetration shortcuts.
- Log straight-down penetration failures separately from normal policy failures. A rollout that reaches the final z without spinning is a physics/contact problem even if the OmniReset-style success metric says it is aligned.

## Collision and SDF Status

The FurnitureBench USDs already author collision approximations. Do not casually replace these with generated convex decompositions if the goal is to test screw-thread contact.

Local inspection of the current SquareLeg USD shows:

- `/square_leg/collisions/leg`: `physics:approximation = convexHull`
- `/square_leg/collisions/bolt`: `physics:approximation = sdf`

Local inspection of the current SquareTableTop USD shows:

- hole collision meshes: `physics:approximation = sdf`
- tabletop and wall collision meshes: `physics:approximation = convexHull`

This is the important SDF piece: the threaded bolt and tabletop holes are SDF mesh collisions. The SimToolReal rollout does not regenerate those meshes. It copies/bakes the USD and authors additional rigid-body/contact attributes while preserving the source collision approximations.

The bake step sets:

- collision contact offset = `0.002`
- collision rest offset = `0.0`
- object `max_depenetration_velocity = 1000.0`
- table and fixture kinematic/gravity-disabled flags
- goal viz collision disabled

## Physics Profiles

The default SimToolReal physics profile is inherited from `SimToolRealEnvCfg`:

```text
solver_type = 1  # TGS
min/max position iterations = 8 / 8
min/max velocity iterations = 0 / 0
bounce_threshold_velocity = 0.2
friction_offset_threshold = 0.04
friction_correlation_distance = 0.025
```

For the leg screw task, `--leg_physics_profile omnireset` applies settings closer to the UWLab OmniReset leg task:

```text
solver_type = 1  # TGS
max_position_iteration_count = 192
max_velocity_iteration_count = 1
bounce_threshold_velocity = 0.02
friction_offset_threshold = 0.01
friction_correlation_distance = 0.0005
gpu_found_lost_aggregate_pairs_capacity = 4 * 1024 * 1024
gpu_total_aggregate_pairs_capacity = 2**23
gpu_max_rigid_contact_count = 2**23
gpu_max_rigid_patch_count = 2**23
gpu_collision_stack_size = 2**31
```

Rigid-body per-asset settings for that profile:

```text
leg object mass = 0.05 kg
fixture mass = 0.5 kg  # kinematic, so mass is not dynamically important
object solver position iterations = 4
object solver velocity iterations = 0
fixture solver position iterations = 4
fixture solver velocity iterations = 0
```

Friction representative values from OmniReset startup randomization:

```text
insertive leg static/dynamic = 1.5 / 1.4
receptive fixture static/dynamic = 0.4 / 0.325
table static/dynamic = 0.45 / 0.35
```

The 0.05 kg leg mass is intentional. OmniReset's config spawns variant insertive objects at 0.001 kg, but the RL-state task also has a startup mass randomization event that overwrites the insertive object mass with an absolute 0.02-0.2 kg sample. The deterministic SimToolReal leg profile uses 0.05 kg as an in-range representative value.

## Robot Gains and Gravity

Robot PD gains are owned by `isaacsimenvs/tasks/simtoolreal/utils/scene_utils.py`.

Important high-level points:

- Arm stiffness: `[600, 600, 500, 400, 200, 200, 200]` for iiwa joints 1-7.
- Arm damping: roughly `[27.0, 27.0, 24.7, 22.1, 9.75, 9.15, 9.15]`.
- Hand stiffness/damping/armature are also configured there for the 22 Sharpa joints.
- The robot USD is baked with `disable_gravity=True`. This is equivalent to relying on a gravity-compensated arm model for this rollout.
- The dynamic leg still has gravity enabled.

If policy performance looks wrong, first verify observation/action ordering and the action rescaling path before changing gains.

## Practical Checks

Useful checks when modifying the leg task:

- Re-run the claw hammer scenario first. It is the regression that validates the SimToolReal observation and action path.
- Keep `isaacsim_conversion` untouched as a historical reference, but avoid using it as the source of truth for matching policy behavior.
- If the goal object blocks visual inspection, use `--hide_goal_viz`; this should only change USD visibility, not observations or physics.
- For contact sanity, use fixed robot plus teleport trajectory before trusting closed-loop policy behavior.
- For screw sanity, watch whether the threaded end jams/spins rather than translating straight through SDF hole geometry.
- If the leg explodes or tunnels, try the OmniReset physics profile before changing object frames or reward code.
