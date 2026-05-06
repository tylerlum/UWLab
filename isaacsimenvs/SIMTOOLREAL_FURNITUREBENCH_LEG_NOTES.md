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

### Interactive alignment viewer

Use this standalone Viser tool to inspect the raw FurnitureBench asset frames and thread alignment without starting Isaac Sim:

```bash
.venv-viser/bin/python scripts/tools/view_furniturebench_leg_alignment.py --port 8082
```

The Viser environment is intentionally separate from `env_uwlab` so it does not disturb Isaac Sim package pins. If it needs to be recreated, install `viser`, `trimesh`, `scipy`, `numpy`, and `usd-core` into a local viewer venv.

The viewer shows:

- `/table`: raw `SquareTableTop` USD root frame.
- `/table/assembled_hole`: OmniReset assembled/hole frame from tabletop metadata.
- `/leg`: draggable raw `SquareLeg` USD root frame.
- `/leg/policy_frame`: virtual SimToolReal policy frame, with red `+x` pointing handle-to-threads.
- `/leg/assembled_tip`: OmniReset assembled/thread-tip frame from leg metadata.
- `/goal_leg`: translucent assembled-pose ghost.

This viewer also answers a common frame question: the SquareLeg mesh is not rewritten to make USD `+x` point along the long axis. The physical USD remains in its original frame. The handle-to-thread long-axis convention is applied as a virtual policy-frame transform in observations and when converting scripted policy goals back to the USD asset frame.

The W&B/Three.js pose viewer is intentionally lighter weight than the Viser USD viewer. It still simulates with the OmniReset USDs, but renders the FurnitureBench URDF/OBJ meshes by URL instead of embedding USD geometry into the HTML. Those source OBJ meshes are Y-up:

- `square_table_leg1.obj`: long axis is local `Y`.
- `square_table_top.obj`: tabletop normal/thickness is local `Y`.

OmniReset's USDs are Z-up:

- `square_leg.usd`: long axis is local `Z`, with negative `Z` toward the screw threads.
- `square_table_top.usd`: tabletop normal/thickness is local `Z`.

For viewer-only HTML, `pose_viewer.py` rewrites the FurnitureBench URDF text to add a `+90 deg` roll about `X` on the mesh visual/collision origins, plus sub-millimeter bbox-center offsets. That makes the URL-loaded URDF visuals line up with the sim USD root frames without changing physics assets or bloating the HTML. The Three.js OBJ loader deliberately does not request a hard-coded `material.mtl`; FurnitureBench OBJ files reference per-object MTL names, and the pose viewer only needs geometry/color overrides.

### FurnitureBench mesh structure

The FurnitureBench leg/tabletop pair is not a single monolithic mesh, and it is not loaded through URDF in this rollout. Each part is one USD file with multiple mesh prims:

- `SquareLeg/square_leg.usd`, root prim `/square_leg`, currently 4 mesh prims:
  - `/square_leg/visuals/bolt`
  - `/square_leg/visuals/leg`
  - `/square_leg/collisions/leg`
  - `/square_leg/collisions/bolt`
- `SquareTableTop/square_table_top.usd`, root prim `/square_table`, currently 18 mesh prims:
  - visual tabletop/walls/hole meshes
  - matching collision tabletop/walls/hole meshes

The important authored collision approximations are still on the source USDs:

- leg handle collision: convex hull
- leg threaded bolt collision: SDF
- tabletop hole collisions: SDF
- tabletop and wall collisions: convex hull

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

The policy-frame goal orientation points the leg long axis down, so the threaded end points toward the hole. Because policy `+x` points down into the hole, positive rotation about local policy `+x` appears clockwise when viewed from above looking down the hole. Positive `--leg_spin_turns` therefore means spin into the hole.

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
- Track positive yaw progress about local policy `+x`, which is clockwise when viewed from above/down the hole.
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

### Authoring mixed-collider USDs

A single rigid object can have multiple child collision mesh prims with different collision approximations. This is how the working FurnitureBench USDs are structured:

```text
/square_leg
  /visuals/leg             visible, no collision
  /visuals/bolt            visible, no collision
  /collisions/leg          invisible, collisionEnabled, approximation = convexHull
  /collisions/bolt         invisible, collisionEnabled, approximation = sdf

/square_table
  /visuals/table_top       visible, no collision
  /visuals/wall*           visible, no collision
  /visuals/hole*           visible, no collision
  /collisions/table_top    invisible, collisionEnabled, approximation = convexHull
  /collisions/wall*        invisible, collisionEnabled, approximation = convexHull
  /collisions/hole*        invisible, collisionEnabled, approximation = sdf
```

The root prim carries the rigid body, mass, kinematic, and gravity properties. The child collision mesh prims carry the per-part collider approximation. PhysX treats these children as a compound collider for the one root rigid body.

This is different from a simple URDF conversion. A URDF like:

```xml
<collision>
  <geometry>
    <mesh filename="../mesh/square_table/square_table_top.obj"/>
  </geometry>
  <sdf resolution="512"/>
</collision>
```

describes one collision mesh for the link. It does not identify which triangles are the hole/thread geometry and which triangles are the tabletop/body geometry. It is too coarse for "hole SDF, rest convex hull" unless the converter or a postprocess step can split the mesh into semantic child prims.

The practical rule is:

- Use Blender, CAD, or a trusted mesh-processing step to split the asset into meaningful submeshes: `hole*`, `thread*`, `bolt*`, `table_top`, `wall*`, `leg`, `handle`, etc.
- Use SDF only on the detailed threaded or receiving-hole surfaces where non-convex contact matters.
- Use convex hulls for simple cuboids, walls, handles, and support geometry.
- Keep visual meshes separate from collision meshes; visual prims should not have collision APIs.
- Do not rely on SAM or 2D segmentation for this. The needed output is named, watertight 3D collision submeshes, not image masks.

Isaac Sim supports the final mixed setup, but it does not automatically know which faces should become SDF and which should become convex hull. That semantic split has to come from the mesh authoring process or from a custom script.

### Recommended conversion workflow

For a new object pair, prefer a repeatable CLI/postprocess workflow after one-time mesh cleanup:

1. Prepare the mesh in Blender/CAD.
   - Bake real-world scale and orientation into vertices.
   - Set the object origin intentionally.
   - Split semantic collision pieces into named mesh objects.
   - For the leg, split handle/body from threaded bolt.
   - For the tabletop, split simple plate/walls from the receiving holes.
2. Export as USD, USDZ, OBJ, or another format Isaac Sim can import while preserving object names.
3. Run a custom USD builder/postprocessor that:
   - creates a clean root prim, e.g. `/square_table`;
   - copies imported meshes into `/visuals`;
   - copies the same or simplified meshes into `/collisions`;
   - makes collision prims invisible;
   - applies `physics:approximation = sdf` to names matching `hole*`, `thread*`, or `bolt*`;
   - applies `physics:approximation = convexHull` to names matching `table_top`, `wall*`, `leg`, `handle*`, etc.;
   - applies rigid-body/mass/kinematic/gravity properties to the root prim.

The existing `scripts/tools/convert_mesh.py` is useful but not sufficient by itself for this mixed case. It accepts `--collision-approximation sdf` and `--collision-approximation convexHull`, but that applies one collision approximation to the converted asset. Mixed SDF/convex assets need per-child mesh prim authoring.

The workflow can still be CLI-based. The missing piece is a script shaped roughly like:

```bash
python scripts/tools/build_mixed_collider_usd.py \
  --input /path/to/square_table_top.usd \
  --output /path/to/SquareTableTop/square_table_top.usd \
  --root square_table \
  --sdf-regex "hole.*|thread.*|bolt.*" \
  --convex-regex "table_top|wall.*|leg|handle.*" \
  --kinematic
```

This script should fail loudly if a mesh name matches neither rule or if all SDF/convex groups are empty. Silent fallbacks are dangerous for threaded insertion tasks because a visually correct asset can have physically wrong collisions.

### Mixed-collider validation checklist

Use static USD inspection before running physics:

- Root prim has the expected rigid body API.
- Dynamic insertive object root has mass and gravity enabled.
- Receptive fixture root is kinematic and gravity disabled.
- Visual mesh prims have no `PhysicsCollisionAPI`.
- Collision mesh prims are invisible and have `physics:collisionEnabled = True`.
- Threaded bolt and receiving-hole collision prims have `physics:approximation = sdf`.
- Handle, wall, plate, and other simple collision prims have `physics:approximation = convexHull`.
- Collision mesh bounds roughly match the visual geometry and are not scaled or rotated unexpectedly.

Then use visual and physics checks:

- Visualize the USD with color-coded collision overlays: SDF parts in one color, convex parts in another, visual meshes semi-transparent or separate.
- Load the asset in Isaac Sim headless and step the world to confirm PhysX can cook the SDF colliders without errors.
- Spawn the receptive object kinematic and the insertive object dynamic.
- Negative test: push straight down without spin. The threaded object should not teleport through the hole.
- Positive test: push while spinning in the correct direction. The threads should advance without explosions or severe penetration.
- Record video and trajectory logs for regressions.

These validation steps are more important than the exact authoring tool. Blender/Isaac Sim GUI is fine for initial debugging, but any asset we depend on for training should be reproducible by a script and validated by static USD inspection plus a small physics smoke test.

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

For large-batch training, VRAM is not the only limiter. A 2026-05-06 RTX PRO 6000 smoke at `12288` envs fit in GPU memory but produced PhysX `PxGpuDynamicsMemoryConfig::collisionStackSize buffer overflow` errors and dropped contacts. That is bad data for this task. The cluster launcher caps the RTX PRO 6000 jobs at `6144` envs / `1024` SAPG block size unless a higher count is re-smoked without any collision-stack warnings.

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
