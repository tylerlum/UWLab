# Distillation Follow-Ups

This is the working backlog for the Isaac Sim distillation path. The current priority is to keep the simple, high-signal pieces working before adding more sensors or visualization layers.

## Immediate

1. **W&B rollout viewer**
   - Log an interactive rollout artifact to W&B during eval/training.
   - Current v1 should show robot body positions, object pose, goal pose, table, goal index, and rollout time.
   - Next improvement: overlay the student auxiliary predicted object position with a distinct color from actual object and goal.

2. **W&B rollout video**
   - Log the same rendered low-resolution RGB stream used by the policy as a W&B video.
   - Keep this tied to existing camera render settings; do not create a separate lower-resolution debug camera unless needed.
   - Use it for quick sanity checks that the policy input moves and matches the intended viewpoint.

3. **Low-resolution RGB u1 experiments**
   - Continue using the `u1` online update cadence because it has outperformed `u4` so far.
   - Compare same-FOV lower render resolutions: `320x180`, `160x90`, and `80x45`.
   - Preserve aspect ratio and camera FOV when reducing resolution.

## Wrist Camera

4. **Wrist camera placement iteration**
   - Render RGB and depth debug images from the current wrist-camera guess.
   - Tune the pose interactively based on what is visible.
   - The desired camera is near the palm, backed away from the fingers, looking toward the palm/finger workspace.
   - Depth ranges will differ from the third-person camera, so inspect raw min/max and normalized images separately.

5. **Wrist camera distillation ablation**
   - Once placement is acceptable, run a wrist-only student.
   - Compare against the current third-person camera at matched resolution, update cadence, and approximate training budget.

6. **Third-person plus wrist camera**
   - Add a multi-camera student input path.
   - Default architecture should use separate visual encoders per camera followed by late fusion, because the viewpoints and depth ranges are substantially different.
   - Shared encoders are a later ablation, not the default.

## Later

7. **Predicted object-position visualization**
   - Add optional W&B viewer traces for the student auxiliary object-position prediction.
   - Show actual object, goal object, and predicted object as separate colors.
   - Use this to debug whether image features are learning object localization before successful manipulation.

8. **Longer DEXTRAH-style loops**
   - Run beta=0 from the start as a direct DEXTRAH-style comparison.
   - Run longer after beta reaches zero for curriculum runs, since image policies were still improving at the end.
   - Track student-only monitor envs as the main progress metric, not only mixed-policy envs.

9. **Reset-condition audit**
   - Keep Isaac Sim reset conditions aligned with the Isaac Gym environment.
   - Confirm time-limit reset, drop reset, goal/task completion reset, and any object/robot out-of-bounds conditions.

10. **Action chunking**
    - Consider only after the single-step action distillation baseline is clearly understood.
    - If tested, predict a short future action sequence but still use teacher corrections at each visited state.
