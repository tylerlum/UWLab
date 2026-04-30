from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    _compute_keypoint_positions,
    matrix_to_quaternion_xyzw_scipy,
    OBJECT_KEYPOINT_OFFSETS_np,
    PALM_OFFSET_np,
    FINGERTIP_OFFSETS_np,
    Q_LOWER_LIMITS_np,
    Q_UPPER_LIMITS_np,
)
from isaacgymenvs.utils.torch_jit_utils import quat_rotate, unscale
from isaacsim_conversion.isaacsim_env import (
    CONTROL_DT,
    DEFAULT_ASSET_FRICTION,
    DEFAULT_JOINT_POS,
    FINGERTIP_FRICTION,
    FINGERTIP_LINK_NAMES,
    JOINT_DAMPINGS,
    JOINT_DAMPINGS_COMPENSATED,
    JOINT_NAMES_ISAACGYM,
    JOINT_STIFFNESSES,
    JOINT_STIFFNESSES_COMPENSATED,
    PHYSICS_DT,
    PHYSICS_SUBSTEPS,
    _log,
)
from isaacsim_conversion.task_utils import CameraPose, TaskSpec, xyzw_to_wxyz
from isaacsim_conversion.task_utils import CameraIntrinsics


OBS_LIST = [
    "joint_pos",
    "joint_vel",
    "prev_action_targets",
    "palm_pos",
    "palm_rot",
    "object_rot",
    "fingertip_pos_rel_palm",
    "keypoints_rel_palm",
    "keypoints_rel_goal",
    "object_scales",
]
OBJECT_BASE_SIZE = 0.04
KEYPOINT_SCALE = 1.5


@dataclass
class SimState:
    q: np.ndarray
    qd: np.ndarray
    object_pose: np.ndarray
    goal_pose: np.ndarray
    goal_idx: np.ndarray
    kp_dist: np.ndarray
    near_goal_steps: np.ndarray
    object_pos_world_env_frame: np.ndarray


class IsaacSimDistillEnv:
    def __init__(
        self,
        task_spec: TaskSpec,
        app: Any,
        headless: bool,
        camera_modality: str = "depth",
        use_real_camera_transform: bool = True,
        camera_pose_override: CameraPose | None = None,
        camera_intrinsics: CameraIntrinsics | None = None,
        num_envs: int = 1,
        env_spacing: float = 2.0,
        object_start_mode: str = "fixed",
        object_pos_noise_xyz: tuple[float, float, float] = (0.03, 0.03, 0.01),
        object_yaw_noise_deg: float = 20.0,
        enable_camera: bool = True,
        camera_backend: str = "tiled",
        ground_plane_size: float = 500.0,
        depth_preprocess_mode: str = "clip_divide",
        depth_min_m: float = 0.0,
        depth_max_m: float = 5.0,
        episode_length: int = 600,
        reset_when_dropped: bool = False,
    ):
        self.task_spec = task_spec
        self.camera_modality = camera_modality
        self.num_envs = num_envs
        self.env_spacing = env_spacing
        self.object_start_mode = object_start_mode
        self.object_pos_noise_xyz = np.array(object_pos_noise_xyz, dtype=np.float32)
        self.object_yaw_noise_deg = float(object_yaw_noise_deg)
        self.app = app
        self.headless = headless
        self.camera_pose = camera_pose_override or task_spec.camera_pose
        self.camera_intrinsics = camera_intrinsics or CameraIntrinsics()
        self.use_real_camera_transform = use_real_camera_transform
        self.enable_camera = enable_camera
        self.camera_backend = camera_backend
        self.ground_plane_size = float(ground_plane_size)
        if self.camera_backend not in {"tiled", "standard"}:
            raise ValueError(f"Unsupported camera_backend={self.camera_backend!r}")
        if self.ground_plane_size <= 0:
            raise ValueError("ground_plane_size must be > 0")
        self.depth_preprocess_mode = depth_preprocess_mode
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)
        self.episode_length = int(episode_length)
        self.reset_when_dropped = bool(reset_when_dropped)
        if self.depth_preprocess_mode not in {"clip_divide", "window_normalize", "metric"}:
            raise ValueError(f"Unsupported depth_preprocess_mode={self.depth_preprocess_mode!r}")
        if self.depth_max_m <= self.depth_min_m:
            raise ValueError("depth_max_m must be greater than depth_min_m")
        self.action_dim = 29
        self.student_proprio_dim = 29 + 29 + 29 + 1
        self.object_scales_batch = np.repeat(self.task_spec.object_scales.astype(np.float32), self.num_envs, axis=0)
        self.camera_data_types = self._data_types_for_modality(camera_modality)
        self.goal_idx = np.zeros(self.num_envs, dtype=np.int32)
        self.max_goal_idx = np.zeros(self.num_envs, dtype=np.int32)
        self.near_goal_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.progress_buf = np.zeros(self.num_envs, dtype=np.int32)
        self.lifted_object = np.zeros(self.num_envs, dtype=bool)
        self.reset_reason_counts = {
            "object_z_low": 0,
            "time_limit": 0,
            "max_goals": 0,
            "hand_far": 0,
            "dropped_after_lift": 0,
        }
        self.goal_pose = np.repeat(self.task_spec.goals[0][None], self.num_envs, axis=0).astype(np.float32)
        self.current_start_pose = np.repeat(self.task_spec.start_pose[None], self.num_envs, axis=0).astype(np.float32)
        self.prev_targets: np.ndarray | None = None
        self.camera_world_pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.camera_world_quat_wxyz = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.permutation: np.ndarray | None = None
        self.inverse_permutation: np.ndarray | None = None
        self.permutation_torch: torch.Tensor | None = None
        self.inverse_permutation_torch: torch.Tensor | None = None
        self.palm_body_idx: int | None = None
        self.fingertip_body_indices: torch.Tensor | None = None

        self._setup_scene()

    def _camera_mount_mode(self) -> str:
        return getattr(self.camera_pose, "mount", "world")

    def _data_types_for_modality(self, modality: str) -> list[str]:
        if modality == "depth":
            return ["distance_to_image_plane"]
        if modality == "rgb":
            return ["rgb"]
        if modality == "rgbd":
            return ["rgb", "distance_to_image_plane"]
        raise ValueError(f"Unsupported student camera modality: {modality}")

    def _convert_robot_urdf(self, sim_utils, UrdfConverterCfg, UrdfConverter, urdf_path: str) -> str:
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir="/tmp/isaaclab_usd_cache_distill/robot",
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=True,
            merge_fixed_joints=True,
            self_collision=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=JOINT_STIFFNESSES_COMPENSATED,
                    damping=JOINT_DAMPINGS_COMPENSATED,
                ),
            ),
        )
        return UrdfConverter(cfg).usd_path

    def _convert_static_urdf(self, UrdfConverterCfg, UrdfConverter, urdf_path: str, usd_dir: str, fix_base: bool) -> str:
        cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=usd_dir,
            force_usd_conversion=True,
            make_instanceable=False,
            fix_base=fix_base,
            merge_fixed_joints=True,
            joint_drive=None,
        )
        return UrdfConverter(cfg).usd_path

    def _make_scene_cfg(self, sim_utils, robot_usd: str, table_usd: str, object_usd: str):
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        robot_joint_vel = {name: 0.0 for name in JOINT_NAMES_ISAACGYM}
        default_joint_pos = dict(DEFAULT_JOINT_POS)
        if self.enable_camera:
            from isaaclab.sensors import CameraCfg, TiledCameraCfg

            camera_types = list(self.camera_data_types)
            camera_cfg_cls = TiledCameraCfg if self.camera_backend == "tiled" else CameraCfg
            camera_mount = self._camera_mount_mode()
            spawn_tiled_at_parent_pose = self.camera_backend == "tiled" and camera_mount in {"world", "wrist"}
            camera_offset_pos = self.camera_pose.pos if spawn_tiled_at_parent_pose else (0.0, 0.0, 0.0)
            camera_offset_rot = self.camera_pose.quat_wxyz if spawn_tiled_at_parent_pose else (1.0, 0.0, 0.0, 0.0)
            camera_offset_convention = self.camera_pose.convention if spawn_tiled_at_parent_pose else "ros"
            if self.camera_backend == "tiled" and camera_mount == "wrist":
                camera_link_name = self.camera_pose.link_name or "iiwa14_link_7"
                camera_prim_path = f"{{ENV_REGEX_NS}}/Robot/{camera_link_name}/DistillCamera"
            else:
                camera_prim_path = "{ENV_REGEX_NS}/DistillCamera"

            @configclass
            class DistillSceneCfg(InteractiveSceneCfg):
                robot = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/Robot",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=robot_usd,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True,
                            max_depenetration_velocity=1000.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            enabled_self_collisions=False,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=0,
                        ),
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.8, 0.0),
                        joint_pos=default_joint_pos,
                        joint_vel=robot_joint_vel,
                    ),
                    actuators={
                        "arm": ImplicitActuatorCfg(
                            joint_names_expr=["iiwa14_joint_.*"],
                            stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("iiwa14")},
                            damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("iiwa14")},
                        ),
                        "hand": ImplicitActuatorCfg(
                            joint_names_expr=["left_.*"],
                            stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("left")},
                            damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("left")},
                        ),
                    },
                )

                table = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=table_usd,
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
                )

                object = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=object_usd,
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                    ),
                )

                camera = camera_cfg_cls(
                    prim_path=camera_prim_path,
                    update_period=0,
                    update_latest_camera_pose=True,
                    height=self.camera_intrinsics.height,
                    width=self.camera_intrinsics.width,
                    data_types=camera_types,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=self.camera_intrinsics.focal_length,
                        focus_distance=self.camera_intrinsics.focus_distance,
                        horizontal_aperture=self.camera_intrinsics.horizontal_aperture,
                        clipping_range=self.camera_intrinsics.clipping_range,
                    ),
                    offset=camera_cfg_cls.OffsetCfg(
                        # Static world TiledCamera render products do not reliably
                        # pick up poses written after initialization, so spawn them
                        # at the per-env local camera pose. Tiled wrist cameras are
                        # spawned under the wrist link with the configured local
                        # offset so articulation motion drives the camera transform.
                        pos=camera_offset_pos,
                        rot=camera_offset_rot,
                        convention=camera_offset_convention,
                    ),
                )

                light = AssetBaseCfg(
                    prim_path="/World/DomeLight",
                    spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
                )
        else:
            @configclass
            class DistillSceneCfg(InteractiveSceneCfg):
                robot = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/Robot",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=robot_usd,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True,
                            max_depenetration_velocity=1000.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            enabled_self_collisions=False,
                            solver_position_iteration_count=8,
                            solver_velocity_iteration_count=0,
                        ),
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.8, 0.0),
                        joint_pos=default_joint_pos,
                        joint_vel=robot_joint_vel,
                    ),
                    actuators={
                        "arm": ImplicitActuatorCfg(
                            joint_names_expr=["iiwa14_joint_.*"],
                            stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("iiwa14")},
                            damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("iiwa14")},
                        ),
                        "hand": ImplicitActuatorCfg(
                            joint_names_expr=["left_.*"],
                            stiffness={k: v for k, v in JOINT_STIFFNESSES.items() if k.startswith("left")},
                            damping={k: v for k, v in JOINT_DAMPINGS.items() if k.startswith("left")},
                        ),
                    },
                )

                table = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Table",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=table_usd,
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38)),
                )

                object = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=object_usd,
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.002,
                            rest_offset=0.0,
                        ),
                    ),
                )

                light = AssetBaseCfg(
                    prim_path="/World/DomeLight",
                    spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
                )

        return DistillSceneCfg

    def _setup_scene(self):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        # IsaacLab's default ground plane is 100m x 100m. With many envs at
        # env_spacing=4.0, far envs can render off-plane as a white background,
        # which corrupts camera policy inputs while physics still runs.
        ground_plane_cfg = sim_utils.GroundPlaneCfg(size=(self.ground_plane_size, self.ground_plane_size))
        ground_plane_cfg.func("/World/GroundPlane", ground_plane_cfg)
        sim_cfg = SimulationCfg(
            dt=PHYSICS_DT,
            render_interval=PHYSICS_SUBSTEPS,
            gravity=(0.0, 0.0, -9.81),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=DEFAULT_ASSET_FRICTION,
                dynamic_friction=DEFAULT_ASSET_FRICTION,
                restitution=0.0,
            ),
            physx=PhysxCfg(
                solver_type=1,
                bounce_threshold_velocity=0.2,
                friction_offset_threshold=0.04,
                friction_correlation_distance=0.025,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        _log(
            "SimulationContext created "
            f"(control_dt={CONTROL_DT:.6f}, physics_dt={PHYSICS_DT:.6f}, substeps={PHYSICS_SUBSTEPS})"
        )

        robot_usd = self._convert_robot_urdf(sim_utils, UrdfConverterCfg, UrdfConverter, self.task_spec.robot_urdf)
        table_usd = self._convert_static_urdf(
            UrdfConverterCfg, UrdfConverter, self.task_spec.table_urdf, "/tmp/isaaclab_usd_cache_distill/table", True
        )
        object_usd = self._convert_static_urdf(
            UrdfConverterCfg, UrdfConverter, self.task_spec.object_urdf, "/tmp/isaaclab_usd_cache_distill/object", False
        )

        scene_cfg_type = self._make_scene_cfg(sim_utils, robot_usd, table_usd, object_usd)
        self.scene = InteractiveScene(
            scene_cfg_type(
                num_envs=self.num_envs,
                env_spacing=self.env_spacing,
                lazy_sensor_update=False,
                replicate_physics=True,
            )
        )
        self.robot = self.scene.articulations["robot"]
        self.object_rigid = self.scene.rigid_objects["object"]
        self.camera = self.scene.sensors["camera"] if self.enable_camera else None
        self.env_origins = self.scene.env_origins
        self.device = self.sim.device
        self.q_lower_limits_t = torch.tensor(Q_LOWER_LIMITS_np, dtype=torch.float32, device=self.device)
        self.q_upper_limits_t = torch.tensor(Q_UPPER_LIMITS_np, dtype=torch.float32, device=self.device)
        self.palm_offset_t = torch.tensor(PALM_OFFSET_np, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.fingertip_offsets_t = torch.tensor(FINGERTIP_OFFSETS_np, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.object_keypoint_offsets_t = (
            torch.tensor(OBJECT_KEYPOINT_OFFSETS_np, dtype=torch.float32, device=self.device)
            * (OBJECT_BASE_SIZE * KEYPOINT_SCALE / 2.0)
        ).unsqueeze(0)
        self._apply_physics_material_overrides(sim_utils)
        self.sim.reset()
        self.scene.reset()
        self._validate_joint_ordering()
        if self.camera is not None:
            _log(
                "Distill camera created "
                f"(backend={self.camera_backend}, instances={self.camera.num_instances}, modality={self.camera_modality}, pos={self.camera_pose.pos}, "
                f"quat_wxyz={self.camera_pose.quat_wxyz}, convention={self.camera_pose.convention})"
            )
        else:
            _log("Distill camera disabled for this run")
        self.reset()

    def _validate_joint_ordering(self):
        sim_names = list(self.robot.joint_names)
        self.palm_body_idx = self.robot.body_names.index("iiwa14_link_7")
        self.fingertip_body_indices = torch.tensor(
            [self.robot.body_names.index(name) for name in FINGERTIP_LINK_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        if sim_names == JOINT_NAMES_ISAACGYM:
            self.permutation = None
            self.inverse_permutation = None
            self.permutation_torch = None
            self.inverse_permutation_torch = None
            _log("Distill joint ordering matches Isaac Gym")
            return
        self.permutation = np.array([sim_names.index(name) for name in JOINT_NAMES_ISAACGYM], dtype=np.int32)
        self.inverse_permutation = np.zeros_like(self.permutation)
        for gym_idx, sim_idx in enumerate(self.permutation):
            self.inverse_permutation[sim_idx] = gym_idx
        self.permutation_torch = torch.tensor(self.permutation, dtype=torch.long, device=self.device)
        self.inverse_permutation_torch = torch.tensor(self.inverse_permutation, dtype=torch.long, device=self.device)
        _log(f"Distill joint permutation set: {self.permutation}")

    def _sim_to_gym_order(self, values: np.ndarray) -> np.ndarray:
        if self.permutation is None:
            return values
        return values[:, self.permutation]

    def _gym_to_sim_order(self, values: np.ndarray) -> np.ndarray:
        if self.permutation is None:
            return values
        reordered = np.zeros_like(values)
        reordered[:, self.permutation] = values
        return reordered

    def _iter_collision_prim_paths(self, root_prim_path: str) -> list[str]:
        from isaaclab.sim.utils import get_current_stage

        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            return []

        collision_paths: list[str] = []
        prims_to_visit = [root_prim]
        while prims_to_visit:
            prim = prims_to_visit.pop()
            if prim.GetName() == "collisions":
                collision_paths.append(str(prim.GetPath()))
            prims_to_visit.extend(list(prim.GetChildren()))
        return collision_paths

    def _set_collision_offsets(self, collision_prim_paths: list[str], contact_offset: float, rest_offset: float):
        from isaaclab.sim.utils import get_current_stage
        from pxr import PhysxSchema

        stage = get_current_stage()
        for prim_path in collision_prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
            if not physx_collision_api:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision_api.CreateContactOffsetAttr().Set(contact_offset)
            physx_collision_api.CreateRestOffsetAttr().Set(rest_offset)

    def _apply_material_to_collision_prims(self, material_path: str, collision_prim_paths: list[str]):
        from isaaclab.sim.utils import bind_physics_material

        raw_bind_physics_material = getattr(bind_physics_material, "__wrapped__", bind_physics_material)
        for prim_path in collision_prim_paths:
            raw_bind_physics_material(prim_path, material_path)

    def _apply_physics_material_overrides(self, sim_utils):
        all_collision_paths: list[str] = []
        fingertip_collision_paths: list[str] = []
        for env_id in range(self.num_envs):
            env_ns = f"/World/envs/env_{env_id}"
            robot_root = f"{env_ns}/Robot"
            table_root = f"{env_ns}/Table"
            object_root = f"{env_ns}/Object"
            all_collision_paths.extend(self._iter_collision_prim_paths(robot_root))
            all_collision_paths.extend(self._iter_collision_prim_paths(table_root))
            all_collision_paths.extend(self._iter_collision_prim_paths(object_root))
            fingertip_collision_paths.extend(
                f"{robot_root}/{link_name}/collisions" for link_name in FINGERTIP_LINK_NAMES
            )

        self._set_collision_offsets(all_collision_paths, contact_offset=0.002, rest_offset=0.0)

        default_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=DEFAULT_ASSET_FRICTION,
            dynamic_friction=DEFAULT_ASSET_FRICTION,
            restitution=0.0,
        )
        default_material_path = "/World/PhysicsMaterials/distill_default_asset_material"
        default_material_cfg.func(default_material_path, default_material_cfg)
        self._apply_material_to_collision_prims(default_material_path, all_collision_paths)

        fingertip_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=FINGERTIP_FRICTION,
            dynamic_friction=FINGERTIP_FRICTION,
            restitution=0.0,
        )
        fingertip_material_path = "/World/PhysicsMaterials/distill_fingertip_material"
        fingertip_material_cfg.func(fingertip_material_path, fingertip_material_cfg)
        self._apply_material_to_collision_prims(fingertip_material_path, fingertip_collision_paths)
        _log(
            "Distill physics materials applied "
            f"(default_friction={DEFAULT_ASSET_FRICTION}, fingertip_friction={FINGERTIP_FRICTION}, "
            f"colliders={len(all_collision_paths)})"
        )

    def _apply_camera_world_poses(self, env_ids: torch.Tensor | None = None):
        if self.camera is None:
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        env_ids_np = env_ids.detach().cpu().numpy()
        if self._camera_mount_mode() == "wrist":
            positions, orientations = self._compute_wrist_camera_world_poses(env_ids_np)
        else:
            env_origins = self.env_origins[env_ids].detach().cpu().numpy()
            positions = env_origins + np.asarray(self.camera_pose.pos, dtype=np.float32)[None, :]
            orientations = np.repeat(np.asarray(self.camera_pose.quat_wxyz, dtype=np.float32)[None, :], len(env_ids_np), axis=0)
        if self.camera_backend == "tiled":
            self.camera_world_pos[env_ids_np] = positions
            self.camera_world_quat_wxyz[env_ids_np] = orientations
            return
        self.camera.set_world_poses(
            positions=positions,
            orientations=orientations,
            env_ids=env_ids,
            convention=self.camera_pose.convention,
        )
        self.camera_world_pos[env_ids_np] = positions
        self.camera_world_quat_wxyz[env_ids_np] = orientations

    def _compute_wrist_camera_world_poses(self, env_ids_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        link_name = self.camera_pose.link_name or "iiwa14_link_7"
        link_idx = self.robot.body_names.index(link_name)
        body_state = self.robot.data.body_state_w[env_ids_np, link_idx]
        link_pos = body_state[:, :3].detach().cpu().numpy()
        link_quat_wxyz = body_state[:, 3:7].detach().cpu().numpy()
        link_rot = R.from_quat(link_quat_wxyz[:, [1, 2, 3, 0]]).as_matrix()

        cam_offset = np.asarray(self.camera_pose.pos, dtype=np.float32)[None, :]
        cam_pos = link_pos + np.einsum("nij,nj->ni", link_rot, cam_offset)

        rel_quat_wxyz = np.asarray(self.camera_pose.quat_wxyz, dtype=np.float32)
        rel_rot = R.from_quat(rel_quat_wxyz[[1, 2, 3, 0]]).as_matrix()
        cam_rot = link_rot @ rel_rot[None, :, :]
        cam_quat_xyzw = matrix_to_quaternion_xyzw_scipy(cam_rot)
        cam_quat_wxyz = cam_quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
        return cam_pos.astype(np.float32), cam_quat_wxyz

    def _sample_start_poses(self) -> np.ndarray:
        start = np.repeat(self.task_spec.start_pose[None], self.num_envs, axis=0).astype(np.float32)
        if self.object_start_mode == "fixed":
            return start
        if self.object_start_mode != "randomized":
            raise ValueError(f"Unsupported object_start_mode: {self.object_start_mode}")

        noise = (np.random.uniform(-1.0, 1.0, size=(self.num_envs, 3)).astype(np.float32) * self.object_pos_noise_xyz[None])
        start[:, :3] += noise
        yaw_noise = np.deg2rad(
            np.random.uniform(-self.object_yaw_noise_deg, self.object_yaw_noise_deg, size=self.num_envs).astype(np.float32)
        )
        for i in range(self.num_envs):
            base = R.from_quat(start[i, 3:7])
            yaw = R.from_euler("z", float(yaw_noise[i]))
            start[i, 3:7] = (yaw * base).as_quat().astype(np.float32)
        start[:, 2] = np.maximum(start[:, 2], self.task_spec.start_pose[2] - self.object_pos_noise_xyz[2])
        return start

    def _local_pose_to_world_pose(self, local_pose_xyzw: np.ndarray) -> torch.Tensor:
        world_pose = np.zeros((self.num_envs, 7), dtype=np.float32)
        world_pose[:, :3] = local_pose_xyzw[:, :3] + self.env_origins.cpu().numpy()
        for i in range(self.num_envs):
            world_pose[i, 3:] = xyzw_to_wxyz(local_pose_xyzw[i, 3:])
        return torch.tensor(world_pose, dtype=torch.float32, device=self.device)

    def reset(self):
        for key in self.reset_reason_counts:
            self.reset_reason_counts[key] = 0
        self.max_goal_idx[:] = 0
        self._reset_envs(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _reset_envs(self, env_ids: torch.Tensor, reset_reasons: dict[str, np.ndarray] | None = None):
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        env_ids_np = env_ids.detach().cpu().numpy()
        if reset_reasons is not None:
            for key, mask in reset_reasons.items():
                if key in self.reset_reason_counts:
                    self.reset_reason_counts[key] += int(np.count_nonzero(mask))
        root_state = self.robot.data.default_root_state.clone()
        root_state[env_ids, :3] += self.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root_state[env_ids, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_state[env_ids, 7:], env_ids=env_ids)
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos[env_ids], env_ids=env_ids)

        start_poses = self._sample_start_poses()[env_ids_np]
        self.current_start_pose[env_ids_np] = start_poses
        object_pose_w = self._local_pose_to_world_pose(self.current_start_pose)
        zeros_vel = torch.zeros((len(env_ids_np), 6), dtype=torch.float32, device=self.device)
        self.object_rigid.write_root_pose_to_sim(object_pose_w[env_ids], env_ids=env_ids)
        self.object_rigid.write_root_velocity_to_sim(zeros_vel, env_ids=env_ids)
        self._apply_camera_world_poses(env_ids)

        self.scene.write_data_to_sim()
        self.sim.step(render=not self.headless)
        self.scene.update(PHYSICS_DT)
        if self.camera is not None:
            self.sim.render()
        if self.camera is not None:
            self.camera.update(0.0, force_recompute=True)
        joint_pos_sim = self.robot.data.joint_pos.detach().cpu().numpy().copy()
        if self.prev_targets is None:
            self.prev_targets = self._sim_to_gym_order(joint_pos_sim)
        else:
            self.prev_targets[env_ids_np] = self._sim_to_gym_order(joint_pos_sim)[env_ids_np]
        self.goal_idx[env_ids_np] = 0
        self.near_goal_steps[env_ids_np] = 0
        self.progress_buf[env_ids_np] = 0
        self.lifted_object[env_ids_np] = False
        self.goal_pose[env_ids_np] = np.repeat(self.task_spec.goals[0][None], len(env_ids_np), axis=0).astype(np.float32)

    def apply_action(self, targets: np.ndarray):
        self.prev_targets = targets.copy()
        targets_sim = self._gym_to_sim_order(targets)
        targets_t = torch.tensor(targets_sim, dtype=torch.float32, device=self.device)
        self.robot.set_joint_position_target(targets_t)

    def step(self, render: bool = True):
        self.scene.write_data_to_sim()
        for _ in range(PHYSICS_SUBSTEPS):
            self.sim.step(render=render)
        self.scene.update(PHYSICS_DT)
        self.progress_buf += 1
        if self.camera is not None and self._camera_mount_mode() == "wrist":
            self._apply_camera_world_poses()
            if self.camera_backend != "tiled":
                # Standard cameras are moved explicitly; push those dynamic wrist
                # poses before forcing a render/update.
                self.scene.write_data_to_sim()
        if self.camera is not None:
            self.sim.render()
            self.camera.update(PHYSICS_DT, force_recompute=True)

    def compute_sim_state(self) -> SimState:
        q = self._sim_to_gym_order(self.robot.data.joint_pos.detach().cpu().numpy())
        qd = self._sim_to_gym_order(self.robot.data.joint_vel.detach().cpu().numpy())
        object_pos_w = self.object_rigid.data.root_pos_w.detach().cpu().numpy()
        object_quat_w = self.object_rigid.data.root_quat_w.detach().cpu().numpy()

        object_pose = np.zeros((self.num_envs, 7), dtype=np.float32)
        # Keep object pose env-local by subtracting scene env origins. This matches the
        # DEXTRAH env convention:
        #   object_pos = self.object.data.root_pos_w - self.scene.env_origins
        # and is the frame used for aux_info["object_pos"] in their distillation code.
        object_pose[:, :3] = object_pos_w - self.env_origins.cpu().numpy()
        object_pose[:, 3:] = object_quat_w[:, [1, 2, 3, 0]]

        object_kps = _compute_keypoint_positions(object_pose, self.object_scales_batch)
        goal_kps = _compute_keypoint_positions(self.goal_pose, self.object_scales_batch)
        kp_dist = np.max(np.linalg.norm(object_kps - goal_kps, axis=-1), axis=-1).astype(np.float32)
        return SimState(
            q=q.copy(),
            qd=qd.copy(),
            object_pose=object_pose.copy(),
            goal_pose=self.goal_pose.copy(),
            goal_idx=self.goal_idx.copy(),
            kp_dist=kp_dist.copy(),
            near_goal_steps=self.near_goal_steps.copy(),
            # This is the DEXTRAH-style auxiliary target frame: env-local world position,
            # not camera-relative and not robot-relative.
            object_pos_world_env_frame=object_pose[:, :3].copy(),
        )

    def capture_viewer_frame(self, env_id: int, sim_state: SimState | None = None) -> dict[str, np.ndarray | list[str] | int | float]:
        if env_id < 0 or env_id >= self.num_envs:
            raise ValueError(f"viewer env_id {env_id} out of range for num_envs={self.num_envs}")
        if sim_state is None:
            sim_state = self.compute_sim_state()
        env_origin = self.env_origins[env_id].detach().cpu().numpy()
        body_state = self.robot.data.body_state_w[env_id].detach().cpu().numpy()
        body_positions = body_state[:, :3] - env_origin[None, :]
        robot_root_state = self.robot.data.root_state_w[env_id].detach().cpu().numpy()
        robot_base_pose = np.zeros(7, dtype=np.float32)
        robot_base_pose[:3] = robot_root_state[:3] - env_origin
        robot_base_pose[3:] = robot_root_state[3:7][[1, 2, 3, 0]]
        table_pose = np.array([0.0, 0.0, 0.38, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return {
            "env_id": int(env_id),
            "robot_joint_names": list(self.robot.joint_names),
            "robot_joint_pos": self.robot.data.joint_pos[env_id].detach().cpu().numpy().astype(np.float32),
            "robot_body_names": list(self.robot.body_names),
            "robot_body_positions": body_positions.astype(np.float32),
            "robot_base_pose": robot_base_pose.astype(np.float32),
            "object_pose": sim_state.object_pose[env_id].astype(np.float32),
            "goal_pose": sim_state.goal_pose[env_id].astype(np.float32),
            "table_pose": table_pose,
            "goal_idx": int(sim_state.goal_idx[env_id]),
            "kp_dist": float(sim_state.kp_dist[env_id]),
        }

    def maybe_advance_goal(self, sim_state: SimState) -> bool:
        lifted_now = sim_state.object_pose[:, 2] > (self.current_start_pose[:, 2] + 0.1)
        self.lifted_object |= lifted_now
        for env_id in range(self.num_envs):
            if self.goal_idx[env_id] >= len(self.task_spec.goals):
                continue
            if sim_state.kp_dist[env_id] < self.task_spec.keypoint_tolerance:
                self.near_goal_steps[env_id] += 1
            else:
                self.near_goal_steps[env_id] = 0
            if self.near_goal_steps[env_id] < self.task_spec.success_steps:
                continue
            self.goal_idx[env_id] += 1
            self.max_goal_idx[env_id] = max(self.max_goal_idx[env_id], self.goal_idx[env_id])
            self.near_goal_steps[env_id] = 0
            if self.goal_idx[env_id] < len(self.task_spec.goals):
                self.goal_pose[env_id] = self.task_spec.goals[self.goal_idx[env_id]]
        return bool(np.all(self.goal_idx >= len(self.task_spec.goals)))

    def compute_reset_masks(self, sim_state: SimState) -> dict[str, np.ndarray]:
        object_z_low = sim_state.object_pose[:, 2] < 0.1
        time_limit = self.progress_buf >= self.episode_length - 1
        max_goals = self.goal_idx >= len(self.task_spec.goals)
        hand_far = self._compute_hand_far_mask()
        dropped_after_lift = (
            (sim_state.object_pose[:, 2] < self.current_start_pose[:, 2])
            & self.lifted_object
            if self.reset_when_dropped
            else np.zeros(self.num_envs, dtype=bool)
        )
        return {
            "object_z_low": object_z_low,
            "time_limit": time_limit,
            "max_goals": max_goals,
            "hand_far": hand_far,
            "dropped_after_lift": dropped_after_lift,
        }

    def _compute_hand_far_mask(self) -> np.ndarray:
        body_state = self.robot.data.body_state_w
        fingertip_state = body_state[:, self.fingertip_body_indices]
        fingertip_pos = fingertip_state[:, :, :3] - self.env_origins.unsqueeze(1)
        object_pos = self.object_rigid.data.root_pos_w - self.env_origins
        distances = torch.linalg.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1)
        return (torch.max(distances, dim=-1).values > 1.5).detach().cpu().numpy()

    def reset_done_envs(self, sim_state: SimState) -> np.ndarray:
        reset_masks = self.compute_reset_masks(sim_state)
        done_mask = np.zeros(self.num_envs, dtype=bool)
        for mask in reset_masks.values():
            done_mask |= mask
        done_env_ids = np.where(done_mask)[0].astype(np.int64)
        if done_env_ids.size == 0:
            return done_env_ids
        reason_subset = {key: mask[done_env_ids] for key, mask in reset_masks.items()}
        self._reset_envs(torch.tensor(done_env_ids, device=self.device, dtype=torch.long), reset_reasons=reason_subset)
        return done_env_ids

    def task_finished(self, sim_state: SimState) -> bool:
        return bool(np.all(self.goal_idx >= len(self.task_spec.goals)))

    def compute_progress_metrics(self, sim_state: SimState) -> dict[str, float]:
        progress_goal_idx = np.maximum(self.goal_idx, self.max_goal_idx)
        goal_completion = progress_goal_idx.astype(np.float32) / len(self.task_spec.goals)
        return {
            "goal_idx": float(np.mean(progress_goal_idx)),
            "goal_completion_ratio": float(np.mean(goal_completion)),
            "kp_dist": float(np.mean(sim_state.kp_dist)),
            "near_goal_steps": float(np.mean(self.near_goal_steps)),
        }

    def reset_dropped_envs(self, sim_state: SimState) -> np.ndarray:
        reset_masks = {"object_z_low": sim_state.object_pose[:, 2] < 0.1}
        dropped_env_ids = np.where(reset_masks["object_z_low"])[0].astype(np.int64)
        if dropped_env_ids.size == 0:
            return dropped_env_ids
        _log(f"Resetting dropped envs: {dropped_env_ids.tolist()}")
        self._reset_envs(
            torch.tensor(dropped_env_ids, device=self.device, dtype=torch.long),
            reset_reasons={"object_z_low": reset_masks["object_z_low"][dropped_env_ids]},
        )
        return dropped_env_ids

    def build_teacher_obs(self, sim_state: SimState) -> np.ndarray:
        if self.prev_targets is None:
            raise RuntimeError("Environment must be reset before building teacher observations")
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        if self.permutation_torch is not None:
            joint_pos = joint_pos[:, self.permutation_torch]
            joint_vel = joint_vel[:, self.permutation_torch]
        joint_pos_unscaled = unscale(joint_pos, self.q_lower_limits_t, self.q_upper_limits_t)

        body_state = self.robot.data.body_state_w
        env_origins = self.env_origins

        palm_state = body_state[:, self.palm_body_idx]
        palm_pos = palm_state[:, :3] - env_origins
        palm_quat_wxyz = palm_state[:, 3:7]
        palm_quat_xyzw = palm_quat_wxyz[:, [1, 2, 3, 0]]
        palm_center_pos = palm_pos + quat_rotate(palm_quat_xyzw, self.palm_offset_t)

        fingertip_state = body_state[:, self.fingertip_body_indices]
        fingertip_pos = fingertip_state[:, :, :3] - env_origins.unsqueeze(1)
        fingertip_quat_xyzw = fingertip_state[:, :, 3:7][:, :, [1, 2, 3, 0]]
        fingertip_pos_offset = fingertip_pos + quat_rotate(
            fingertip_quat_xyzw.reshape(-1, 4),
            self.fingertip_offsets_t.reshape(-1, 3),
        ).reshape(self.num_envs, len(FINGERTIP_LINK_NAMES), 3)
        fingertip_pos_rel_palm = fingertip_pos_offset - palm_center_pos.unsqueeze(1)

        object_pos = self.object_rigid.data.root_pos_w - env_origins
        object_quat_xyzw = self.object_rigid.data.root_quat_w[:, [1, 2, 3, 0]]
        goal_pose_t = torch.tensor(self.goal_pose, dtype=torch.float32, device=self.device)
        object_scales_t = torch.tensor(self.object_scales_batch, dtype=torch.float32, device=self.device)
        object_keypoint_offsets = self.object_keypoint_offsets_t.repeat(self.num_envs, 1, 1) * object_scales_t.unsqueeze(1)
        obj_keypoint_pos = object_pos.unsqueeze(1) + quat_rotate(
            object_quat_xyzw.unsqueeze(1).repeat(1, object_keypoint_offsets.shape[1], 1).reshape(-1, 4),
            object_keypoint_offsets.reshape(-1, 3),
        ).reshape(self.num_envs, object_keypoint_offsets.shape[1], 3)
        goal_keypoint_pos = goal_pose_t[:, :3].unsqueeze(1) + quat_rotate(
            goal_pose_t[:, 3:7].unsqueeze(1).repeat(1, object_keypoint_offsets.shape[1], 1).reshape(-1, 4),
            object_keypoint_offsets.reshape(-1, 3),
        ).reshape(self.num_envs, object_keypoint_offsets.shape[1], 3)
        keypoints_rel_palm = obj_keypoint_pos - palm_center_pos.unsqueeze(1)
        keypoints_rel_goal = obj_keypoint_pos - goal_keypoint_pos

        prev_targets_t = torch.tensor(self.prev_targets, dtype=torch.float32, device=self.device)
        obs_dict = {
            "joint_pos": joint_pos_unscaled,
            "joint_vel": joint_vel,
            "prev_action_targets": prev_targets_t,
            "palm_pos": palm_center_pos,
            "palm_rot": palm_quat_xyzw,
            "object_rot": object_quat_xyzw,
            "fingertip_pos_rel_palm": fingertip_pos_rel_palm.reshape(self.num_envs, -1),
            "keypoints_rel_palm": keypoints_rel_palm.reshape(self.num_envs, -1),
            "keypoints_rel_goal": keypoints_rel_goal.reshape(self.num_envs, -1),
            "object_scales": object_scales_t,
        }
        obs = torch.cat([obs_dict[key] for key in OBS_LIST], dim=-1)
        return obs.detach().cpu().numpy()

    def _read_camera_outputs(self) -> dict[str, torch.Tensor]:
        if self.camera is None:
            raise RuntimeError("Camera is disabled for this environment")
        outputs = self.camera.data.output
        available = {key: value for key, value in outputs.items() if value is not None}
        if not available:
            raise RuntimeError("Camera produced no outputs")
        return available

    def _preprocess_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.float()
        if self.depth_preprocess_mode == "clip_divide":
            return torch.clamp(depth, self.depth_min_m, self.depth_max_m) / self.depth_max_m
        valid = (depth >= self.depth_min_m) & (depth <= self.depth_max_m)
        if self.depth_preprocess_mode == "metric":
            return torch.where(valid, depth, torch.zeros_like(depth))
        normalized = (depth - self.depth_min_m) / (self.depth_max_m - self.depth_min_m)
        return torch.where(valid, normalized, torch.zeros_like(depth))

    def build_student_obs(self, sim_state: SimState, camera_modality: str | None = None) -> dict[str, torch.Tensor]:
        modality = camera_modality or self.camera_modality
        outputs = self._read_camera_outputs()
        image_dict: dict[str, torch.Tensor] = {}
        if modality in ("depth", "rgbd"):
            depth = outputs.get("distance_to_image_plane")
            if depth is None:
                raise RuntimeError(f"Depth output missing. Available camera outputs: {list(outputs.keys())}")
            if depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth.permute(0, 3, 1, 2)
            elif depth.dim() == 3:
                depth = depth.unsqueeze(1)
            image_dict["depth"] = self._preprocess_depth(depth)
        if modality in ("rgb", "rgbd"):
            rgb = outputs.get("rgb")
            if rgb is None:
                raise RuntimeError(f"RGB output missing. Available camera outputs: {list(outputs.keys())}")
            rgb = rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0
            image_dict["rgb"] = rgb

        progress = (self.goal_idx.astype(np.float32) / len(self.task_spec.goals))[:, None]
        proprio = torch.from_numpy(
            np.concatenate(
                [
                    sim_state.q.astype(np.float32),
                    sim_state.qd.astype(np.float32),
                    self.prev_targets.astype(np.float32),
                    progress.astype(np.float32),
                ],
                axis=1,
            )
        ).to(self.device)
        return {
            "images": image_dict,
            "proprio": proprio,
        }

    def close(self):
        self.scene = None
        self.camera = None
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
