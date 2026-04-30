"""Test camera sensor setup in Isaac Lab standalone (no InteractiveScene)."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless", "--enable_cameras"])
app_launcher = AppLauncher(args)
app = app_launcher.app

# Post-launch imports
import sys
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.sensors import Camera, CameraCfg


def _log(msg):
    sys.stderr.write(f"[CameraTest] {msg}\n")
    sys.stderr.flush()


def main():
    # Create sim
    sim_cfg = SimulationCfg(dt=1/60, render_interval=1)
    sim = SimulationContext(sim_cfg)
    _log("SimulationContext created")

    # Add a ground plane and light so there's something to see
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/DomeLight", sim_utils.DomeLightCfg(intensity=2000.0))

    # Add a simple cube
    sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.1),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    ).func("/World/Cube", sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)), translation=(0.0, 0.0, 0.5))
    _log("Scene objects created")

    # Create camera
    camera_cfg = CameraCfg(
        prim_path="/World/Camera",
        update_period=0,  # Update every step
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.0, 2.0, 2.0),
            rot=(0.7071, 0.0, 0.3536, -0.6124),
            convention="world",
        ),
    )
    camera = Camera(cfg=camera_cfg)
    _log("Camera created")

    # Reset sim to initialize everything
    sim.reset()
    _log("Sim reset done")

    # Step a few times to let renderer warm up
    for i in range(10):
        sim.step()
        camera.update(sim.get_physics_dt())

    # Capture frame
    rgb = camera.data.output["rgb"]
    _log(f"RGB data: type={type(rgb)}, shape={rgb.shape if hasattr(rgb, 'shape') else 'N/A'}")

    if rgb is not None and hasattr(rgb, 'shape') and rgb.shape[0] > 0:
        frame = rgb[0].cpu().numpy()
        _log(f"Frame: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")

        # Save as image
        import imageio
        imageio.imwrite("/tmp/isaacsim_camera_test.png", frame[:, :, :3].astype(np.uint8))
        _log("Saved /tmp/isaacsim_camera_test.png")
    else:
        _log("WARNING: No RGB data captured!")

    app.close()
    _log("Done")


if __name__ == "__main__":
    main()
