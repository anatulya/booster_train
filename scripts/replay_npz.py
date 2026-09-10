"""Replay K1 robot motions from an NPZ file.

The motion is written directly into the articulation kinematically:
    reference pose -> robot articulation -> render

No physics step is performed, so the rendered motion is exactly the reference
motion mapped onto the K1 meshes.

Examples
--------

Interactive viewer:

    python replay_npz.py \
        --motion <path_to_motion.npz>

Headless MP4 recording:

    python replay_npz.py \
        --headless \
        --device cuda:0 \
        --motion <path_to_motion.npz> \
        --video output.mp4

When --video is supplied, camera rendering is enabled automatically.
"""

"""Launch Isaac Sim first."""

import argparse
import os
import pathlib

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#

parser = argparse.ArgumentParser(description="Replay converted K1 motions.")

parser.add_argument(
    "--motion",
    type=str,
    default=None,
    help="Path to the motion NPZ file.",
)

parser.add_argument(
    "--registry_name",
    type=str,
    default=None,
    help="W&B artifact name containing motion.npz.",
)

parser.add_argument(
    "--video",
    type=str,
    default=None,
    help="Optional output MP4 path.",
)

parser.add_argument(
    "--video_width",
    type=int,
    default=1280,
    help="Video width.",
)

parser.add_argument(
    "--video_height",
    type=int,
    default=720,
    help="Video height.",
)

parser.add_argument(
    "--camera_offset",
    nargs=3,
    type=float,
    default=(2.0, 2.0, 0.5),
    metavar=("X", "Y", "Z"),
    help="Camera position relative to the robot root.",
)

parser.add_argument(
    "--camera_lookat_offset",
    nargs=3,
    type=float,
    default=(0.0, 0.0, 0.25),
    metavar=("X", "Y", "Z"),
    help="Offset from the robot root used as the camera look-at point.",
)


# Isaac Lab arguments:
#   --headless
#   --device
#   --enable_cameras
#   ...
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


# Video in headless mode requires the offscreen camera pipeline.
# Do this automatically instead of requiring the caller to pass
# --enable_cameras manually.
if args_cli.video is not None:
    args_cli.enable_cameras = True


# Launch Isaac Sim.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything below this point can import Isaac Sim / Isaac Lab modules."""


import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionLoader


# -----------------------------------------------------------------------------#
# Scene configuration
# -----------------------------------------------------------------------------#


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Scene used to visualize a K1 motion."""

    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Robot
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Dedicated recording camera.
    #
    # Do NOT use /OmniverseKit_Persp for recording. That is the viewport
    # camera. Creating an actual Isaac Lab camera makes headless rendering
    # independent of the GUI/viewport.
    camera = CameraCfg(
        prim_path="/World/ReplayCamera",
        update_period=0.0,
        width=1280,
        height=720,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 1000.0),
        ),
    )

    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/"
                "PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


# -----------------------------------------------------------------------------#
# Motion loading
# -----------------------------------------------------------------------------#


def resolve_motion_file() -> str:
    """Resolve the motion file from either a local path or W&B."""

    if args_cli.motion is not None:
        path = os.path.abspath(args_cli.motion)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Motion file not found: {path}")

        return path

    if args_cli.registry_name is not None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is required when using --registry_name. "
                "Install it with: pip install wandb"
            ) from exc

        registry_name = args_cli.registry_name

        if ":" not in registry_name:
            registry_name += ":latest"

        print(f"[INFO]: Downloading W&B artifact: {registry_name}")

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        artifact_dir = pathlib.Path(artifact.download())

        motion_path = artifact_dir / "motion.npz"

        if not motion_path.is_file():
            raise FileNotFoundError(
                f"W&B artifact does not contain motion.npz: {artifact_dir}"
            )

        return str(motion_path)

    raise ValueError(
        "Either --motion or --registry_name must be provided."
    )


# -----------------------------------------------------------------------------#
# Rendering helpers
# -----------------------------------------------------------------------------#


def set_camera_pose(
    sim: sim_utils.SimulationContext,
    camera: Camera,
    root_position: torch.Tensor,
):
    """Make the camera follow the robot.

    Args:
        sim:
            Isaac Lab simulation context.

        camera:
            Dedicated Isaac Lab recording camera.

        root_position:
            Robot root position with shape (1, 3).
    """

    device = root_position.device

    camera_offset = torch.tensor(
        args_cli.camera_offset,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    lookat_offset = torch.tensor(
        args_cli.camera_lookat_offset,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    target = root_position + lookat_offset
    eye = root_position + camera_offset

    # Recording camera.
    camera.set_world_poses_from_view(
        eyes=eye,
        targets=target,
    )

    # Also update the GUI viewport when running interactively.
    if not args_cli.headless:
        sim.set_camera_view(
            eye[0].detach().cpu().tolist(),
            target[0].detach().cpu().tolist(),
        )


def get_rgb_frame(camera: Camera) -> np.ndarray | None:
    """Read an RGB frame from the Isaac Lab camera."""

    rgb = camera.data.output.get("rgb")

    if rgb is None:
        return None

    if rgb.numel() == 0:
        return None

    # Camera output:
    #     (num_cameras, H, W, 3)
    #
    # We only have one camera.
    rgb = rgb[0]

    if rgb.dtype != torch.uint8:
        rgb = rgb.clamp(0, 255).to(torch.uint8)

    return rgb.detach().cpu().numpy()


# -----------------------------------------------------------------------------#
# Replay
# -----------------------------------------------------------------------------#


@torch.inference_mode()
def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
):
    """Replay the motion and optionally record it."""

    robot: Articulation = scene["robot"]
    camera: Camera = scene["camera"]

    motion_file = resolve_motion_file()

    motion = MotionLoader(
        motion_file,
        robot.body_names,
        robot.joint_names,
        tail_len=0,
        device=str(sim.device),
    )

    motion_fps = int(np.asarray(motion.fps).reshape(-1)[0])
    num_frames = int(motion.time_step_total)

    frame_dt = 1.0 / motion_fps

    print(
        f"[INFO]: Loaded motion: {motion_file}\n"
        f"        frames : {num_frames}\n"
        f"        fps    : {motion_fps}\n"
        f"        length : {num_frames / motion_fps:.2f} s"
    )

    # -------------------------------------------------------------------------
    # Video writer
    # -------------------------------------------------------------------------

    video_writer = None

    if args_cli.video is not None:
        video_path = os.path.abspath(args_cli.video)

        os.makedirs(
            os.path.dirname(video_path) or ".",
            exist_ok=True,
        )

        video_writer = imageio.get_writer(
            video_path,
            fps=motion_fps,
            codec="libx264",
            macro_block_size=None,
        )

        print(f"[VIDEO]: Recording to {video_path}")

    # -------------------------------------------------------------------------
    # Replay
    # -------------------------------------------------------------------------

    frame_index = 0
    frames_written = 0

    # Single environment.
    time_steps = torch.zeros(
        scene.num_envs,
        dtype=torch.long,
        device=sim.device,
    )

    try:
        while simulation_app.is_running():

            # -------------------------------------------------------------
            # Select reference frame
            # -------------------------------------------------------------

            time_steps.fill_(frame_index)

            # -------------------------------------------------------------
            # Root pose
            # -------------------------------------------------------------
            #
            # MotionLoader body index 0 is the articulation root.
            #
            # IMPORTANT:
            # We intentionally write ONLY the pose here.
            #
            # This is a kinematic visualization. There is no physics step,
            # so root linear/angular velocity has no effect on what is
            # rendered.
            #
            # This also avoids any ambiguity between:
            #   root-link velocity
            #   vs.
            #   root-COM velocity
            #
            # that write_root_state_to_sim() would otherwise introduce.

            root_pose = torch.empty(
                (scene.num_envs, 7),
                dtype=motion.body_pos_w.dtype,
                device=sim.device,
            )

            root_pose[:, :3] = (
                motion.body_pos_w[time_steps, 0]
                + scene.env_origins
            )

            root_pose[:, 3:7] = motion.body_quat_w[
                time_steps, 0
            ]

            robot.write_root_pose_to_sim(root_pose)

            # -------------------------------------------------------------
            # Joint positions
            # -------------------------------------------------------------
            #
            # Again, velocity is unnecessary because physics is never
            # stepped.

            robot.write_joint_position_to_sim(
                motion.joint_pos[time_steps]
            )

            # These APIs directly write state into the simulator.
            # scene.write_data_to_sim() is therefore NOT needed here.

            # -------------------------------------------------------------
            # Camera
            # -------------------------------------------------------------

            set_camera_pose(
                sim,
                camera,
                root_pose[:, :3],
            )

            # -------------------------------------------------------------
            # Render
            # -------------------------------------------------------------
            #
            # Deliberately DO NOT call sim.step().
            #
            # sim.step():
            #     advances physics
            #
            # sim.render():
            #     updates/render the current state only

            sim.render()

            # Synchronize Isaac Lab entity/sensor buffers.
            scene.update(frame_dt)

            # -------------------------------------------------------------
            # Record
            # -------------------------------------------------------------

            if video_writer is not None:
                rgb = get_rgb_frame(camera)

                # RTX/Replicator can require a few initial renders before
                # the first valid image arrives. If that happens, keep the
                # current motion frame fixed and render again.
                if rgb is None:
                    continue

                video_writer.append_data(rgb)

                frames_written += 1

                if (
                    frames_written == 1
                    or frames_written % 50 == 0
                    or frames_written == num_frames
                ):
                    print(
                        f"[VIDEO]: {frames_written}/{num_frames} frames"
                    )

            # -------------------------------------------------------------
            # Advance reference frame
            # -------------------------------------------------------------

            frame_index += 1

            if frame_index >= num_frames:

                if video_writer is not None:
                    # For recording, replay exactly once.
                    break

                # Interactive mode loops forever.
                frame_index = 0

    finally:
        if video_writer is not None:
            video_writer.close()

            print(
                f"[INFO]: Video saved to {os.path.abspath(args_cli.video)} "
                f"({frames_written} frames)"
            )


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#


def main():
    # The physics timestep is mostly irrelevant here because we don't call
    # sim.step(), but SimulationContext still requires a simulation cfg.
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=0.02,
    )

    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )

    # Camera resolution is set through CameraCfg, not through Replicator.
    scene_cfg.camera.width = args_cli.video_width
    scene_cfg.camera.height = args_cli.video_height

    scene = InteractiveScene(scene_cfg)

    # Initializes the articulation, camera sensor, render products, etc.
    sim.reset()

    print("[INFO]: Simulation initialized.")

    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()