"""Replay InterMimic reference clips on the K1, with the arms under PD control and the box under physics.

This is a feasibility check on the references themselves, not a policy rollout: can the arms, driven only by
a PD controller with the real actuator limits, actually grasp and lift the box? Per clip, in ``--root pinned``
(the default, matching the Isaac Gym rig these references were checked with):

* **Root teleported.** The base pose and velocities are written from the reference every control step, so any
  tipping is erased immediately. The robot cannot fall and the legs never have to hold it up -- balance is
  deliberately taken out of the question so the grasp is what is being measured.
* **Joints under PD.** Only frame 0 writes joint angles. After that the joints only ever receive
  ``set_joint_position_target(next reference pose)``, held across the physics substeps like a 50 Hz policy,
  through the real K1 actuator model (derived PD gains, per-motor torque limits, torque-speed clipping, 2-8
  substep command delays) and the URDF joint limits. So the arms can lag, sag, and be blocked by the box.
* **Box under physics.** It starts at its reference frame-0 pose with the captured mass and moves only through
  gravity and contact with the robot, the floor and the table.
* Clips whose reference box starts off the ground (it rests on a table in the source capture) get a static
  support slab under it, sized from the box mesh's lowest vertex at the frame-0 rotation.

``--root free`` writes nothing after frame 0, which is a stability check instead: the robot falls within about
a second, because a PD controller tracking reference joint *states* has no balance feedback.

Each clip is rendered to its own mp4 with reference-vs-simulated hand contact, box height and arm tracking
error burnt in, and the run writes a metrics table.

The .pt layout, the DOF alias table and the camera/mp4 plumbing are the same as
:file:`pt_to_npz_with_offline_video.py`; see that script's docstring for the column map.

.. code-block:: bash

    # timing probe: one batch, a few frames, prints seconds/frame and the projected total
    python scripts/replay_pd_references.py --headless --refs "booster_assets/references/tracker/*.pt" --probe 20

    # one group of clips
    python scripts/replay_pd_references.py --headless --refs "booster_assets/references/tracker/*.pt" \
        --out logs/pd_replay/tracker

    # everything, group by group (see --help for the driver loop)
    for d in booster_assets/references/tracker booster_assets/references/exag/*/; do
        python scripts/replay_pd_references.py --headless --refs "$d/*.pt" \
            --out "logs/pd_replay/$(basename $(dirname $d/x))" --skip_existing
    done
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PD-only physics replay of InterMimic reference clips.")
parser.add_argument("--refs", type=str, required=True, help="Glob of reference .pt files (quote it).")
parser.add_argument("--out", type=str, required=True, help="Output directory for mp4s and metrics.")
parser.add_argument("--root", choices=("pinned", "free"), default="pinned",
                    help="pinned: teleport the base from the reference every control step (balance removed)."
                         " free: write nothing after frame 0.")  # fmt: skip
parser.add_argument("--target_lead", type=int, default=1,
                    help="PD targets aim at the reference pose this many frames ahead, as a policy would.")  # fmt: skip
parser.add_argument("--fps", type=int, default=50, help="Reference/control rate; one .pt row per control step.")
parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per control step (training: 4).")
parser.add_argument("--num_dof", type=int, default=22)
parser.add_argument("--box_mass", type=float, default=0.5, help="Box mass override; the URDF declares 0.1 kg.")
parser.add_argument("--box_contact_offset", type=float, default=0.02)
parser.add_argument("--box_rest_offset", type=float, default=0.001)
parser.add_argument("--box_friction", type=float, default=None, help="Static/dynamic friction; default: leave PhysX.")
parser.add_argument("--robot_friction", type=float, default=None, help="Static/dynamic friction; default: leave PhysX.")
parser.add_argument("--support_min_height", type=float, default=0.05,
                    help="Spawn a support slab when the box's frame-0 bottom is above this height.")  # fmt: skip
parser.add_argument("--fall_drop", type=float, default=0.20,
                    help="Root height this far below the reference's counts as down.")  # fmt: skip
parser.add_argument("--fall_tilt_deg", type=float, default=60.0, help="Trunk tilt from upright that counts as down.")
parser.add_argument("--contact_threshold", type=float, default=0.1, help="Hand-box contact force threshold, N.")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--stride", type=int, default=1, help="Render every Nth control step; physics always runs at fps.")
parser.add_argument("--camera_offset", type=float, nargs=3, default=(1.9, 2.1, 0.9))
parser.add_argument("--camera_lookat_offset", type=float, nargs=3, default=(0.0, 0.0, -0.1))
parser.add_argument("--max_frames", type=int, default=None, help="Cap frames per clip (debugging).")
parser.add_argument("--probe", type=int, default=None, help="Render this many frames, print timing, then exit.")
parser.add_argument("--skip_existing", action="store_true", help="Skip clips whose mp4 already exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Camera sensors need the rendering extensions even when headless.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.objects.boxes import LARGEBOX_0539923_CFG as OBJECT_CFG
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG

# Same source DOF order as pt_to_npz_with_offline_video.py: one row per .pt joint column, listing every
# spelling the joint is known by across the InterMimic retarget model and the Booster URDF.
K1_DOF_ALIASES = [
    ("AAHead_yaw", "aahead_yaw_joint"),
    ("Head_pitch", "aahead_pitch_joint"),
    ("ALeft_Shoulder_Pitch", "aaleft_shoulder_pitch_joint", "left_shoulder_pitch_joint"),
    ("Left_Shoulder_Roll", "left_shoulder_roll_joint"),
    ("Left_Elbow_Pitch", "left_elbow_pitch_joint"),
    ("Left_Elbow_Yaw", "left_elbow_yaw_joint"),
    ("ARight_Shoulder_Pitch", "aaright_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
    ("Right_Shoulder_Roll", "right_shoulder_roll_joint"),
    ("Right_Elbow_Pitch", "right_elbow_pitch_joint"),
    ("Right_Elbow_Yaw", "right_elbow_yaw_joint"),
    ("Left_Hip_Pitch", "left_hip_pitch_joint"),
    ("Left_Hip_Roll", "left_hip_roll_joint"),
    ("Left_Hip_Yaw", "left_hip_yaw_joint"),
    ("Left_Knee_Pitch", "left_knee_pitch_joint"),
    ("Left_Ankle_Pitch", "left_ankle_pitch_joint"),
    ("Left_Ankle_Roll", "left_ankle_roll_joint"),
    ("Right_Hip_Pitch", "right_hip_pitch_joint"),
    ("Right_Hip_Roll", "right_hip_roll_joint"),
    ("Right_Hip_Yaw", "right_hip_yaw_joint"),
    ("Right_Knee_Pitch", "right_knee_pitch_joint"),
    ("Right_Ankle_Pitch", "right_ankle_pitch_joint"),
    ("Right_Ankle_Roll", "right_ankle_roll_joint"),
]

# (source contact column, K1 body name) for the two hands, as in the bake script.
CONTACT_CHANNELS = [(6, "left_hand_link"), (11, "right_hand_link")]

BOX_LINK = "largebox_0539923_link"
BOX_MESH = f"{BOOSTER_ASSETS_DIR}/motions/K1/largebox_0539923/largebox_0539923.obj"
# Support slab for clips whose box starts on a table; kinematic so it never falls but can still be teleported.
# Only slightly wider than the box footprint (25 x 25 cm): a larger slab reaches the robot's feet, since the
# box starts ~0.3 m from the root, and the robot ends up standing on the table.
SUPPORT_SIZE = (0.36, 0.36, 0.5)
SUPPORT_PARKED_Z = -10.0


@configclass
class PdReplaySceneCfg(InteractiveSceneCfg):
    """One env per reference clip: robot, box, table support, hand contact sensors and a render camera."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    object: RigidObjectCfg = OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")
    support: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=SUPPORT_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.3, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SUPPORT_PARKED_Z)),
    )
    # Filtered contact is one sensor body to many targets, so each hand needs its own sensor. The history
    # covers every physics substep of a control step, so the peak force is not missed between renders.
    left_hand_box = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/" + BOX_LINK],
        history_length=args_cli.substeps,
    )
    right_hand_box = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/" + BOX_LINK],
        history_length=args_cli.substeps,
    )
    camera: CameraCfg | None = None


class Mp4Writer:
    """Small streaming MP4 writer so we never keep the full video in RAM (same as the bake script's)."""

    def __init__(self, path: str, fps: int):
        import imageio.v2 as imageio

        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None)

    def append(self, frame: np.ndarray):
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._writer.append_data(np.ascontiguousarray(frame))

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def resolve_joint_indexes(robot_joint_names: list[str], aliases: list[tuple[str, ...]]) -> list[int]:
    """Map each .pt DOF column onto the spawned robot's joint order; first alias present wins."""
    indexes = []
    for row in aliases:
        for name in row:
            if name in robot_joint_names:
                indexes.append(robot_joint_names.index(name))
                break
        else:
            raise RuntimeError(f"None of the aliases {row} exist on the spawned robot.")
    if len(set(indexes)) != len(indexes):
        raise RuntimeError("Alias table resolved two .pt columns onto the same joint.")
    return indexes


def load_clip(path: str, num_dof: int) -> dict:
    """Parse one InterMimic reference .pt (plus its sidecar json) into world-frame tensors, quats as wxyz."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(raw) or raw.ndim != 2:
        raise TypeError(f"Expected a (T, raw) tensor in {path}.")
    raw = raw.to(torch.float32)
    residual = raw.shape[1] - 26 - 2 * num_dof
    if residual <= 0 or residual % 14 != 0:
        raise ValueError(f"{path}: cannot infer body count from width {raw.shape[1]} with num_dof={num_dof}.")
    num_bodies = residual // 14
    obj = 13 + 2 * num_dof
    body_start = 26 + 2 * num_dof
    contact = raw[:, body_start + 13 * num_bodies : body_start + 14 * num_bodies]
    clip = {
        "path": path,
        "name": Path(path).stem,
        "frames": raw.shape[0],
        "root_pos": raw[:, 0:3],
        "root_quat": raw[:, 3:7][:, [3, 0, 1, 2]],  # xyzw -> wxyz
        "root_lin_vel": raw[:, 7:10],
        "root_ang_vel": raw[:, 10:13],
        "joint_pos": raw[:, 13 : 13 + num_dof],
        "joint_vel": raw[:, 13 + num_dof : 13 + 2 * num_dof],
        "object_pos": raw[:, obj : obj + 3],
        "object_quat": raw[:, obj + 3 : obj + 7][:, [3, 0, 1, 2]],
        "object_lin_vel": raw[:, obj + 7 : obj + 10],
        "object_ang_vel": raw[:, obj + 10 : obj + 13],
        "contact": contact[:, [index for index, _ in CONTACT_CHANNELS]],
    }
    for key in ("root_quat", "object_quat"):
        norms = clip[key].norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
            raise ValueError(f"{path}: {key} is not unit norm (max deviation {(norms - 1).abs().max():.4f}).")
    sidecar = Path(path).with_suffix(".json")
    clip["meta"] = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    return clip


def box_bottom_height(mesh_vertices: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> float:
    """World height of the box mesh's lowest vertex at the given pose (quat as wxyz)."""
    w, x, y, z = quat_wxyz
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    return float((mesh_vertices @ rot.T)[:, 2].min() + pos[2])


def grasp_window(clip: dict) -> tuple[int, int] | None:
    """Reference grasp interval from the sidecar json: earliest hand-on to latest hand-off."""
    hands = clip.get("meta", {}).get("hands")
    if not hands:
        return None
    on = [h["on"] for h in hands.values() if h.get("on") is not None]
    off = [h["off"] for h in hands.values() if h.get("off") is not None]
    return (min(on), max(off)) if on and off else None


def draw_overlay(img: np.ndarray, clip: dict, f: int, state: dict, threshold: float) -> np.ndarray:
    """Burn in clip name, reference vs simulated hand contact, box height and a box-height timeline."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    white, green, red, grey, yellow = (255, 255, 255), (60, 220, 60), (255, 70, 70), (150, 150, 150), (255, 210, 0)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 86), (0, 0, 0), -1)
    cv2.putText(img, f"{clip['group']}/{clip['name']}", (8, 18), font, 0.45, white, 1)
    cv2.putText(img, f"f{f}/{clip['frames']}  t={f / args_cli.fps:.2f}s", (8, 38), font, 0.45, white, 1)
    window = clip.get("window")
    if window is not None:
        inside = window[0] <= f <= window[1]
        cv2.putText(img, f"ref grasp {window[0]}-{window[1]}", (170, 38), font, 0.45,
                    yellow if inside else grey, 1)  # fmt: skip
    for i, hand in enumerate(("L", "R")):
        ref_on = state["ref_contact"][i] > 0.5
        force = state["force"][i]
        sim_on = force > threshold
        colour = green if sim_on == ref_on else red
        cv2.putText(img, f"{hand}  ref {'ON ' if ref_on else 'off'}  sim {'ON ' if sim_on else 'off'} {force:6.1f}N",
                    (8 + 300 * i, 58), font, 0.45, colour, 1)  # fmt: skip
    cv2.putText(img, f"box z {state['box_z']:.3f} (ref {state['ref_box_z']:.3f})   arm err {state['arm_err']:4.1f} deg",
                (8, 78), font, 0.45, yellow, 1)  # fmt: skip
    cv2.putText(img, f"root {args_cli.root}", (w - 130, 18), font, 0.45, grey, 1)
    if state["fallen"]:
        cv2.putText(img, "FALLEN", (w - 110, 40), font, 0.7, red, 2)

    # box height timeline: reference over the whole clip, simulated filled in so far
    x0, y0, pw, ph = 8, h - 62, w - 16, 52
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 0), -1)
    ref_z = clip["object_pos"][:, 2].numpy()
    sim_z = state["box_z_history"]
    zmax = max(float(ref_z.max()), float(np.max(sim_z)) if len(sim_z) else 0.0, 0.3) + 0.05
    xs = lambda i: x0 + int(pw * i / max(clip["frames"] - 1, 1))
    ys = lambda z: y0 + ph - int(ph * float(np.clip(z, 0.0, zmax)) / zmax)
    for i in range(1, clip["frames"]):
        cv2.line(img, (xs(i - 1), ys(ref_z[i - 1])), (xs(i), ys(ref_z[i])), yellow, 1)
    for i in range(1, len(sim_z)):
        cv2.line(img, (xs(i - 1), ys(sim_z[i - 1])), (xs(i), ys(sim_z[i])), green, 2)
    if window is not None:
        for edge in window:
            cv2.line(img, (xs(edge), y0), (xs(edge), y0 + ph), grey, 1)
    cv2.line(img, (xs(f), y0), (xs(f), y0 + ph), white, 1)
    return img


def clip_metrics(clip: dict, log: dict, threshold: float) -> dict:
    """Lift/carry/contact/fall summary for one clip."""
    box_z = np.asarray(log["box_z"])
    ref_z = clip["object_pos"][: len(box_z), 2].numpy()
    box_err = np.asarray(log["box_err"])
    root_z = np.asarray(log["root_z"])
    sim_contact = np.asarray(log["contact"]) > 0.5
    ref_contact = clip["contact"][: len(box_z)].numpy() > 0.5
    z0 = log["box_z0"]
    ref_lifted = ref_z > z0 + 0.10
    carried = ref_lifted & (np.abs(box_z - ref_z) < 0.10)
    fallen = np.asarray(log["down"])
    tilt = np.asarray(log["tilt_deg"])
    fall_frame = int(np.argmax(fallen)) if fallen.any() else None
    window = clip.get("window")
    if window is not None:
        lo, hi = window[0], min(window[1], len(box_z) - 1)
        window_contact = float(sim_contact[lo : hi + 1].any(axis=-1).mean()) if hi >= lo else 0.0
    else:
        window_contact = None
    return {
        "group": clip["group"],
        "clip": clip["name"],
        "frames": len(box_z),
        "ref_lift_m": round(float(ref_z.max() - z0), 3),
        "sim_lift_m": round(float(box_z.max() - z0), 3),
        "carried_frac": round(float(carried.sum() / max(ref_lifted.sum(), 1)), 3),
        "box_err_mean_m": round(float(box_err.mean()), 3),
        "box_err_final_m": round(float(box_err[-1]), 3),
        "contact_recall": round(float((sim_contact & ref_contact).sum() / max(ref_contact.sum(), 1)), 3),
        "contact_precision": round(float((sim_contact & ref_contact).sum() / max(sim_contact.sum(), 1)), 3),
        "grasp_window_contact_frac": None if window_contact is None else round(window_contact, 3),
        # Only meaningful with a free base; pinned runs report None rather than a number that means nothing.
        "fell": None if args_cli.root == "pinned" else fall_frame is not None,
        "fall_time_s": None if fall_frame is None else round(fall_frame / args_cli.fps, 2),
        "arm_err_mean_deg": round(float(np.mean(log["arm_err_deg"])), 2),
        "arm_err_max_deg": round(float(np.max(log["arm_err_deg"])), 2),
        "min_root_z": round(float(root_z.min()), 3),
        "max_tilt_deg": round(float(tilt.max()), 1),
        "root_mode": args_cli.root,
    }


def run(sim: sim_utils.SimulationContext, scene: InteractiveScene, clips: list[dict]) -> list[dict]:
    robot: Articulation = scene["robot"]
    box: RigidObject = scene["object"]
    support: RigidObject = scene["support"]
    camera: Camera = scene["camera"]
    sensors: list[ContactSensor] = [scene["left_hand_box"], scene["right_hand_box"]]
    device = sim.device
    num_envs = len(clips)
    origins = scene.env_origins
    ids = torch.arange(num_envs)

    # ---- box physics: captured mass, and the collider offsets the tasks train with ----
    view = box.root_physx_view
    mass0 = view.get_masses().clone()
    view.set_masses(torch.full_like(mass0, args_cli.box_mass), ids)
    view.set_inertias(view.get_inertias() * (args_cli.box_mass / mass0.view(-1, 1)), ids)
    view.set_contact_offsets(torch.full_like(view.get_contact_offsets(), args_cli.box_contact_offset), ids)
    view.set_rest_offsets(torch.full_like(view.get_rest_offsets(), args_cli.box_rest_offset), ids)
    for asset, friction in ((box, args_cli.box_friction), (robot, args_cli.robot_friction)):
        if friction is not None:
            props = asset.root_physx_view.get_material_properties()
            props[..., 0] = friction
            props[..., 1] = friction
            asset.root_physx_view.set_material_properties(props, ids)
    print(
        f"[PHYS] box mass {view.get_masses()[0].item():.3f} kg, contact/rest offset"
        f" {view.get_contact_offsets()[0].item():.3f}/{view.get_rest_offsets()[0].item():.4f} m,"
        f" box mu {args_cli.box_friction or 'physx default'}, robot mu {args_cli.robot_friction or 'physx default'}",
        flush=True,
    )

    joint_indexes = resolve_joint_indexes(list(robot.joint_names), K1_DOF_ALIASES)
    mesh_vertices = np.asarray(trimesh.load(BOX_MESH, force="mesh").vertices)

    # ---- reference targets, padded to the longest clip by holding the last frame ----
    max_frames = max(clip["frames"] for clip in clips)
    targets = torch.zeros(num_envs, max_frames, robot.num_joints, device=device)
    ref_root = torch.zeros(num_envs, max_frames, 13, device=device)  # pos, quat, lin vel, ang vel
    for env, clip in enumerate(clips):
        pad = max_frames - clip["frames"]
        hold = lambda key, columns=None: torch.cat(
            [
                clip[key] if columns is None else clip[key][:, columns],
                (clip[key] if columns is None else clip[key][:, columns])[-1:].expand(pad, -1),
            ]
        ).to(device)
        targets[env] = robot.data.default_joint_pos[env].unsqueeze(0).repeat(max_frames, 1)
        targets[env][:, joint_indexes] = hold("joint_pos")
        ref_root[env, :, :3] = hold("root_pos") + origins[env]
        ref_root[env, :, 3:7] = hold("root_quat")
        ref_root[env, :, 7:10] = hold("root_lin_vel")
        ref_root[env, :, 10:] = hold("root_ang_vel")
    ref_root_z = ref_root[:, :, 2] - origins[:, 2:3]
    # Arm joints only: with the root pinned, arm tracking error is the quantity that decides the grasp.
    arm_indexes = [
        i for i, name in enumerate(robot.joint_names)
        if any(part in name for part in ("Shoulder", "Elbow"))
    ]  # fmt: skip

    # ---- frame 0: the only time joint angles are written; with --root pinned the base is rewritten each step ----
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    box_state = box.data.default_root_state.clone()
    support_state = support.data.default_root_state.clone()
    support_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    for env, clip in enumerate(clips):
        root_state[env, :3] = clip["root_pos"][0].to(device) + origins[env]
        root_state[env, 3:7] = clip["root_quat"][0].to(device)
        root_state[env, 7:10] = clip["root_lin_vel"][0].to(device)
        root_state[env, 10:] = clip["root_ang_vel"][0].to(device)
        joint_pos[env, joint_indexes] = clip["joint_pos"][0].to(device)
        joint_vel[env, joint_indexes] = clip["joint_vel"][0].to(device)
        box_state[env, :3] = clip["object_pos"][0].to(device) + origins[env]
        box_state[env, 3:7] = clip["object_quat"][0].to(device)
        box_state[env, 7:10] = clip["object_lin_vel"][0].to(device)
        box_state[env, 10:] = clip["object_ang_vel"][0].to(device)
        # A box that starts off the ground is resting on a table in the source capture; stand in a slab whose
        # top face meets its lowest vertex. Otherwise park the slab far below the floor, where it is inert.
        bottom = box_bottom_height(
            mesh_vertices, clip["object_pos"][0].numpy(), clip["object_quat"][0].numpy()
        )
        clip["box_bottom"] = bottom
        clip["supported"] = bottom > args_cli.support_min_height
        if clip["supported"]:
            support_state[env, :3] = torch.tensor(
                [
                    clip["object_pos"][0, 0].item() + origins[env, 0].item(),
                    clip["object_pos"][0, 1].item() + origins[env, 1].item(),
                    bottom - 0.5 * SUPPORT_SIZE[2],
                ],
                device=device,
            )
        else:
            support_state[env, :3] = torch.tensor(
                [origins[env, 0].item(), origins[env, 1].item(), SUPPORT_PARKED_Z], device=device
            )
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(targets[:, 0])
    robot.set_joint_velocity_target(torch.zeros_like(joint_pos))
    box.write_root_state_to_sim(box_state)
    support.write_root_state_to_sim(support_state)
    robot.reset()  # clear the actuator delay buffers so every clip starts the same way
    box.reset()
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)
    for clip in clips:
        print(
            f"[CLIP] {clip['group']}/{clip['name']}: {clip['frames']} frames, box bottom"
            f" {clip['box_bottom']:+.3f} m -> {'table support' if clip['supported'] else 'on floor'}",
            flush=True,
        )

    # ---- per-clip logs, writers, camera ----
    logs = []
    writers = []
    for env, clip in enumerate(clips):
        clip["window"] = grasp_window(clip)
        logs.append(
            {
                "box_z": [],
                "ref_box_z": [],
                "box_err": [],
                "root_z": [],
                "tilt_deg": [],
                "arm_err_deg": [],
                "down": [],
                "contact": [],
                "box_z0": box.data.root_pos_w[env, 2].item(),
            }
        )
        path = os.path.join(args_cli.out, clip["group"], f"{clip['name']}.mp4")
        writers.append(Mp4Writer(path, max(args_cli.fps // args_cli.stride, 1)))
        clip["video"] = path
    camera_offset = torch.tensor(args_cli.camera_offset, device=device).unsqueeze(0)
    lookat_offset = torch.tensor(args_cli.camera_lookat_offset, device=device).unsqueeze(0)
    # Once a robot is down the root tumbles, so the view follows the last upright position instead.
    camera_anchor = robot.data.root_pos_w.clone()
    fallen = torch.zeros(num_envs, dtype=torch.bool, device=device)

    total_frames = max_frames if args_cli.max_frames is None else min(max_frames, args_cli.max_frames)
    if args_cli.probe is not None:
        total_frames = min(total_frames, args_cli.probe)
    render_started = None
    rendered = 0
    for f in range(total_frames):
        force_peak = torch.zeros(num_envs, 2, device=device)
        # Root teleported from the reference once per control step, so any tipping is erased before it grows.
        if args_cli.root == "pinned":
            robot.write_root_state_to_sim(ref_root[:, f])
        target = targets[:, min(f + args_cli.target_lead, max_frames - 1)]
        for _ in range(args_cli.substeps):
            # Joints only ever get a zero-order-held position target; no joint state is written after frame 0.
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            for i, sensor in enumerate(sensors):
                history = sensor.data.force_matrix_w_history  # (N, H, 1, 1, 3), newest first
                peak = history[:, : args_cli.substeps].reshape(num_envs, -1, 3).norm(dim=-1).amax(dim=1)
                force_peak[:, i] = torch.maximum(force_peak[:, i], peak)

        box_pos = box.data.root_pos_w
        root_pos = robot.data.root_pos_w
        # A deep pick-up squat drops the root to ~0.37 m in these references, so height alone cannot call a
        # fall: count it as down only when the trunk tips over or the root sinks well below the reference's.
        upright = -robot.data.projected_gravity_b[:, 2]  # 1 upright, 0 horizontal
        tilt_deg = torch.rad2deg(torch.arccos(upright.clamp(-1.0, 1.0)))
        # How far the arms actually are from the pose they were told to hold: PD lag, sag, or the box blocking.
        arm_err_deg = torch.rad2deg(
            (robot.data.joint_pos[:, arm_indexes] - targets[:, f][:, arm_indexes]).abs().mean(dim=-1)
        )
        if args_cli.root == "pinned":
            # The base is teleported from the reference, so tilt and height are the reference's own -- a deep
            # pick-up bend reads as 90 degrees. There is no fall to detect.
            newly_down = torch.zeros(num_envs, dtype=torch.bool, device=device)
        else:
            newly_down = (tilt_deg > args_cli.fall_tilt_deg) | (root_pos[:, 2] < ref_root_z[:, f] - args_cli.fall_drop)
        camera_anchor = torch.where(newly_down.unsqueeze(-1), camera_anchor, root_pos)
        fallen |= newly_down

        for env, clip in enumerate(clips):
            if f >= clip["frames"]:
                continue
            ref_pos = clip["object_pos"][f].to(device) + origins[env]
            logs[env]["box_z"].append(box_pos[env, 2].item())
            logs[env]["ref_box_z"].append(ref_pos[2].item())
            logs[env]["box_err"].append((box_pos[env] - ref_pos).norm().item())
            logs[env]["root_z"].append(root_pos[env, 2].item())
            logs[env]["tilt_deg"].append(tilt_deg[env].item())
            logs[env]["arm_err_deg"].append(arm_err_deg[env].item())
            logs[env]["down"].append(bool(newly_down[env].item()))
            logs[env]["contact"].append((force_peak[env] > args_cli.contact_threshold).float().tolist())

        if f % args_cli.stride:
            continue
        if render_started is None:
            render_started = time.perf_counter()
        camera.set_world_poses_from_view(camera_anchor + camera_offset, camera_anchor + lookat_offset)
        sim.render()
        scene.update(sim.get_physics_dt())
        rgb = camera.data.output["rgb"].detach().cpu().numpy()
        rendered += 1
        for env, clip in enumerate(clips):
            if f >= clip["frames"]:
                continue
            state = {
                "ref_contact": clip["contact"][f].tolist(),
                "force": force_peak[env].tolist(),
                "box_z": logs[env]["box_z"][-1],
                "ref_box_z": logs[env]["ref_box_z"][-1],
                "arm_err": logs[env]["arm_err_deg"][-1],
                "fallen": bool(fallen[env].item()),
                "box_z_history": logs[env]["box_z"],
            }
            writers[env].append(draw_overlay(rgb[env], clip, f, state, args_cli.contact_threshold))
        if f % 25 == 0:
            elapsed = time.perf_counter() - render_started
            print(f"[STEP] frame {f}/{total_frames}  {elapsed / max(rendered, 1):.2f} s/rendered frame", flush=True)

    for writer in writers:
        writer.close()
    if render_started is not None:
        per_frame = (time.perf_counter() - render_started) / max(rendered, 1)
        print(
            f"[TIMING] {rendered} rendered frames of {num_envs} envs at {args_cli.width}x{args_cli.height}:"
            f" {per_frame:.3f} s/frame -> {per_frame * max_frames / 60:.1f} min for this group,"
            f" {per_frame * max_frames * 10 / 60:.1f} min projected for all 10 groups",
            flush=True,
        )
    return [clip_metrics(clip, log, args_cli.contact_threshold) for clip, log in zip(clips, logs)]


def main():
    paths = sorted(glob.glob(args_cli.refs, recursive=True))
    if not paths:
        raise SystemExit(f"No reference .pt files matched {args_cli.refs!r}.")
    clips = []
    for path in paths:
        clip = load_clip(path, args_cli.num_dof)
        clip["group"] = Path(path).parent.name
        video = os.path.join(args_cli.out, clip["group"], f"{clip['name']}.mp4")
        if args_cli.skip_existing and os.path.isfile(video):
            print(f"[SKIP] {video} exists", flush=True)
            continue
        clips.append(clip)
    if not clips:
        print("[INFO] nothing to do", flush=True)
        return
    print(f"[INFO] {len(clips)} clips, one env each, out -> {args_cli.out}", flush=True)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / (args_cli.fps * args_cli.substeps)  # 0.005 s, as in training
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = PdReplaySceneCfg(num_envs=len(clips), env_spacing=10.0)
    scene_cfg.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RenderCamera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    metrics = run(sim, scene, clips)

    os.makedirs(args_cli.out, exist_ok=True)
    # Tagged by group so running one group at a time into a shared --out does not overwrite the last run's.
    tag = "_".join(sorted({clip["group"] for clip in clips}))
    with open(os.path.join(args_cli.out, f"metrics_{tag}.json"), "w") as handle:
        json.dump(metrics, handle, indent=1)
    with open(os.path.join(args_cli.out, f"metrics_{tag}.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    for row in sorted(metrics, key=lambda r: -r["carried_frac"]):
        print("[METRIC]", json.dumps(row), flush=True)
    print(f"[INFO] videos and metrics in {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
    print("[INFO]: Closing Isaac Sim...", flush=True)
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
