"""Run a trained beyond_mimic tracking policy on one clip with the real box in the scene.

The tracker is trained without any object. This asks whether, once the box is placed at its captured frame-0
pose (on a table slab when it starts off the ground) and fully simulated, simply tracking the reference is
enough to pick it up. The policy sees exactly its training observations; the box is invisible to it.

Usage:
    ~/env_isaaclab/bin/python scripts/eval_tracker_with_box.py --headless \
        --policy logs/rsl_rl/.../exported/<name>.pt --clip sub12_largebox_014
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy", required=True, help="TorchScript policy exported by play.py.")
parser.add_argument("--clip", default="sub12_largebox_014", help="Clip name under motions/K1/tracker/npz.")
parser.add_argument("--out", default="logs/eval_box")
parser.add_argument("--tag", default=None, help="Suffix for the output files (default: policy file stem).")
parser.add_argument("--box_mass", type=float, default=0.5)
parser.add_argument("--friction", type=float, default=None,
                    help="Static/dynamic friction for robot and box; default: the scene's material (1.0, multiply).")  # fmt: skip
parser.add_argument("--support_min_height", type=float, default=0.05)
parser.add_argument("--support_xy", type=float, default=0.36, help="Table slab width/depth, m.")
parser.add_argument("--box_shift", type=float, default=0.0,
                    help="Move the box (and slab) this far toward the robot root along the root->box line, m.")  # fmt: skip
parser.add_argument("--box_offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"),
                    help="World-frame xy offset added to the box (and slab) start, m; e.g. the policy's root drift.")  # fmt: skip
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--contact_threshold", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg, ContactSensorCfg

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.objects.boxes import LARGEBOX_0539923_CFG
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.largebox_tracker.env_cfg import (
    PlayFlatWoStateEstimationEnvCfg,
)

BOX_LINK = "largebox_0539923_link"
BOX_MESH = f"{BOOSTER_ASSETS_DIR}/motions/K1/largebox_0539923/largebox_0539923.obj"
# Same slab as replay_pd_references.py: barely wider than the box so the robot cannot stand on it.
SUPPORT_HEIGHT = 0.5


def box_bottom_height(mesh_vertices: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    return float((mesh_vertices @ rot.T)[:, 2].min() + pos[2])


def build_cfg(motion_file: str, box_pos, box_quat, support_pos) -> PlayFlatWoStateEstimationEnvCfg:
    cfg = PlayFlatWoStateEstimationEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    cfg.commands.motion.motion_file = motion_file
    # Start exactly on the reference: no reset noise, no pushes, no DR, clean observations, no early resets.
    cfg.commands.motion.pose_range = {}
    cfg.commands.motion.velocity_range = {}
    cfg.commands.motion.joint_position_range = (0.0, 0.0)
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    cfg.observations.policy.enable_corruption = False
    cfg.events.physics_material = None
    cfg.events.add_joint_default_pos = None
    cfg.events.base_com = None
    cfg.events.push_robot = None
    for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        setattr(cfg.terminations, name, None)
    cfg.episode_length_s = 60.0

    cfg.scene.object = LARGEBOX_0539923_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(box_pos), rot=tuple(box_quat)),
    )
    cfg.scene.support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.CuboidCfg(
            size=(args_cli.support_xy, args_cli.support_xy, SUPPORT_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.3, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(support_pos)),
    )
    for side in ("left", "right"):
        setattr(
            cfg.scene,
            f"{side}_hand_box",
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_hand_link",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/" + BOX_LINK],
                history_length=cfg.decimation,
            ),
        )
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RenderCamera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    return cfg


def draw_overlay(img, f, n, box_z, ref_z, force, ref_contact, root_z, z_hist, ref_hist):
    img = np.ascontiguousarray(img[..., :3]).copy()
    white, green, yellow, red = (255, 255, 255), (80, 220, 80), (240, 220, 60), (240, 80, 80)
    lines = [
        (f"{args_cli.clip}  {args_cli.tag}   frame {f}/{n}", white),
        (f"box z  sim {box_z:.3f}  ref {ref_z:.3f}  (err {box_z - ref_z:+.3f})", white),
        (f"root z {root_z:.3f}", white),
    ]
    for i, side in enumerate(("L", "R")):
        on = force[i] > args_cli.contact_threshold
        lines.append((f"{side} hand  sim {'CONTACT' if on else '-'} {force[i]:5.1f} N   ref {int(ref_contact[i])}",
                      green if on else white))  # fmt: skip
    for i, (text, color) in enumerate(lines):
        cv2.putText(img, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
    # Box-height timeline: reference yellow, sim green.
    h, w = img.shape[:2]
    x0, y0, pw, ph = 12, h - 110, w - 24, 96
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (30, 30, 30), -1)
    lo, hi = min(min(ref_hist), min(z_hist)) - 0.05, max(max(ref_hist), max(z_hist)) + 0.05
    xs = lambda i: int(x0 + pw * i / max(n - 1, 1))  # noqa: E731
    ys = lambda z: int(y0 + ph - ph * (z - lo) / (hi - lo))  # noqa: E731
    for hist, color in ((ref_hist, yellow), (z_hist, green)):
        pts = np.array([[xs(i), ys(z)] for i, z in enumerate(hist)], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, color, 2)
    cv2.line(img, (xs(f), y0), (xs(f), y0 + ph), red, 1)
    return img


def main():
    motion_file = f"{BOOSTER_ASSETS_DIR}/motions/K1/tracker/npz/{args_cli.clip}_hold.npz"
    ref = np.load(motion_file)
    n = ref["joint_pos"].shape[0]
    box_pos0, box_quat0 = ref["object_pos_w"][0].copy(), ref["object_quat_w"][0]
    box_pos0[:2] += np.asarray(args_cli.box_offset)
    if args_cli.box_shift:
        toward = ref["body_pos_w"][0, 0, :2] - box_pos0[:2]
        box_pos0[:2] += args_cli.box_shift * toward / np.linalg.norm(toward)
        print(f"[INFO] box shifted {args_cli.box_shift * 100:.0f} cm toward the robot:"
              f" {np.linalg.norm(toward):.3f} -> {np.linalg.norm(ref['body_pos_w'][0, 0, :2] - box_pos0[:2]):.3f} m"
              " root-to-box (xy)", flush=True)  # fmt: skip
    mesh_vertices = np.asarray(trimesh.load(BOX_MESH, force="mesh").vertices)
    bottom = box_bottom_height(mesh_vertices, box_pos0, box_quat0)
    supported = bottom > args_cli.support_min_height
    support_pos = (box_pos0[0], box_pos0[1], bottom - 0.5 * SUPPORT_HEIGHT) if supported else (0.0, 0.0, -10.0)
    print(f"[INFO] {args_cli.clip}: {n} frames, box bottom {bottom:+.3f} m ->"
          f" {'table slab' if supported else 'floor'}", flush=True)  # fmt: skip

    env = ManagerBasedRLEnv(build_cfg(motion_file, box_pos0, box_quat0, support_pos))
    device = env.device
    robot, box, support, camera = (env.scene[k] for k in ("robot", "object", "support", "camera"))
    hand_ids = [robot.body_names.index(b) for b in ("left_hand_link", "right_hand_link")]
    ref_hand_ids = [list(ref["body_names"]).index(b) for b in ("left_hand_link", "right_hand_link")]
    hands = [env.scene["left_hand_box"], env.scene["right_hand_box"]]

    view = box.root_physx_view
    ids = torch.arange(view.count, device="cpu")
    mass0 = view.get_masses().clone()
    view.set_masses(torch.full_like(mass0, args_cli.box_mass), ids)
    view.set_inertias(view.get_inertias() * (args_cli.box_mass / mass0.view(-1, 1)), ids)
    view.set_contact_offsets(torch.full_like(view.get_contact_offsets(), 0.02), ids)
    view.set_rest_offsets(torch.full_like(view.get_rest_offsets(), 0.001), ids)
    if args_cli.friction is not None:
        for asset in (box, robot):
            props = asset.root_physx_view.get_material_properties()
            props[..., 0] = args_cli.friction
            props[..., 1] = args_cli.friction
            asset.root_physx_view.set_material_properties(props, torch.arange(asset.num_instances, device="cpu"))

    policy = torch.jit.load(args_cli.policy, map_location=device).eval()
    obs, _ = env.reset()
    origin = env.scene.env_origins[0]
    # Put the box (and slab) back on its captured pose after the reset placed the robot on frame 0.
    box_state = box.data.default_root_state.clone()
    box_state[:, :3] = torch.tensor(box_pos0, device=device, dtype=torch.float32) + origin
    box_state[:, 3:7] = torch.tensor(box_quat0, device=device, dtype=torch.float32)
    box_state[:, 7:] = 0.0
    box.write_root_state_to_sim(box_state)
    support_state = support.data.default_root_state.clone()
    support_state[:, :3] = torch.tensor(support_pos, device=device, dtype=torch.float32) + origin
    support_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    support.write_root_state_to_sim(support_state)

    # Static side-on camera: perpendicular to the robot->box line, looking at their midpoint.
    root0 = ref["body_pos_w"][0, 0]
    mid = 0.5 * (root0[:2] + box_pos0[:2])
    d = box_pos0[:2] - root0[:2]
    d = d / max(np.linalg.norm(d), 1e-6)
    perp = np.array([-d[1], d[0]])
    target = torch.tensor([[mid[0], mid[1], 0.45]], device=device, dtype=torch.float32) + origin
    eye = target + torch.tensor([[perp[0] * 2.4, perp[1] * 2.4, 0.6]], device=device, dtype=torch.float32)

    tag = args_cli.tag or Path(args_cli.policy).stem
    os.makedirs(args_cli.out, exist_ok=True)
    stem = os.path.join(args_cli.out, f"{args_cli.clip}_{tag}")
    import imageio.v2 as imageio

    writer = imageio.get_writer(stem + ".mp4", fps=50, codec="libx264", quality=8, macro_block_size=None)
    ref_shift = box_pos0 - ref["object_pos_w"][0]  # compare against the reference moved by the same shift
    log = {"box_z": [], "ref_box_z": [], "box_err": [], "root_z": [], "contact": [], "root_xy_err": [], "root_xy_off": []}
    z0 = float(box_pos0[2])
    with torch.inference_mode():
        for f in range(n - 1):
            camera.set_world_poses_from_view(eye, target)
            obs, _, _, _, _ = env.step(policy(obs["policy"]))
            frame = f + 1  # the command advanced one frame this step
            box_p = (box.data.root_pos_w[0] - origin).cpu().numpy()
            ref_p = ref["object_pos_w"][frame] + ref_shift
            root_p = (robot.data.root_pos_w[0] - origin).cpu().numpy()
            force = [
                float(s.data.force_matrix_w_history[0].norm(dim=-1).amax().item()) for s in hands
            ]  # peak over this step's physics substeps
            log["box_z"].append(float(box_p[2]))
            log.setdefault("box_xy", []).append(box_p[:2].tolist())
            log["ref_box_z"].append(float(ref_p[2]))
            log["box_err"].append(float(np.linalg.norm(box_p - ref_p)))
            log["root_z"].append(float(root_p[2]))
            hand_p = (robot.data.body_pos_w[0, hand_ids] - origin).cpu().numpy()
            log.setdefault("hand_off", []).append((hand_p - ref["body_pos_w"][frame, ref_hand_ids]).tolist())
            log["root_xy_off"].append((root_p[:2] - ref["body_pos_w"][frame, 0, :2]).tolist())
            log["root_xy_err"].append(float(np.linalg.norm(root_p[:2] - ref["body_pos_w"][frame, 0, :2])))
            log["contact"].append([x > args_cli.contact_threshold for x in force])
            rgb = camera.data.output["rgb"][0].cpu().numpy()
            writer.append_data(draw_overlay(
                rgb, frame, n, box_p[2], ref_p[2], force, ref["contact"][frame], root_p[2],
                log["box_z"], ref["object_pos_w"][1:, 2] + ref_shift[2],
            ))  # fmt: skip
    writer.close()

    box_z, ref_z = np.array(log["box_z"]), np.array(log["ref_box_z"])
    contact = np.array(log["contact"])
    ref_lifted = ref_z > z0 + 0.10
    # Where the policy's root has drifted to by the time the reference first touches the box: adding this to
    # --box_offset puts the box where the drifted robot's hands will actually arrive.
    ref_touch = np.flatnonzero(ref["contact"][1:].any(axis=1))
    touch = int(ref_touch[0]) if len(ref_touch) else 0
    # Mean sim-minus-reference hand position over the frames where the reference is touching the box, per hand
    # and averaged: the xy part is where to move the box so it meets the hands the policy actually produces.
    hand_off = np.array(log["hand_off"])  # [frames, 2 hands, 3]
    touching = ref["contact"][1:n].any(axis=1)
    hand_off_touch = hand_off[touching].mean(axis=0) if touching.any() else np.zeros((2, 3))
    metrics = {
        "clip": args_cli.clip,
        "policy": args_cli.policy,
        "box_shift_m": args_cli.box_shift,
        "box_offset_m": list(args_cli.box_offset),
        "box_start_z": z0,
        "supported": bool(supported),
        "ref_max_lift_m": float(ref_z.max() - z0),
        "sim_max_lift_m": float(box_z.max() - z0),
        "sim_final_lift_m": float(box_z[-1] - z0),
        "carried_frac": float((np.abs(box_z - ref_z)[ref_lifted] < 0.10).mean()) if ref_lifted.any() else None,
        "box_err_mean_m": float(np.mean(log["box_err"])),
        "box_err_final_m": float(log["box_err"][-1]),
        "left_contact_frac": float(contact[:, 0].mean()),
        "right_contact_frac": float(contact[:, 1].mean()),
        "both_contact_frac": float(contact.all(axis=1).mean()),
        "root_xy_err_final_m": float(log["root_xy_err"][-1]),
        "hand_offset_during_ref_contact_L": [round(float(v), 3) for v in hand_off_touch[0]],
        "hand_offset_during_ref_contact_R": [round(float(v), 3) for v in hand_off_touch[1]],
        "hand_offset_during_ref_contact_mean": [round(float(v), 3) for v in hand_off_touch.mean(axis=0)],
        "ref_first_contact_frame": touch + 1,
        "root_xy_offset_at_first_contact": [round(v, 3) for v in log["root_xy_off"][touch]],
        "box_xy_moved_m": float(np.linalg.norm(box_z.size and (np.array(log["box_xy"])[-1] - box_pos0[:2]))),
        "box_xy_final": [round(v, 3) for v in log["box_xy"][-1]],
        "ref_box_xy_final": [round(float(v), 3) for v in (ref["object_pos_w"][-1, :2] + ref_shift[:2])],
        "box_yaw_final_deg": float(np.degrees(np.arctan2(*(lambda q: (2 * (q[0] * q[3] + q[1] * q[2]),
                                   1 - 2 * (q[2] ** 2 + q[3] ** 2)))(box.data.root_quat_w[0].cpu().numpy())))),
        "ref_box_yaw_final_deg": float(np.degrees(np.arctan2(*(lambda q: (2 * (q[0] * q[3] + q[1] * q[2]),
                                       1 - 2 * (q[2] ** 2 + q[3] ** 2)))(ref["object_quat_w"][-1])))),
        "root_yaw_err_final_deg": float(np.degrees(np.arctan2(*(lambda q: (2 * (q[0] * q[3] + q[1] * q[2]),
                                        1 - 2 * (q[2] ** 2 + q[3] ** 2)))(robot.data.root_quat_w[0].cpu().numpy())))
                                        - np.degrees(np.arctan2(*(lambda q: (2 * (q[0] * q[3] + q[1] * q[2]),
                                        1 - 2 * (q[2] ** 2 + q[3] ** 2)))(ref["body_quat_w"][-1, 0])))),
        "ref_box_xy_moved_m": float(np.linalg.norm(ref["object_pos_w"][-1, :2] - ref["object_pos_w"][0, :2])),
        "min_root_z": float(min(log["root_z"])),
    }
    with open(stem + ".json", "w") as handle:
        json.dump(metrics, handle, indent=1)
    print("[METRIC]", json.dumps(metrics, indent=1), flush=True)
    print(f"[INFO] video -> {stem}.mp4", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
