"""Push the hands further into the object during its grasp window, in a baked motion npz.

The tracker references have the hands resting on the object's surface, which is why tracking them produces a
graze rather than a grip. This drives each hand a further ``--depth`` toward the object centre on the frames
where the reference reports contact, solving the arm joints with a damped-least-squares IK on the hand link,
then rewrites the body poses from FK. Legs, root and object are untouched; the push-in is ramped over
``--blend`` frames at both ends of the window so joint velocities stay smooth.

    ~/env_isaaclab/bin/python scripts/deepen_grasp_npz.py --headless \
        --input  booster_assets/motions/K1/tracker/npz/<clip>_hold.npz \
        --output booster_assets/motions/K1/tracker/npz/<clip>_deep40_hold.npz --depth 0.04
"""

import argparse

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", required=True, help="Source motion npz (any baked clip, with or without a hold).")
parser.add_argument("--output", required=True)
parser.add_argument("--depth", type=float, default=0.04, help="Extra push-in toward the object centre, m.")
parser.add_argument("--blend", type=int, default=12, help="Frames to ramp the push-in in/out at the window edges.")
parser.add_argument("--iters", type=int, default=8, help="IK iterations per frame.")
parser.add_argument("--damping", type=float, default=0.05, help="Damped-least-squares lambda.")
parser.add_argument("--horizontal", action="store_true", default=True,
                    help="Push in horizontally only (default), so the hands do not sink into the object's top.")  # fmt: skip
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG

HANDS = (("left_hand_link", "Left_"), ("right_hand_link", "Right_"))
ARM_PATTERNS = ("Shoulder_Pitch", "Shoulder_Roll", "Elbow_Pitch", "Elbow_Yaw")


def main():
    src = {k: np.asarray(v) for k, v in np.load(args_cli.input).items()}
    n = src["joint_pos"].shape[0]
    contact = src["contact"] > 0.5  # [frames, 2] left/right hand
    if not contact.any():
        raise SystemExit(f"{args_cli.input} reports no contact frames; nothing to deepen.")

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 50.0))
    robot = Articulation(ROBOT_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    device = sim.device

    joint_names = list(src["joint_names"])
    body_names = list(src["body_names"])
    if joint_names != list(robot.joint_names) or body_names != list(robot.body_names):
        raise SystemExit("npz joint/body names do not match the spawned K1; was it baked with this robot?")

    joint_pos = torch.tensor(src["joint_pos"], device=device, dtype=torch.float32)
    root_pos = torch.tensor(src["body_pos_w"][:, 0], device=device, dtype=torch.float32)
    root_quat = torch.tensor(src["body_quat_w"][:, 0], device=device, dtype=torch.float32)
    obj_pos = torch.tensor(src["object_pos_w"], device=device, dtype=torch.float32)
    limits = robot.data.soft_joint_pos_limits[0]

    hand_ids = [robot.body_names.index(name) for name, _ in HANDS]
    arm_ids = [
        [i for i, name in enumerate(robot.joint_names) if name.startswith(prefix) and any(p in name for p in ARM_PATTERNS)]
        for _, prefix in HANDS
    ]  # fmt: skip
    print(f"[IK] arm joints: " + ", ".join(
        f"{HANDS[s][0]} <- {[robot.joint_names[i] for i in arm_ids[s]]}" for s in range(2)
    ), flush=True)  # fmt: skip

    # Ramp: full push-in in the middle of each hand's contact window, tapering over --blend frames at both ends.
    weight = np.zeros((n, 2))
    for side in range(2):
        idx = np.flatnonzero(contact[:, side])
        if idx.size == 0:
            continue
        w = contact[:, side].astype(float)
        ramp = np.minimum(
            np.cumsum(w) / max(args_cli.blend, 1),                       # frames since the window started
            np.cumsum(w[::-1])[::-1] / max(args_cli.blend, 1),           # frames until it ends
        )
        weight[:, side] = np.clip(ramp, 0.0, 1.0) * w

    def write(frame_joint_pos, frame):
        state = robot.data.default_root_state.clone()
        state[:, :3] = root_pos[frame]
        state[:, 3:7] = root_quat[frame]
        state[:, 7:] = 0.0
        robot.write_root_state_to_sim(state)
        robot.write_joint_state_to_sim(frame_joint_pos.unsqueeze(0), torch.zeros_like(frame_joint_pos).unsqueeze(0))
        sim.physics_sim_view.update_articulations_kinematic()
        robot.update(0.0)

    out_joint_pos = joint_pos.clone()
    moved = np.zeros((n, 2))
    for f in range(n):
        if weight[f].max() <= 0:
            continue
        q = out_joint_pos[f].clone()
        targets = {}
        write(q, f)
        for side in range(2):
            if weight[f, side] <= 0:
                continue
            hand = robot.data.body_pos_w[0, hand_ids[side]]
            direction = obj_pos[f] - hand
            if args_cli.horizontal:
                direction = direction * torch.tensor([1.0, 1.0, 0.0], device=device)
            norm = torch.linalg.norm(direction)
            if norm < 1e-6:
                continue
            targets[side] = hand + direction / norm * (args_cli.depth * float(weight[f, side]))

        for _ in range(args_cli.iters):
            write(q, f)
            jacobians = robot.root_physx_view.get_jacobians()[0]
            done = True
            for side, target in targets.items():
                err = target - robot.data.body_pos_w[0, hand_ids[side]]
                if torch.linalg.norm(err) < 1e-4:
                    continue
                done = False
                cols = [6 + i for i in arm_ids[side]]  # floating base: first 6 columns are the root
                jac = jacobians[hand_ids[side] - 1][:3][:, cols]
                lam = args_cli.damping**2 * torch.eye(3, device=device)
                dq = jac.T @ torch.linalg.solve(jac @ jac.T + lam, err)
                q[arm_ids[side]] = torch.clamp(
                    q[arm_ids[side]] + dq, limits[arm_ids[side], 0], limits[arm_ids[side], 1]
                )
            if done:
                break

        write(q, f)
        for side, target in targets.items():
            moved[f, side] = float(torch.linalg.norm(target - robot.data.body_pos_w[0, hand_ids[side]]))
        out_joint_pos[f] = q

    # Rewrite body poses from FK for every edited frame, then velocities over the edited span by finite difference.
    out = dict(src)
    out["joint_pos"] = out_joint_pos.cpu().numpy().astype(src["joint_pos"].dtype)
    body_pos = src["body_pos_w"].copy()
    body_quat = src["body_quat_w"].copy()
    edited = np.flatnonzero(weight.max(axis=1) > 0)
    for f in edited:
        write(out_joint_pos[f], int(f))
        body_pos[f] = robot.data.body_pos_w[0].cpu().numpy()
        body_quat[f] = robot.data.body_quat_w[0].cpu().numpy()
    out["body_pos_w"] = body_pos
    out["body_quat_w"] = body_quat

    lo, hi = int(edited.min()), int(edited.max())
    span = slice(max(lo - 1, 1), min(hi + 2, n - 1))
    fps = float(np.asarray(src["fps"]).flatten()[0])
    idx = np.arange(span.start, span.stop)
    out["joint_vel"] = src["joint_vel"].copy()
    out["joint_vel"][idx] = (out["joint_pos"][idx + 1] - out["joint_pos"][idx - 1]) * (fps / 2)
    out["body_lin_vel_w"] = src["body_lin_vel_w"].copy()
    out["body_lin_vel_w"][idx] = (body_pos[idx + 1] - body_pos[idx - 1]) * (fps / 2)

    np.savez(args_cli.output, **out)
    per_hand = [moved[contact[:, s], s] for s in range(2)]
    print(f"[IK] edited {len(edited)} frames (contact L {int(contact[:, 0].sum())}, R {int(contact[:, 1].sum())})")
    for side in range(2):
        dq = np.abs(out["joint_pos"] - src["joint_pos"])[:, arm_ids[side]].max()
        residual = per_hand[side].max() if per_hand[side].size else 0.0
        print(f"[IK] {HANDS[side][0]}: max arm joint change {dq:.3f} rad, worst IK residual {residual * 1000:.1f} mm")
    print(f"[IK] wrote {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
