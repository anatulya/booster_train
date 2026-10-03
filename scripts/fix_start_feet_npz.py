"""Flatten the feet over the first frames of a baked HOI npz (run after append_hold_npz.py).

For frames 0..N-1 the ankle pitch/roll of each foot are re-solved (Newton, finite-difference FK) so the sole's up axis
is a blend from vertical (frame 0) to the original tilt (frame N), with weight w = 1 - smoothstep(f / N). The root is
then shifted in z so the lowest sole corner sits at (1 - w) * its original height: on the floor at frame 0, back to
the original at frame N. Body poses for those frames are recomputed by FK; joint, root and body velocities are
finite-differenced over the edited range. Everything else (object, contact, stance, later frames) is copied as is.

    python scripts/fix_start_feet_npz.py --input <hold npz> --output <fixed npz> [--frames 25]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--frames", type=int, default=25, help="Length of the blend back to the original motion.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import os
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from booster_train.assets.robots.booster import BOOSTER_K1_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=0.02))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
sim.reset()
robot = scene["robot"]
dev = sim.device
origin = scene.env_origins[0]

z = dict(np.load(args.input))
fps = float(np.asarray(z["fps"]).flatten()[0])
dt = 1.0 / fps
N = args.frames
jnames, bnames = list(z["joint_names"]), list(z["body_names"])
jidx = [jnames.index(n) for n in robot.joint_names]  # npz column for each robot joint
bidx = [robot.body_names.index(n) for n in bnames]  # robot body for each npz body
feet = [robot.body_names.index(n) for n in ("left_foot_link", "right_foot_link")]
ankle = [[jnames.index(f"{s}_Ankle_Pitch"), jnames.index(f"{s}_Ankle_Roll")] for s in ("Left", "Right")]
root_b = bnames.index(robot.body_names[0])
SOLE = [[0.0995, 0.04, -0.0382], [0.0995, -0.04, -0.0382], [-0.0657, 0.04, -0.0382], [-0.0657, -0.04, -0.0382]]
SOLE = torch.tensor([SOLE, [[x, -y, c] for x, y, c in SOLE]], device=dev)  # (2 feet, 4, 3)
UP = torch.tensor([0.0, 0.0, 1.0], device=dev)
t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)


def fk(f: int, q_npz: np.ndarray, root_z: float | None = None):
    root = robot.data.default_root_state.clone()
    root[:, :3] = t(z["body_pos_w"][f, root_b]) + origin
    if root_z is not None:
        root[:, 2] = root_z
    root[:, 3:7] = t(z["body_quat_w"][f, root_b])
    root[:, 7:] = 0.0
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(t(q_npz[jidx])[None], torch.zeros(1, len(jidx), device=dev))
    sim.physics_sim_view.update_articulations_kinematic()
    robot.update(0.0)


def foot_up_xy() -> torch.Tensor:  # (2, 2)
    return quat_apply(robot.data.body_quat_w[0, feet], UP.expand(2, 3))[:, :2]


def sole_min() -> torch.Tensor:  # (2,) lowest corner z per foot, world (env origin z is 0)
    pts = robot.data.body_pos_w[0, feet][:, None] + quat_apply(robot.data.body_quat_w[0, feet][:, None].expand(2, 4, 4), SOLE)
    return pts[..., 2].min(dim=1).values - origin[2]


new_q = z["joint_pos"].copy()
new_pos, new_quat = z["body_pos_w"].copy(), z["body_quat_w"].copy()
print("frame |  w   | sole tilt before -> after (deg, L/R) | lowest corner before -> after (mm, L/R)")
for f in range(N):
    s = f / N
    w = 1.0 - (3 * s**2 - 2 * s**3)
    q = z["joint_pos"][f].astype(np.float64).copy()
    fk(f, q)
    up0, low0 = foot_up_xy().clone(), sole_min().clone()
    target = (1.0 - w) * up0
    for _ in range(8):
        fk(f, q)
        e = foot_up_xy() - target
        if e.abs().max() < 1e-5:
            break
        for k in range(2):
            jac = torch.zeros(2, 2, device=dev)
            for c in range(2):
                dq = q.copy()
                dq[ankle[k][c]] += 1e-3
                fk(f, dq)
                jac[:, c] = (foot_up_xy()[k] - target[k] - e[k]) / 1e-3
            q[ankle[k]] -= torch.linalg.solve(jac, e[k]).cpu().numpy()
    fk(f, q)
    low1 = sole_min()
    # Lowest corner of the two feet goes to (1 - w) of its original height.
    dz = float((1.0 - w) * low0.min() - low1.min())
    root_z = float(z["body_pos_w"][f, root_b, 2] + origin[2]) + dz
    fk(f, q, root_z)
    tilt = lambda u: torch.rad2deg(torch.asin(u.norm(dim=-1).clamp(max=1.0))).tolist()
    low2 = sole_min()
    print(f"{f:5d} | {w:4.2f} | {tilt(up0)[0]:5.1f} {tilt(up0)[1]:5.1f} -> {tilt(foot_up_xy())[0]:5.1f} {tilt(foot_up_xy())[1]:5.1f}"
          f" | {1000 * low0[0]:6.1f} {1000 * low0[1]:6.1f} -> {1000 * low2[0]:6.1f} {1000 * low2[1]:6.1f}")
    new_q[f] = q
    new_pos[f] = (robot.data.body_pos_w[0, bidx] - origin).cpu().numpy()
    new_quat[f] = robot.data.body_quat_w[0, bidx].cpu().numpy()

# Velocities over the edited range (+1 frame of overlap so the seam is differenced too).
E = N + 1
z["joint_vel"][:E] = np.gradient(new_q[: E + 1], dt, axis=0)[:E]
z["body_lin_vel_w"][:E] = np.gradient(new_pos[: E + 1], dt, axis=0)[:E]
qq = new_quat[: E + 1].astype(np.float64)
for i in range(1, len(qq)):  # consistent hemisphere
    flip = np.sum(qq[i] * qq[i - 1], axis=-1) < 0
    qq[i][flip] *= -1
dq = np.gradient(qq, dt, axis=0)
w_, v = qq[..., :1], qq[..., 1:]
dw, dv = dq[..., :1], dq[..., 1:]
# omega_world = 2 * vec(dq * conj(q)) = 2 * (w*dv - dw*v - dv x v)
omega = 2 * (w_ * dv - dw * v - np.cross(dv, v))
z["body_ang_vel_w"][:E] = omega[:E]
z["joint_pos"], z["body_pos_w"], z["body_quat_w"] = new_q, new_pos, new_quat
np.savez(args.output, **z)
print(f"[INFO] wrote {args.output}: first {N} frames re-solved; max |joint_vel| change in range "
      f"{np.abs(np.load(args.input)['joint_vel'][:E] - z['joint_vel'][:E]).max():.2f} rad/s")
# Kit can hang in app.close() for a headless run like this; the npz is already on disk, so leave directly.
sys.stdout.flush()
os._exit(0)
