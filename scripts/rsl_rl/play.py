# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--clip_joint_limits",
    action="store_true",
    default=False,
    help="Clamp the commanded joint targets to the robot's soft joint limits. hoi_track's residual action term "
    "is unclipped during training, so the reference's own overshoot is commanded verbatim; this renders the "
    "clipped variant for comparison. Ignored by tasks whose action term has no clip_to_soft_limits option.",
)
parser.add_argument(
    "--disable_object_dr",
    action="store_true",
    default=False,
    help="Disable the manipulated object's startup domain randomization (mass, friction/restitution, inertia"
    " scale, CoM), e.g. to render against the nominal object. Ignored by tasks with no such events.",
)
parser.add_argument(
    "--enable_pushes",
    action="store_true",
    default=False,
    help="Re-enable the push events (push_robot, and push_object where the task has one) that -Play configs"
    " disable for clean rendering, copying each verbatim from the non-Play (training) task so the push"
    " magnitude/interval always matches what the policy trained against. Ignored by tasks with no push events.",
)
parser.add_argument(
    "--actuator_stress",
    action="store_true",
    default=False,
    help="Run the actuators at fixed worst-case values instead of nominal: Kp x0.6, Kd x1, strength x0.8 and every"
    " joint's friction at the top of its training range. Restores actuator_dr from the non-Play (training) task with"
    " those values pinned. -Play configs otherwise run nominal actuators.",
)
parser.add_argument(
    "--scenario",
    choices=("regular", "drop", "noobj", "mixed", "each"),
    default="regular",
    help="Object scenario to play. -Play configs run the regular task only; anything else restores"
    " object_scenario from the non-Play (training) task, so the drop force, cap and release settings match"
    " training, and overrides only which scenario each env gets. drop: every env has the box knocked out of"
    " its hands at a reference contact frame. noobj: the box is parked out of the scene. mixed: the training"
    " mix. each: env i runs scenario i mod 3 (regular, drop, noobj), so --num_envs 3 shows all three."
    " Ignored by tasks with no object_scenario event.",
)
parser.add_argument(
    "--overlay_terminations",
    action="store_true",
    default=False,
    help="Draw, on every recorded frame, each env's episode step, scenario and drop status, and which termination"
    " ended its previous episode (or time_out), so a video says why it reset. Needs --video; shows up to 4 envs.",
)
parser.add_argument(
    "--overlay_cop",
    action="store_true",
    default=False,
    help="Draw each foot's centre of pressure on every recorded frame: the sole outline seen from above, toe up,"
    " with the current CoP and a short trail, plus heel-to-toe position and load. Adds a render-only foot-ground"
    " contact sensor. A per-foot summary of heel/toe time is printed at the end. Needs --video; shows env 0.",
)
parser.add_argument(
    "--overlay_foot_flat",
    action="store_true",
    default=False,
    help="Draw each foot's four sole-corner heights above the floor on every recorded frame (upper right), with the"
    " reference stance label and the foot_flat_stance score, the sim's corners next to the reference's own. A per-foot"
    " summary over stance steps is printed at the end. Needs --video; shows env 0.",
)
parser.add_argument(
    "--motor_strength",
    type=float,
    default=1.0,
    help="Scale every robot actuator's output torque by this factor (e.g. 0.8), as legged_gym's motor strength"
    " does: for the explicit PD actuators that is Kp, Kd and the effort limit all scaled together. Render-time"
    " robustness test only; training is not affected.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_train.tasks  # noqa: F401


class TerminationOverlay(gym.Wrapper):
    """Annotates the rendered frames with why each env's last episode ended and where the current one stands.

    Sits between the env and RecordVideo. Episodes reset inside ``step``, so the frame after the ending step is
    already the next episode; the reason is therefore kept and shown for the whole of that next episode. Reasons
    come from the termination manager's per-term flags, which still hold the ended episode's values right after
    ``step``. Red is a failure term, green is ``time_out``.
    """

    MAX_ENVS = 4

    def __init__(self, env):
        super().__init__(env)
        self.n = min(env.unwrapped.num_envs, self.MAX_ENVS)
        self.ep_steps = [0] * self.n
        self.last_end = [None] * self.n  # (reasons, steps) of the last finished episode
        events = env.unwrapped.event_manager
        self.scenario = (
            events.get_term_cfg("object_scenario").func if "object_scenario" in events.active_terms.get("interval", []) else None
        )

    def reset(self, **kwargs):
        self.ep_steps = [0] * self.n
        return self.env.reset(**kwargs)

    def step(self, action):
        out = self.env.step(action)
        terminations = self.env.unwrapped.termination_manager
        dones = terminations.dones[: self.n].tolist()
        for i in range(self.n):
            self.ep_steps[i] += 1
            if dones[i]:
                # Every env "ends" on the very first step of a play run, before the first real reset has taken
                # effect; that is not an episode, so it is not reported.
                if not (self.ep_steps[i] == 1 and self.last_end[i] is None):
                    reasons = [n for n in terminations.active_terms if bool(terminations.get_term(n)[i])]
                    self.last_end[i] = (reasons or ["?"], self.ep_steps[i])
                self.ep_steps[i] = 0
        return out

    def _status(self, i: int) -> str:
        term = self.scenario
        if term is None:
            return ""
        kind = int(term.scenario[i])
        text = term.SCENARIO_NAMES[kind]
        if kind == term.DROP:
            if bool(term.grasp_dropped[i]):
                text += ": released"
            elif bool(term.force_capped[i]):
                text += ": force capped, grip held"
            elif bool(term.drop_triggered[i]):
                text += ": force on"
            else:
                text += ": waiting"
        if bool(term.object_absent[i]):
            text += " [object absent]"
        return text

    def render(self):
        import cv2

        frame = self.env.render()
        if frame is None:
            return frame
        frame = np.ascontiguousarray(frame)
        dt = self.env.unwrapped.step_dt
        lines = []  # (text, RGB colour)
        for i in range(self.n):
            head = f"env {i}  step {self.ep_steps[i]} ({self.ep_steps[i] * dt:.1f}s)"
            status = self._status(i)
            lines.append((head + (f"  |  {status}" if status else ""), (255, 255, 255)))
            if self.last_end[i] is None:
                lines.append(("   last end: none yet", (170, 170, 170)))
            else:
                reasons, steps = self.last_end[i]
                colour = (90, 220, 120) if reasons == ["time_out"] else (255, 90, 90)
                lines.append((f"   last end: {' + '.join(reasons)}  at step {steps} ({steps * dt:.1f}s)", colour))

        scale = frame.shape[0] / 720 * 0.6
        thick = max(1, round(scale * 2))
        height = int(30 * scale * 1.3)
        box_w = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] for t, _ in lines) + 24 * scale)
        box_h = height * len(lines) + int(12 * scale)
        region = frame[8 : 8 + box_h, 8 : 8 + box_w]
        region[:] = (region * 0.35).astype(frame.dtype)  # darken the backing so the text reads on any floor
        for k, (text, colour) in enumerate(lines):
            cv2.putText(frame, text, (8 + int(10 * scale), 8 + height * (k + 1) - int(6 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)
        return frame


class CoPOverlay(gym.Wrapper):
    """Per-foot centre of pressure, drawn as a top-down sole diagram in the lower-left of each frame.

    The CoP is the normal-force-weighted mean of the foot's ground contact points, taken straight from the PhysX
    contact view. Isaac Lab's ``contact_pos_w`` cannot be used: it is the *unweighted* mean of the points, so a
    foot loaded almost entirely on its toe still reads mid-sole. The point is then expressed in the foot link's
    frame, whose origin is the ankle, and drawn against the K1 sole outline measured from the foot STL.

    The dot turns orange in the outer HEEL_TOE_ZONE of the sole and red in the outer EDGE_ZONE -- the regime where
    a foot is about to roll onto its heel or toe. ``summary()`` reports, per foot, the share of loaded steps spent
    in each zone.

    A second panel (lower right) shows the ground plane from above, turned so the trunk heading points up: both
    sole outlines, the per-foot CoPs, the load-weighted global CoP, and the *reference support centre* -- the mean
    sole centre of the feet the reference has in stance -- inside the convex hull of those feet. The motion files
    carry no foot labels, so stance is derived from the reference kinematics: ankle within STANCE_HEIGHT of that
    foot's lowest point in the clip, and horizontal speed below STANCE_SPEED, with runs shorter than STANCE_MIN_RUN
    frames absorbed into their neighbours. Height is clip-relative because the retargeted feet do not sit at one
    height: sub3_largebox_003's right ankle holds still at 3.1 cm above its minimum for a quarter of the clip,
    which an absolute cutoff at the flat-foot height (5.8 cm) left with no foot in stance on 25% of frames. The
    support centre and hull use the *simulated* soles, so the CoP-to-support gap measures balance, not foot
    placement.
    """

    # Sole outline in the foot link frame (m), from meshes/{Left,Right}_Foot.STL; identical up to mirroring in y.
    SOLE_X = (-0.066, 0.109)  # heel, toe
    SOLE_Y = (-0.040, 0.040)
    HEEL_TOE_ZONE = 0.20  # outer fraction of sole length counted as "on heel" / "on toe"
    EDGE_ZONE = 0.07  # outer fraction where the foot is effectively balanced on its edge
    MIN_LOAD = 5.0  # N; below this the foot is in the air and has no meaningful CoP
    TRAIL = 25
    SOLE_BOTTOM_Z = -0.038  # sole bottom below the ankle, foot frame
    SOLE_CENTRE = (0.0215, 0.0, -0.038)
    STANCE_HEIGHT = 0.04  # m above that foot's lowest reference ankle height in the clip
    STANCE_SPEED = 0.25  # m/s, reference foot horizontal speed
    STANCE_MIN_RUN = 3  # frames
    VIEW_M = 0.6  # ground panel width in metres

    def __init__(self, env, sensor_name: str, env_index: int = 0):
        super().__init__(env)
        self.i = env_index
        self.sensor = env.unwrapped.scene[sensor_name]
        robot = env.unwrapped.scene["robot"]
        self.robot = robot
        self.feet = []  # (label, sensor body index, robot body index)
        for label, name in (("L", "left_foot_link"), ("R", "right_foot_link")):
            self.feet.append((label, self.sensor.body_names.index(name), robot.body_names.index(name)))
        self.cop = {label: None for label, _, _ in self.feet}  # (x, y) in foot frame, or None when unloaded
        self.load = {label: 0.0 for label, _, _ in self.feet}
        self.trail = {label: [] for label, _, _ in self.feet}
        self.counts = {label: {"loaded": 0, "heel": 0, "toe": 0, "heel_edge": 0, "toe_edge": 0} for label, _, _ in self.feet}

        # -- ground-plane support view
        command = env.unwrapped.command_manager.get_term("motion")
        self.command = command
        self.anchor_index = robot.body_names.index(command.cfg.anchor_body_name)
        motion = command.motion
        from booster_train.tasks.manager_based.hoi_track.mdp.rewards import reference_stance

        table, labelled = reference_stance(command, ("left_foot_link", "right_foot_link"))
        self.stance = {"L": table[:, 0].cpu().numpy(), "R": table[:, 1].cpu().numpy()}
        print(f"[CoP] reference stance: baked labels on {labelled.float().mean().item() * 100:.0f}% of frames; elsewhere"
              f" ankle < {self.STANCE_HEIGHT} m above its clip minimum and xy speed < {self.STANCE_SPEED} m/s,"
              f" runs < {self.STANCE_MIN_RUN} frames absorbed")
        names = getattr(motion, "clip_names", [str(c) for c in range(motion.num_clips)])
        for c, (start, length) in enumerate(zip(motion.clip_starts.tolist(), motion.clip_lengths.tolist())):
            fl = self.stance["L"][start : start + length].mean()
            fr = self.stance["R"][start : start + length].mean()
            both = (self.stance["L"][start : start + length] & self.stance["R"][start : start + length]).mean()
            print(f"[CoP]   {names[c]}: L stance {100 * fl:.0f}% | R stance {100 * fr:.0f}% | double {100 * both:.0f}%")
        self.ground = None  # per-step ground-plane state, filled by _measure
        self.global_trail = []
        self.support = {"n": 0, "dist": [], "outside": 0, "min_margin": float("inf")}

    @classmethod
    def _debounce(cls, raw: np.ndarray, starts: list[int], lengths: list[int]) -> np.ndarray:
        """Absorb stance/swing runs shorter than STANCE_MIN_RUN into the run before them, per clip."""
        out = raw.copy()
        for start, length in zip(starts, lengths):
            seg = out[start : start + length]
            i = 0
            while i < length:
                j = i
                while j < length and seg[j] == seg[i]:
                    j += 1
                if j - i < cls.STANCE_MIN_RUN and i > 0:
                    seg[i:j] = seg[i - 1]
                i = j
        return out

    def _heel_toe(self, x: float) -> float:
        """0 at the heel edge, 1 at the toe edge."""
        return (x - self.SOLE_X[0]) / (self.SOLE_X[1] - self.SOLE_X[0])

    def _measure(self):
        forces, points, _, _, count, start = self.sensor.contact_physx_view.get_contact_data(
            dt=self.env.unwrapped.physics_dt
        )
        from isaaclab.utils.math import quat_apply

        forces = forces.view(-1).abs()
        num_bodies = self.sensor.num_bodies
        world_cop, soles, centres = {}, {}, {}
        sole_local = torch.tensor(
            [[self.SOLE_X[0], self.SOLE_Y[0], self.SOLE_BOTTOM_Z], [self.SOLE_X[1], self.SOLE_Y[0], self.SOLE_BOTTOM_Z],
             [self.SOLE_X[1], self.SOLE_Y[1], self.SOLE_BOTTOM_Z], [self.SOLE_X[0], self.SOLE_Y[1], self.SOLE_BOTTOM_Z],
             list(self.SOLE_CENTRE)],
            device=self.robot.device,
        )
        for label, s_idx, r_idx in self.feet:
            foot_pos = self.robot.data.body_pos_w[self.i, r_idx]
            foot_quat = self.robot.data.body_quat_w[self.i, r_idx]
            pts = (foot_pos + quat_apply(foot_quat[None].expand(5, -1), sole_local))[:, :2].cpu().numpy()
            soles[label], centres[label] = pts[:4], pts[4]
            world_cop[label] = None
            row = self.i * num_bodies + s_idx
            n = int(count[row, 0])
            f_total = 0.0
            self.cop[label] = None
            if n > 0:
                k = int(start[row, 0])
                f = forces[k : k + n]
                f_total = float(f.sum())
                if f_total > self.MIN_LOAD:
                    p_w = (f[:, None] * points[k : k + n]).sum(0) / f.sum()
                    pos = self.robot.data.body_pos_w[self.i, r_idx]
                    quat = self.robot.data.body_quat_w[self.i, r_idx]
                    from isaaclab.utils.math import quat_apply_inverse

                    local = quat_apply_inverse(quat[None], (p_w - pos)[None])[0]
                    self.cop[label] = (float(local[0]), float(local[1]))
                    world_cop[label] = p_w[:2].cpu().numpy()
            self.load[label] = f_total
            trail = self.trail[label]
            trail.append(self.cop[label])
            del trail[: -self.TRAIL]

            if self.cop[label] is not None:
                c = self.counts[label]
                c["loaded"] += 1
                u = self._heel_toe(self.cop[label][0])
                c["heel"] += u < self.HEEL_TOE_ZONE
                c["toe"] += u > 1 - self.HEEL_TOE_ZONE
                c["heel_edge"] += u < self.EDGE_ZONE
                c["toe_edge"] += u > 1 - self.EDGE_ZONE

        self._measure_support(world_cop, soles, centres)

    def _measure_support(self, world_cop: dict, soles: dict, centres: dict):
        import cv2

        frame = int(self.command.time_steps[self.i])
        stance = {label: bool(self.stance[label][frame]) for label, _, _ in self.feet}
        loads = {label: self.load[label] if world_cop[label] is not None else 0.0 for label, _, _ in self.feet}
        total = sum(loads.values())
        p_cop = None
        if total > self.MIN_LOAD:
            p_cop = sum(loads[l] * world_cop[l] for l in loads if world_cop[l] is not None) / total
        in_stance = [l for l in stance if stance[l]]
        p_support = np.mean([centres[l] for l in in_stance], axis=0) if in_stance else None
        hull = cv2.convexHull(np.concatenate([soles[l] for l in in_stance]).astype(np.float32)) if in_stance else None

        dist = margin = None
        if p_cop is not None and p_support is not None:
            dist = float(np.linalg.norm(p_cop - p_support))
            margin = float(cv2.pointPolygonTest(hull, (float(p_cop[0]), float(p_cop[1])), True))
            st = self.support
            st["n"] += 1
            st["dist"].append(dist)
            st["outside"] += margin < 0
            st["min_margin"] = min(st["min_margin"], margin)

        anchor_quat = self.robot.data.body_quat_w[self.i, self.anchor_index]
        w, x, y, z = anchor_quat.tolist()
        heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.ground = dict(world_cop=world_cop, soles=soles, centres=centres, stance=stance, loads=loads,
                           p_cop=p_cop, p_support=p_support, hull=hull, dist=dist, margin=margin, heading=heading)
        self.global_trail.append(p_cop)
        del self.global_trail[: -self.TRAIL]

    def step(self, action):
        out = self.env.step(action)
        self._measure()
        return out

    def summary(self) -> str:
        lines = ["[CoP] share of loaded steps (env %d):" % self.i]
        for label, _, _ in self.feet:
            c = self.counts[label]
            n = max(c["loaded"], 1)
            lines.append(
                f"[CoP]   {label}: loaded {c['loaded']} steps | heel {100 * c['heel'] / n:.1f}% (edge "
                f"{100 * c['heel_edge'] / n:.1f}%) | toe {100 * c['toe'] / n:.1f}% (edge {100 * c['toe_edge'] / n:.1f}%)"
            )
        st = self.support
        if st["n"]:
            d = np.array(st["dist"]) * 100
            lines.append(
                f"[CoP]   global CoP vs reference support centre, {st['n']} loaded steps: mean {d.mean():.1f} cm | "
                f"p90 {np.percentile(d, 90):.1f} cm | outside ref-stance hull {100 * st['outside'] / st['n']:.1f}% | "
                f"min hull margin {100 * st['min_margin']:.1f} cm"
            )
        return "\n".join(lines)

    def render(self):
        import cv2

        frame = self.env.render()
        if frame is None:
            return frame
        frame = np.ascontiguousarray(frame)
        scale = frame.shape[0] / 720
        px_per_m = 900 * scale  # sole is ~0.175 m long -> ~158 px at 720p
        sole_w = int((self.SOLE_Y[1] - self.SOLE_Y[0]) * px_per_m)
        sole_h = int((self.SOLE_X[1] - self.SOLE_X[0]) * px_per_m)
        pad = int(14 * scale)
        text_h = int(40 * scale)
        panel_w = 2 * sole_w + 3 * pad + int(70 * scale)
        panel_h = sole_h + 2 * pad + 2 * text_h
        x0, y0 = int(8 * scale), frame.shape[0] - panel_h - int(8 * scale)
        region = frame[y0 : y0 + panel_h, x0 : x0 + panel_w]
        region[:] = (region * 0.35).astype(frame.dtype)
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, max(1, round(scale))

        def to_px(ox: int, x: float, y: float) -> tuple[int, int]:
            # top-down, toe up: foot +x is image up, foot +y (left) is image left
            u = ox + int((self.SOLE_Y[1] - y) * px_per_m)
            v = y0 + pad + text_h + int((self.SOLE_X[1] - x) * px_per_m)
            return u, v

        for k, (label, _, _) in enumerate(self.feet):
            ox = x0 + pad + int(35 * scale) + k * (sole_w + pad + int(35 * scale))
            top, bottom = to_px(ox, self.SOLE_X[1], self.SOLE_Y[1]), to_px(ox, self.SOLE_X[0], self.SOLE_Y[0])
            # heel / toe zones shaded, then the outline and the ankle (foot frame origin)
            zone = int(self.HEEL_TOE_ZONE * sole_h)
            for (a, b) in ((top[1], top[1] + zone), (bottom[1] - zone, bottom[1])):
                band = frame[a:b, top[0] : bottom[0]]
                band[:] = np.clip(band.astype(np.int32) + 25, 0, 255).astype(frame.dtype)
            cv2.rectangle(frame, top, bottom, (200, 200, 200), th, cv2.LINE_AA)
            cv2.drawMarker(frame, to_px(ox, 0.0, 0.0), (140, 140, 140), cv2.MARKER_CROSS, int(10 * scale), th)
            cv2.putText(frame, "toe", (top[0], top[1] - int(4 * scale)), font, 0.4 * scale, (170, 170, 170), th, cv2.LINE_AA)
            cv2.putText(frame, "heel", (top[0], bottom[1] + int(14 * scale)), font, 0.4 * scale, (170, 170, 170), th, cv2.LINE_AA)

            points = [p for p in self.trail[label] if p is not None]
            for j in range(1, len(points)):
                cv2.line(frame, to_px(ox, *points[j - 1]), to_px(ox, *points[j]), (90, 160, 255), th, cv2.LINE_AA)

            cop = self.cop[label]
            if cop is None:
                status, colour = f"{label}: air", (170, 170, 170)
            else:
                u = self._heel_toe(cop[0])
                edge = min(u, 1 - u)
                colour = (255, 80, 80) if edge < self.EDGE_ZONE else (255, 170, 60) if edge < self.HEEL_TOE_ZONE else (90, 220, 120)
                cv2.circle(frame, to_px(ox, *cop), int(6 * scale), colour, -1, cv2.LINE_AA)
                status = f"{label}: {100 * u:.0f}% {self.load[label]:.0f}N"
            cv2.putText(frame, status, (ox - int(30 * scale), y0 + pad + int(14 * scale)), font, fs, colour, th, cv2.LINE_AA)

        cv2.putText(frame, "CoP: 0% heel .. 100% toe", (x0 + pad, y0 + panel_h - int(8 * scale)), font, 0.42 * scale,
                    (200, 200, 200), th, cv2.LINE_AA)
        if self.ground is not None:
            self._render_ground(frame, scale, font, th)
        return frame

    def _render_ground(self, frame: np.ndarray, scale: float, font: int, th: int):
        import cv2

        g = self.ground
        size = int(300 * scale)
        text_h = int(62 * scale)
        x0 = frame.shape[1] - size - int(8 * scale)
        y0 = frame.shape[0] - size - text_h - int(8 * scale)
        region = frame[y0 : y0 + size + text_h, x0 : x0 + size]
        region[:] = (region * 0.35).astype(frame.dtype)
        centre = (g["centres"]["L"] + g["centres"]["R"]) / 2
        cos, sin = np.cos(g["heading"]), np.sin(g["heading"])
        ppm = size / self.VIEW_M
        cx, cy = x0 + size // 2, y0 + text_h + size // 2 - int(8 * scale)

        def px(p) -> tuple[int, int]:
            d = np.asarray(p, dtype=float) - centre
            fwd, left = cos * d[0] + sin * d[1], -sin * d[0] + cos * d[1]  # into the heading frame
            return int(cx - left * ppm), int(cy - fwd * ppm)  # heading up, robot's left to image left

        for label, _, _ in self.feet:
            poly = np.array([px(p) for p in g["soles"][label]], dtype=np.int32)
            if g["loads"][label] > self.MIN_LOAD:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly], (120, 120, 120))
                frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            cv2.polylines(frame, [poly], True, (190, 190, 190), th, cv2.LINE_AA)
        if g["hull"] is not None:
            hull = np.array([px(p) for p in g["hull"].reshape(-1, 2)], dtype=np.int32)
            cv2.polylines(frame, [hull], True, (90, 220, 120), max(th, 2), cv2.LINE_AA)
        if g["p_support"] is not None:
            cv2.drawMarker(frame, px(g["p_support"]), (90, 220, 120), cv2.MARKER_CROSS, int(16 * scale), max(th, 2))

        trail = [px(p) for p in self.global_trail if p is not None]
        for a, b in zip(trail, trail[1:]):
            cv2.line(frame, a, b, (200, 200, 200), th, cv2.LINE_AA)
        for label, colour in (("L", (90, 160, 255)), ("R", (230, 90, 230))):
            if g["world_cop"][label] is not None:
                cv2.circle(frame, px(g["world_cop"][label]), int(4 * scale), colour, -1, cv2.LINE_AA)
        if g["p_cop"] is not None:
            inside = g["margin"] is None or g["margin"] >= 0
            cv2.circle(frame, px(g["p_cop"]), int(7 * scale), (255, 255, 255) if inside else (255, 70, 70), -1, cv2.LINE_AA)

        ref = "".join(l for l in ("L", "R") if g["stance"][l]) or "-"
        if g["dist"] is None:
            text = "CoP: n/a"
            colour = (170, 170, 170)
        else:
            text = f"CoP->support {100 * g['dist']:.1f}cm  hull margin {100 * g['margin']:+.1f}cm"
            colour = (255, 255, 255) if g["margin"] >= 0 else (255, 90, 90)
        tx = x0 + int(8 * scale)
        cv2.putText(frame, f"ref stance: {ref}", (tx, y0 + int(18 * scale)), font, 0.42 * scale, (90, 220, 120), th, cv2.LINE_AA)
        cv2.putText(frame, text, (tx, y0 + int(36 * scale)), font, 0.42 * scale, colour, th, cv2.LINE_AA)
        cv2.putText(frame, "o global CoP  + support  L/R CoP  heading up", (tx, y0 + int(54 * scale)),
                    font, 0.36 * scale, (170, 170, 170), th, cv2.LINE_AA)


class FootFlatOverlay(gym.Wrapper):
    """Sole-corner heights of each foot, drawn top-down (toe up) in the upper right of each frame.

    Mirrors the ``foot_flat_stance`` reward (mdp.FootFlatStance): the same four corners, reference stance labels
    (mdp.reference_stance) and score ``exp(-sum(max(h - tol, 0)^2) / sigma^2)``, with tol/sigma read from the env's
    own reward config, so the overlay shows what training pays. Each corner is a dot coloured by its height above the
    floor -- green within ``tol``, yellow to 10 mm, orange to 20 mm, red beyond -- labelled with the sim height and,
    in brackets, the reference's height at that corner on the same frame. ``summary()`` reports, per foot, the mean
    score and corner heights over stance steps and the share of stance steps spent below 0.5.
    """

    SOLE_X = (-0.0657, 0.0995)  # heel, toe corners
    SOLE_Y = (-0.04, 0.04)

    def __init__(self, env, env_index: int = 0):
        super().__init__(env)
        from booster_train.tasks.manager_based.hoi_track.mdp.rewards import SOLE_CORNERS_LEFT, reference_stance

        self.i = env_index
        uw = env.unwrapped
        self.robot = uw.scene["robot"]
        self.command = uw.command_manager.get_term("motion")
        names = ("left_foot_link", "right_foot_link")
        self.robot_ids = [self.robot.body_names.index(n) for n in names]
        self.ref_ids = [self.command.cfg.body_names.index(n) for n in names]
        params = {}
        if "foot_flat_stance" in uw.reward_manager.active_terms:
            params = uw.reward_manager.get_term_cfg("foot_flat_stance").params
        self.tol = params.get("tol", 0.004)
        self.sigma = params.get("sigma", 0.015)
        left = torch.tensor(SOLE_CORNERS_LEFT, device=self.robot.device)
        self.corners = torch.stack([left, left * torch.tensor([1.0, -1.0, 1.0], device=self.robot.device)])  # (2, 4, 3)
        table, labelled = reference_stance(self.command, names)
        self.stance = table.cpu().numpy()
        print(f"[FootFlat] stance from baked labels on {labelled.float().mean().item() * 100:.0f}% of frames;"
              f" tol {self.tol * 1000:.0f} mm, sigma {self.sigma * 1000:.0f} mm")
        self.state = None
        self.stats = {label: {"n": 0, "score": 0.0, "low": 0, "h": np.zeros(4)} for label in ("L", "R")}

    def _heights(self, pos: torch.Tensor, quat: torch.Tensor) -> np.ndarray:
        from booster_train.tasks.manager_based.hoi_track.mdp.rewards import sole_corner_heights

        return sole_corner_heights(pos, quat, self.corners).cpu().numpy()  # (2, 4)

    def _score(self, h: np.ndarray) -> np.ndarray:
        return np.exp(-(np.clip(h - self.tol, 0.0, None) ** 2).sum(-1) / self.sigma**2)

    def step(self, action):
        out = self.env.step(action)
        frame = int(self.command.time_steps[self.i])
        motion = self.command.motion
        sim = self._heights(self.robot.data.body_pos_w[self.i, self.robot_ids], self.robot.data.body_quat_w[self.i, self.robot_ids])
        ref = self._heights(motion.body_pos_w[frame, self.ref_ids], motion.body_quat_w[frame, self.ref_ids])
        stance = self.stance[frame]
        score = self._score(sim)
        self.state = dict(sim=sim, ref=ref, stance=stance, score=score, ref_score=self._score(ref))
        for f, label in enumerate(("L", "R")):
            if stance[f]:
                st = self.stats[label]
                st["n"] += 1
                st["score"] += score[f]
                st["low"] += score[f] < 0.5
                st["h"] += sim[f]
        return out

    def summary(self) -> str:
        lines = [f"[FootFlat] stance steps (env {self.i}); corners toe-out / toe-in / heel-out / heel-in:"]
        for label, st in self.stats.items():
            if st["n"] == 0:
                lines.append(f"[FootFlat]   {label}: no stance steps")
                continue
            h = st["h"] / st["n"] * 1000
            lines.append(
                f"[FootFlat]   {label}: {st['n']} stance steps | mean score {st['score'] / st['n']:.3f} | below 0.5 on "
                f"{100 * st['low'] / st['n']:.1f}% | mean corners {h[0]:.1f} / {h[1]:.1f} / {h[2]:.1f} / {h[3]:.1f} mm"
            )
        return "\n".join(lines)

    def _colour(self, h: float) -> tuple[int, int, int]:
        if h <= self.tol:
            return (90, 220, 120)
        if h <= 0.01:
            return (240, 220, 80)
        if h <= 0.02:
            return (255, 160, 60)
        return (255, 70, 70)

    def render(self):
        import cv2

        frame = self.env.render()
        if frame is None or self.state is None:
            return frame
        frame = np.ascontiguousarray(frame)
        g = self.state
        scale = frame.shape[0] / 720
        font, th = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(scale))
        px_per_m = 700 * scale  # sole ~0.165 m long -> ~116 px at 720p
        sole_w = int((self.SOLE_Y[1] - self.SOLE_Y[0]) * px_per_m)
        sole_h = int((self.SOLE_X[1] - self.SOLE_X[0]) * px_per_m)
        pad, head, side = int(12 * scale), int(48 * scale), int(70 * scale)
        foot_w = sole_w + 2 * side
        panel_w = 2 * foot_w + 3 * pad
        panel_h = head + sole_h + int(56 * scale)
        x0, y0 = frame.shape[1] - panel_w - int(8 * scale), int(8 * scale)
        region = frame[y0 : y0 + panel_h, x0 : x0 + panel_w]
        region[:] = (region * 0.35).astype(frame.dtype)

        for f, label in enumerate(("L", "R")):
            ox = x0 + pad + f * (foot_w + pad) + side  # sole rectangle's left edge

            def to_px(x: float, y: float) -> tuple[int, int]:
                # top-down, toe up, robot's left to image left
                return ox + int((self.SOLE_Y[1] - y) * px_per_m), y0 + head + int((self.SOLE_X[1] - x) * px_per_m)

            stance = bool(g["stance"][f])
            top, bottom = to_px(self.SOLE_X[1], self.SOLE_Y[1]), to_px(self.SOLE_X[0], self.SOLE_Y[0])
            cv2.rectangle(frame, top, bottom, (200, 200, 200) if stance else (110, 110, 110), th, cv2.LINE_AA)
            mid = (top[0] + bottom[0]) // 2
            for c, (cx, cy, _) in enumerate(self.corners[f].tolist()):
                h, r = float(g["sim"][f, c]), float(g["ref"][f, c])
                p = to_px(cx, cy)
                cv2.circle(frame, p, int(7 * scale), self._colour(h), -1, cv2.LINE_AA)
                text = f"{h * 1000:.0f} ({r * 1000:.0f})"
                (tw, _), _ = cv2.getTextSize(text, font, 0.38 * scale, th)
                tx = p[0] - tw - int(10 * scale) if p[0] < mid else p[0] + int(10 * scale)
                cv2.putText(frame, text, (tx, p[1] + int(4 * scale)), font, 0.38 * scale, (230, 230, 230), th, cv2.LINE_AA)
            score = float(g["score"][f])
            colour = (90, 220, 120) if score > 0.8 else (255, 160, 60) if score > 0.5 else (255, 70, 70)
            title = f"{label} STANCE {score:.2f}" if stance else f"{label} swing {score:.2f}"
            tx = ox - side + int(6 * scale)
            cv2.putText(frame, title, (tx, y0 + int(18 * scale)), font, 0.5 * scale,
                        colour if stance else (150, 150, 150), th, cv2.LINE_AA)
            cv2.putText(frame, f"ref {float(g['ref_score'][f]):.2f}", (tx, y0 + int(36 * scale)), font, 0.4 * scale,
                        (170, 170, 170), th, cv2.LINE_AA)
            cv2.putText(frame, "toe", (mid - int(10 * scale), top[1] - int(4 * scale)), font, 0.36 * scale,
                        (150, 150, 150), th, cv2.LINE_AA)
        cv2.putText(frame, "corner height mm: sim (ref)", (x0 + pad, y0 + panel_h - int(10 * scale)), font,
                    0.42 * scale, (200, 200, 200), th, cv2.LINE_AA)
        return frame


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Render-time only: hoi_track's ResidualJointPositionAction is deliberately unclipped in training. Guarded by
    # hasattr because play.py is shared with tasks (beyond_mimic, ...) whose action term has no such option.
    if args_cli.clip_joint_limits:
        action_cfg = getattr(env_cfg.actions, "joint_pos", None)
        if action_cfg is not None and hasattr(action_cfg, "clip_to_soft_limits"):
            action_cfg.clip_to_soft_limits = True
            print("[INFO] Clamping commanded joint targets to the soft joint limits.")
        else:
            print(f"[WARN] --clip_joint_limits ignored: {args_cli.task} has no clippable joint_pos action term.")

    if args_cli.disable_object_dr:
        events_cfg = getattr(env_cfg, "events", None)
        object_dr_terms = ("object_physics_material", "object_mass", "object_inertia_scale", "object_com")
        disabled = [name for name in object_dr_terms if getattr(events_cfg, name, None) is not None]
        for name in disabled:
            setattr(events_cfg, name, None)
        if disabled:
            print(f"[INFO] --disable_object_dr: disabled event terms {disabled}")
        else:
            print(f"[WARN] --disable_object_dr ignored: {args_cli.task} has no object DR event terms.")

    if args_cli.enable_pushes:
        train_events = getattr(load_cfg_from_registry(train_task_name, "env_cfg_entry_point"), "events", None)
        restored = []
        for name in ("push_robot", "push_object"):
            push_cfg = getattr(train_events, name, None)
            if push_cfg is not None:
                setattr(env_cfg.events, name, push_cfg)
                restored.append(name)
                print(
                    f"[INFO] --enable_pushes: restored {name} from {train_task_name} "
                    f"(interval {push_cfg.interval_range_s}, range {push_cfg.params['velocity_range']})"
                )
        if not restored:
            print(f"[WARN] --enable_pushes ignored: {train_task_name} has no push events.")

    if args_cli.actuator_stress:
        dr_cfg = getattr(getattr(load_cfg_from_registry(train_task_name, "env_cfg_entry_point"), "events", None),
                         "actuator_dr", None)
        if dr_cfg is None:
            print(f"[WARN] --actuator_stress ignored: {train_task_name} has no actuator_dr event.")
        else:
            dr_cfg.params = {**dr_cfg.params, "fixed": {"kp": 0.6, "kd": 1.0, "strength": 0.8, "friction": "max"}}
            env_cfg.events.actuator_dr = dr_cfg
            print("[INFO] --actuator_stress: Kp x0.6, Kd x1.0, strength x0.8, joint friction at max "
                  f"{dr_cfg.params['friction_max']}")

    if args_cli.scenario != "regular":
        train_cfg = getattr(getattr(load_cfg_from_registry(train_task_name, "env_cfg_entry_point"), "events", None),
                            "object_scenario", None)
        if train_cfg is None:
            print(f"[WARN] --scenario ignored: {train_task_name} has no object_scenario event.")
        else:
            params = dict(train_cfg.params)
            if args_cli.scenario == "each":
                params["fixed_scenarios"] = (0, 1, 2)
                # The camera below is placed in world coordinates; an env-anchored viewer would offset it.
                env_cfg.viewer.origin_type = "world"
            elif args_cli.scenario != "mixed":
                params["probabilities"] = {"drop": (0.0, 1.0, 0.0), "noobj": (0.0, 0.0, 1.0)}[args_cli.scenario]
            env_cfg.events.object_scenario = train_cfg.replace(params=params)
            print(f"[INFO] --scenario {args_cli.scenario}: restored object_scenario from {train_task_name} "
                  f"(probabilities {params['probabilities']}, fixed {params.get('fixed_scenarios')}, "
                  f"force x{params['force_scale_range']}, cap {params['max_force_s']} s)")

    if args_cli.overlay_cop:
        from isaaclab.sensors import ContactSensorCfg

        # Render-only sensor: feet against the ground collider, with per-contact points so the CoP can be weighted
        # by each contact's normal force. The plane prim is fixed by Isaac Lab's grid ground-plane asset.
        env_cfg.scene.foot_ground_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_foot_link",
            filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
            track_contact_points=True,
            max_contact_data_count_per_prim=32,
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.overlay_terminations:
        if args_cli.video:
            env = TerminationOverlay(env)
        else:
            print("[WARN] --overlay_terminations ignored: it needs --video.")
    cop_overlay = None
    if args_cli.overlay_cop:
        if args_cli.video:
            env = cop_overlay = CoPOverlay(env, "foot_ground_contact")
        else:
            print("[WARN] --overlay_cop ignored: it needs --video.")

    foot_flat_overlay = None
    if args_cli.overlay_foot_flat:
        if args_cli.video:
            env = foot_flat_overlay = FootFlatOverlay(env)
        else:
            print("[WARN] --overlay_foot_flat ignored: it needs --video.")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    elif hasattr(ppo_runner, "obs_normalizer"):     # compatibility for older versions
        normalizer = ppo_runner.obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    run_name = os.path.basename(log_dir)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir,
                         filename=f"{agent_cfg.experiment_name}_{run_name}.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=normalizer, path=export_model_dir,
        filename=f"{agent_cfg.experiment_name}_{run_name}.onnx"
    )

    if args_cli.headless and not args_cli.video:
        print("[INFO] Headless mode and no video recording. Exiting after model export.")
        env.close()
        return

    dt = env.unwrapped.step_dt

    if args_cli.motor_strength != 1.0:
        # The actuator tensors are read on every compute(), so scaling them in place takes effect immediately.
        # Implicit actuators would need the gains written back to PhysX too; say so rather than half-apply it.
        robot = env.unwrapped.scene["robot"]
        for name, actuator in robot.actuators.items():
            if not hasattr(actuator, "_clip_effort") or actuator.__class__.__name__.startswith(("Implicit", "DelayedImplicit")):
                print(f"[WARN] --motor_strength: actuator group '{name}' ({type(actuator).__name__}) is implicit; skipped.")
                continue
            actuator.stiffness *= args_cli.motor_strength
            actuator.damping *= args_cli.motor_strength
            actuator.effort_limit *= args_cli.motor_strength
            print(f"[INFO] --motor_strength {args_cli.motor_strength}: scaled Kp, Kd and effort limit of '{name}' "
                  f"({type(actuator).__name__})")

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs
    timestep = 0

    # Frame all three scenario envs at once: the default view is a close-up of env 0 that hides the others.
    if args_cli.scenario == "each" and hasattr(env.unwrapped, "viewport_camera_controller"):
        origins = env.unwrapped.scene.env_origins[:3]
        centre = origins.mean(dim=0)
        env.unwrapped.viewport_camera_controller.update_view_location(
            eye=(centre + torch.tensor([0.0, -5.0, 5.0], device=centre.device)).tolist(),
            lookat=(centre + torch.tensor([0.0, 0.0, 0.0], device=centre.device)).tolist(),
        )

    # Scenario report: what each env was dealt, then when its drop starts and when the grasp is released, in
    # rollout steps. Frames are clip-relative, so they line up with the reference motion.
    scenario_term = None
    if args_cli.scenario != "regular":
        scenario_cfg = env.unwrapped.event_manager.get_term_cfg("object_scenario") if (
            "object_scenario" in env.unwrapped.event_manager.active_terms.get("interval", [])
        ) else None
        scenario_term = scenario_cfg.func if scenario_cfg is not None else None
    if scenario_term is not None:
        names = scenario_term.SCENARIO_NAMES
        motion = env.unwrapped.command_manager.get_term("motion")
        clip_start = motion.motion.clip_starts[motion.motion_ids]
        for i in range(min(env.unwrapped.num_envs, 16)):
            line = f"[SCENARIO] env {i}: {names[int(scenario_term.scenario[i])]}"
            if int(scenario_term.scenario[i]) == scenario_term.DROP:
                line += f", drop at clip frame {int(scenario_term.drop_frame[i] - clip_start[i])}"
            print(line)
        seen = {k: getattr(scenario_term, k).clone() for k in ("drop_triggered", "grasp_dropped", "force_capped")}
        step_count = 0
        terminations = env.unwrapped.termination_manager

        def describe(i: int) -> str:
            kind = int(scenario_term.scenario[i])
            text = names[kind]
            if kind == scenario_term.DROP:
                text += f", drop at clip frame {int(scenario_term.drop_frame[i] - motion.motion.clip_starts[motion.motion_ids[i]])}"
            return text

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if scenario_term is not None:
            step_count += 1
            # Episode ends come first, so a flag cleared by the reset is not reported as a fresh event below.
            for i in terminations.dones.nonzero().flatten().tolist():
                if i < 16:
                    ended = [n for n in terminations.active_terms if bool(terminations.get_term(n)[i])]
                    print(f"[SCENARIO] step {step_count}: env {i} episode ended on {ended or ['time_out']}; "
                          f"next: {describe(i)}")
            for key, label in (("drop_triggered", "force on"), ("grasp_dropped", "released"),
                               ("force_capped", "force capped, grip held")):
                now = getattr(scenario_term, key)
                for i in (now & ~seen[key]).nonzero().flatten().tolist():
                    if i < 16:
                        print(f"[SCENARIO] step {step_count}: env {i} {label}")
                seen[key] = now.clone()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if cop_overlay is not None:
        print(cop_overlay.summary())
    if foot_flat_overlay is not None:
        print(foot_flat_overlay.summary())

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
