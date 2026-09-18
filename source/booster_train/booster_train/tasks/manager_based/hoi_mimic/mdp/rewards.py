"""ULTRA teacher reward (arXiv:2603.03279, Tables B and C) for K1 hand-object tracking.

``r = r_track + sum_i w_i r_i^smooth`` where ``r_track`` is the product of exponential tracking terms
(:class:`UltraTrackingReward`) and the smoothness/regularization terms below are additive ``RewTerm``s.
Where ULTRA leaves a detail unspecified, the InterMimic implementation is followed (noted per term).
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

from booster_train.tasks.manager_based.hoi_mimic.mdp.commands import MotionCommand
from booster_train.tasks.manager_based.hoi_mimic.mdp.geometry import interaction_offsets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _hand_object_contact_state(
    env: ManagerBasedRLEnv, contact_sensor_names: list[str], force_threshold: float
) -> torch.Tensor:
    """0/1 contact per sensor, ordered like ``contact_sensor_names``.

    Each sensor must cover a single body filtered against a single target -- PhysX filtered contact
    reporting is one-to-many, so each hand needs its own sensor. Contact is the peak force over every
    physics substep of the last control step, so each sensor needs ``history_length >= decimation``;
    reading only the final substep misses touches that land mid-decimation.
    """
    peak = []
    for name in contact_sensor_names:
        history = env.scene.sensors[name].data.force_matrix_w_history  # (N, H, 1, 1, 3), newest first
        assert history.shape[1] >= env.cfg.decimation, (
            f"Contact sensor '{name}' needs history_length >= decimation ({env.cfg.decimation})."
        )
        peak.append(torch.norm(history[:, : env.cfg.decimation, 0, 0], dim=-1).amax(dim=1))
    return (torch.stack(peak, dim=1) > force_threshold).float()


def interaction_error(robot_offsets: torch.Tensor, ref_offsets: torch.Tensor) -> torch.Tensor:
    """InterMimic ``compute_ig_reward`` error: sum_ij |d - d_ref|^2 (w_sim + w_ref) * 0.5, with
    w = 1 / clamp(|d|^2, 0.01) normalized over all (i, j). Offsets are (N, K, M, 3)."""

    def weights(offsets):
        w = 1.0 / torch.clamp(offsets.square().sum(dim=-1), min=0.01)
        return w / w.sum(dim=(-1, -2), keepdim=True)

    err = (robot_offsets - ref_offsets).square().sum(dim=-1) * (weights(robot_offsets) + weights(ref_offsets))
    return 0.5 * err.sum(dim=(-1, -2))


class UltraTrackingReward(ManagerTermBase):
    """``r_track = r_p * r_r * r_pv * r_rv * r_op * r_or * r_opv * r_int * r_ct * r_eng`` (ULTRA Table B).

    Links and the object are compared against the re-anchored reference (heading/xy aligned to the robot),
    so squared norms equal their heading-frame values. Per-sub-term episode means are logged as
    ``Tracking/<term>``.
    """

    TERMS = ("r_p", "r_r", "r_pv", "r_rv", "r_op", "r_or", "r_opv", "r_int", "r_ct", "r_eng")

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        self.prev_joint_vel = self.robot.data.joint_vel.clone()
        self.sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self.TERMS}
        self.steps = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})
        steps = self.steps[ids].clamp(min=1.0)
        for name in self.TERMS:
            log[f"Tracking/{name}"] = (self.sums[name][ids] / steps).mean()
            self.sums[name][ids] = 0.0
        self.steps[ids] = 0.0
        self.prev_joint_vel[ids] = self.robot.data.joint_vel[ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        contact_sensor_names: list[str],
        force_threshold: float,
        k_p: float = 10.0,
        k_r: float = 5.0,
        k_pv: float = 0.1,
        k_rv: float = 0.001,
        k_op: float = 5.0,
        k_or: float = 0.5,
        k_opv: float = 0.1,
        k_int: float = 20.0,
        k_ct: float = 5.0,
        k_eng: float = 5.0e-5,
        huber_delta: float = 1.0,
        eng_grace_steps: int = 2,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        ref = command.reference(0)
        robot = self.robot.data
        obj = command.object.data

        err_p = (ref["body_pos"] - command.robot_body_pos_w).square().sum(dim=-1).mean(dim=-1)
        err_r = (ref["joint_pos"] - robot.joint_pos).square().mean(dim=-1)
        err_pv = (ref["body_lin_vel"] - command.robot_body_lin_vel_w).square().sum(dim=-1).mean(dim=-1)
        err_rv = (ref["joint_vel"] - robot.joint_vel).square().mean(dim=-1)
        err_op = (ref["object_pos"] - obj.root_pos_w).square().sum(dim=-1)
        angle = quat_error_magnitude(ref["object_quat"], obj.root_quat_w)
        err_or = torch.nn.functional.huber_loss(angle, torch.zeros_like(angle), reduction="none", delta=huber_delta)
        err_opv = (ref["object_lin_vel"] - obj.root_lin_vel_w).square().sum(dim=-1)
        err_int = interaction_error(
            interaction_offsets(command.robot_palm_pos_w, command.robot_object_points_w),
            interaction_offsets(ref["palm_pos"], command.object_points_w(ref["object_pos"], ref["object_quat"])),
        )
        contact = _hand_object_contact_state(env, contact_sensor_names, force_threshold)
        err_ct = (contact - ref["contact"]).abs().mean(dim=-1)
        # InterMimic energy term: joint acceleration from consecutive joint velocities, off right after reset.
        joint_acc = (robot.joint_vel - self.prev_joint_vel) / env.step_dt
        active = (env.episode_length_buf > eng_grace_steps).float()
        err_eng = joint_acc.square().mean(dim=-1) * active
        self.prev_joint_vel[:] = robot.joint_vel

        terms = {
            "r_p": torch.exp(-k_p * err_p),
            "r_r": torch.exp(-k_r * err_r),
            "r_pv": torch.exp(-k_pv * err_pv),
            "r_rv": torch.exp(-k_rv * err_rv),
            "r_op": torch.exp(-k_op * err_op),
            "r_or": torch.exp(-k_or * err_or),
            "r_opv": torch.exp(-k_opv * err_opv),
            "r_int": torch.exp(-k_int * err_int),
            "r_ct": torch.exp(-k_ct * err_ct),
            "r_eng": torch.exp(-k_eng * err_eng),
        }
        reward = torch.ones(self.num_envs, device=self.device)
        for name, value in terms.items():
            reward = reward * value
            self.sums[name] += value
        self.steps += 1.0
        return reward


##
# Smoothness and regularization (ULTRA Table C), additive
##


def base_lin_vel_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_lin_vel_b.norm(dim=-1)


def base_ang_vel_sq(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_ang_vel_b.square().sum(dim=-1)


def action_rate_norm(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (env.action_manager.action - env.action_manager.prev_action).norm(dim=-1)


class _PreviousValuePenalty(ManagerTermBase):
    """Squared change of a per-env quantity since the previous step; zero on the first step after reset."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        self.prev = self._value().clone()

    def _value(self) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.prev[ids] = self._value()[ids]

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        value = self._value()
        change = (value - self.prev).square().sum(dim=-1)
        self.prev[:] = value
        return change


class joint_vel_change_sq(_PreviousValuePenalty):
    def _value(self) -> torch.Tensor:
        return self.robot.data.joint_vel


class base_ang_vel_change_sq(_PreviousValuePenalty):
    def _value(self) -> torch.Tensor:
        return self.robot.data.root_ang_vel_b


def torque_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.applied_torque.norm(dim=-1)


def energy_norm(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return (data.applied_torque * data.joint_vel).norm(dim=-1)


def torque_limit_ratio(
    env: ManagerBasedRLEnv, soft_ratio: float = 0.95, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    ratio = data.applied_torque.abs() / data.joint_effort_limits.clamp(min=1e-6)
    return torch.clamp(ratio - soft_ratio, min=0.0).sum(dim=-1)


def feet_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """sum over feet of |g_xy| in the foot frame (0 when the sole is level)."""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity = asset.data.GRAVITY_VEC_W[:, None, :].expand(-1, quat.shape[1], -1)
    return quat_apply_inverse(quat, gravity)[..., :2].norm(dim=-1).sum(dim=-1)


def _feet_in_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]  # (N, H, F, 3)
    return forces.norm(dim=-1).amax(dim=1) > threshold


def foot_slip(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return (torch.sqrt(speed) * _feet_in_contact(env, sensor_cfg, threshold)).sum(dim=-1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.any(forces[..., :2].norm(dim=-1) > 4.0 * forces[..., 2].abs(), dim=-1).float()


def body_pair_distance_outside(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float, max_dist: float
) -> torch.Tensor:
    """Horizontal distance between two bodies, penalized only outside [min_dist, max_dist]."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    dist = (pos[:, 0] - pos[:, 1]).norm(dim=-1)
    return torch.clamp(min_dist - dist, min=0.0) + torch.clamp(dist - max_dist, min=0.0)


def stand_on_feet(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """1 when the reference stands on both feet but the robot does not have both feet in contact."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    sim_stand = torch.all(_feet_in_contact(env, sensor_cfg, threshold), dim=-1)
    return (command.reference(0)["stand"] & ~sim_stand).float()


def swing_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    w_height: float = 10.0,
    w_contact: float = 1.0,
    threshold: float = 1.0,
) -> torch.Tensor:
    """For reference swing feet (above their standing height + tol): penalize the robot foot being lower than
    the reference foot and touching the ground. ULTRA's exact form is unspecified; this follows its terms
    ``(w_h * h + w_c * c) * g_swing``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_z = command.reference(0)["body_pos"][:, command.foot_indexes, 2]
    sim_z = command.robot_body_pos_w[:, command.foot_indexes, 2]
    swing = (ref_z > command.ref_foot_stand_height + command.cfg.stand_height_tol).float()
    height_term = torch.clamp(ref_z - sim_z, min=0.0)
    contact_term = _feet_in_contact(env, sensor_cfg, threshold).float()
    return ((w_height * height_term + w_contact * contact_term) * swing).sum(dim=-1)
