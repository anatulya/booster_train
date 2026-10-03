from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude, yaw_quat

from booster_train.tasks.manager_based.hoi_track.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_adaptive_sigma(env, key: str | float, error: Union[float, torch.Tensor]):
    if isinstance(key, float):
        return key
    sigma_update_rate = 0.9
    if not hasattr(env, 'reward_sigmas_ema'):
        env.reward_sigmas_ema = {}
        env.reward_sigmas = {}

    env.reward_sigmas_ema[key] = (
        sigma_update_rate * env.reward_sigmas_ema.get(key, torch.tensor([100.], device=env.device)) + (1 - sigma_update_rate) * error
    )
    env.reward_sigmas[key] = torch.minimum(env.reward_sigmas_ema[key], env.reward_sigmas.get(key, torch.tensor([100.], device=env.device))).clip(min=1e-8)
    return torch.sqrt(env.reward_sigmas[key])


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float | str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
        env: ManagerBasedRLEnv, command_name: str, std: float | str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_yaw_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    """The heading half of :func:`motion_relative_body_orientation_error_exp`: yaw error only, tilt ignored.

    Paired with :func:`motion_relative_body_tilt_error_exp` so the two halves can be weighted separately. The
    split is clean because ``body_quat_relative_w`` aligns the reference to the robot by yaw alone, so it still
    carries the reference's own tilt for the tilt half to compare against.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            yaw_quat(command.body_quat_relative_w[:, body_indexes]), yaw_quat(command.robot_body_quat_w[:, body_indexes])
        )
        ** 2
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_tilt_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    """The tilt half: the angle between gravity seen from the reference body and from the robot body.

    Gravity in a body's frame does not change under a yaw of that body, so this is blind to heading by
    construction and does not double-count :func:`motion_relative_body_yaw_error_exp`.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    ref_quat = command.body_quat_relative_w[:, body_indexes]
    gravity = command.robot.data.GRAVITY_VEC_W[:, None, :].expand(-1, len(body_indexes), -1)
    g_ref = quat_apply_inverse(ref_quat, gravity)
    g_robot = quat_apply_inverse(command.robot_body_quat_w[:, body_indexes], gravity)
    angle = torch.atan2(torch.norm(torch.cross(g_ref, g_robot, dim=-1), dim=-1), torch.sum(g_ref * g_robot, dim=-1))
    error = (angle**2).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def feet_stance_time(
        env: ManagerBasedRLEnv, asset_name: str, feet_names: list[str], vel_threshold: float, desired_time: float
) -> torch.Tensor:
    if not hasattr(env, '_buf_feet_stance_time'):
        env._buf_feet_stance_time = torch.zeros(env.num_envs, 2, device=env.device)

    robot = env.scene.articulations[asset_name]
    feet_indexes = [robot.body_names.index(name) for name in feet_names]

    stance = robot.data.body_link_lin_vel_w[:, feet_indexes].norm(dim=-1) < vel_threshold

    first_slide = (env._buf_feet_stance_time > 0.) * (~stance)
    rew_stanceTime = torch.sum((env._buf_feet_stance_time - desired_time).clip(max=0.) * first_slide, dim=1)

    env._buf_feet_stance_time += env.step_dt
    env._buf_feet_stance_time *= stance
    return rew_stanceTime


##
# Object interaction.
#
# These connect the object to the reward for the first time: the body-tracking terms above are all re-anchored
# onto the robot's own trunk, so nothing in them asks the robot to actually touch anything.
##


def hand_object_normal_force(
    env: ManagerBasedRLEnv, contact_sensor_names: list[str], reduce: str = "last"
) -> torch.Tensor:
    """Normal contact force per hand against the filtered object, (N, len(contact_sensor_names)).

    ``force_matrix_w`` is already the *normal* force -- PhysX reports friction separately, in the
    ``friction_forces_w`` field -- so this is F_n directly, with no surface-normal approximation.

    One sensor per hand, because PhysX filtered contact reporting is one body to many targets. The sensor
    covers the *whole* body: ``get_contact_force_matrix`` returns one vector per ``(body, filter)`` pair,
    summing every contact point on the link, and ``*_hand_link`` has exactly one ``<collision>``. That link
    hangs off ``*_Elbow_Yaw`` and its mesh (``*_Arm_4.STL``) runs 0.258 m -- the full forearm plus hand stub,
    with the palm keypoint offset ~0.2 m out to its tip. So a single sensor here is the equivalent of HDMI
    summing force vectors across the several links its G1 end-effector spans.

    ``reduce`` picks how the physics substeps of one control step are collapsed. ``scene.update()`` runs
    inside the decimation loop, so the history holds one entry per substep, newest first:

    - ``"last"`` for the reward and termination: ``history[:, 0]`` is written from ``force_matrix_w``, so this
      is exactly the quantity HDMI thresholds. The two reducers share a mean; what differs is variance, and a
      threshold well above the typical force is sensitive to precisely that -- averaging suppresses the peaks
      that would cross it.
    - ``"mean"`` averages the whole control step. Lower variance and arguably the better summary, since PhysX
      already reports impulse/dt per substep, but it is not what the HDMI constants were tuned against.
    - ``"peak"`` for a presence flag, where catching a brief touch is the point.
    """
    out = []
    for name in contact_sensor_names:
        history = env.scene.sensors[name].data.force_matrix_w_history  # (N, H, 1, 1, 3), newest first
        assert history.shape[1] >= env.cfg.decimation, (
            f"Contact sensor '{name}' needs history_length >= decimation ({env.cfg.decimation})."
        )
        magnitude = torch.norm(history[:, : env.cfg.decimation, 0, 0], dim=-1)  # (N, decimation)
        if reduce == "last":
            out.append(magnitude[:, 0])
        elif reduce == "mean":
            out.append(magnitude.mean(dim=1))
        else:
            out.append(magnitude.amax(dim=1))
    return torch.stack(out, dim=1)


def _hand_object_contact_state(
    env: ManagerBasedRLEnv, contact_sensor_names: list[str], force_threshold: float, reduce: str = "peak"
) -> torch.Tensor:
    """0/1 contact per sensor, ordered like ``contact_sensor_names``."""
    return (hand_object_normal_force(env, contact_sensor_names, reduce) > force_threshold).float()


def _object_absent(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """``env.object_absent`` from :class:`~.events.ObjectScenario`, or None when that term is not configured.

    True where the object has been taken away on purpose: a forced drop the hands have actually let go of, or
    an episode dealt without the object. Object rewards and object terminations mean nothing there.
    """
    return getattr(env, "object_absent", None)


def _mask_absent(env: ManagerBasedRLEnv, reward: torch.Tensor) -> torch.Tensor:
    absent = _object_absent(env)
    return reward if absent is None else reward * (~absent).float()


def is_terminated_penalized(
    env: ManagerBasedRLEnv, absent_penalty_terms: tuple[str, ...] = ("anchor_pos", "anchor_ori")
) -> torch.Tensor:
    """``is_terminated``, except that where the object is absent only genuine falls are penalised.

    With the object gone, object terminations cannot fire (they are masked), and ``ee_body_pos`` -- which
    also checks hand height -- may trip on hands that have nothing left to hold while the robot stands fine,
    so it ends the episode without the penalty. Episodes that still have the object are penalised as before.
    """
    terminated = env.termination_manager.terminated
    absent = _object_absent(env)
    if absent is None or not absent.any():
        return terminated.float()
    fell = torch.zeros_like(terminated)
    for name in absent_penalty_terms:
        fell |= env.termination_manager.get_term(name)
    return torch.where(absent, fell, terminated).float()


def motion_global_object_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str
) -> torch.Tensor:
    """World-frame object position tracking; the mirror of ``motion_global_anchor_position_error_exp``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.object_pos_w - command.robot_object_pos_w), dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return _mask_absent(env, torch.exp(-error / std**2))


def motion_global_object_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str
) -> torch.Tensor:
    """World-frame object orientation tracking; the mirror of ``motion_global_anchor_orientation_error_exp``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.object_quat_w, command.robot_object_quat_w) ** 2
    std = _get_adaptive_sigma(env, std, error.mean())
    return _mask_absent(env, torch.exp(-error / std**2))


def motion_global_object_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str
) -> torch.Tensor:
    """World-frame object linear velocity tracking; the object mirror of the per-body velocity term.

    The reference box is stationary for most of a clip (median speed 0.006 m/s over ``sub1_suitcase_029``,
    p90 0.43, peak 1.7), so this is near-free outside the lift and only bites while the box is actually
    moving. That asymmetry is the point: it shapes *how* the box is carried rather than adding a constant
    offset, and a dropped box reaching 1-2 m/s in free fall scores essentially zero.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.object_lin_vel_w - command.robot_object_lin_vel_w), dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return _mask_absent(env, torch.exp(-error / std**2))


class HandObjectInteraction(ManagerTermBase):
    r"""Pays a hand for being on its contact point *and* pressing on the object.

    Per hand :math:`i`, with :math:`c_i` the reference contact label:

    .. math::
        r_{int,i} &= \exp(-d_i / \sigma_p),\quad d_i = \|p_{palm,i} - p_{target,i}\| \\
        r_{F,i}   &= \exp(\min(F_{n,i} - F_{thr}, 0) / \sigma_F) \\
        r_i       &= c_i\,g\,(r_{int,i}\,r_{F,i}) + (1 - c_i)

    and the reward is the mean over hands. The formulation and every default is HDMI's ``eef_contact_exp``
    (``cfg/task/base/hdmi-base.yaml:220``).

    The target rides the *actual* object: it is the reference contact point carried onto wherever the object
    currently is, so if the object gets knocked out of place the target follows it instead of pointing at where
    the reference expected it to be.

    ``r_F`` saturates at 1.0 once the force reaches ``force_threshold``, so that threshold stays the definition
    of a successful grip, but it decays smoothly below rather than switching off -- a hand at zero force still
    scores ``exp(-10/40) = 0.779``. That matters: with a hard indicator, ``r_int`` would be multiplied by zero
    for the entire approach and the position term could never teach the hand to close the last few centimetres.
    At these widths ``r_F`` spans only 0.779 to 1.0, so force is a nudge and position does the shaping;
    :class:`~.terminations.LostContact` is what actually enforces contact.

    Where the reference reports no contact the term is treated as already satisfied and contributes 1, so it
    applies no pressure outside the grasp windows. ``gain`` is what makes that survivable as a design: without
    it, a single weight sets both the grasping gradient (``weight * gain``, on gated hands) and that flat
    constant (``weight``, everywhere else), and the constant dominates -- measured at 25k iterations, 9.24 of
    the 10.05 earned was the constant and only 0.83 was grasping. ``gain`` scales only the part the policy can
    move.

    ``Interaction/*`` logs the *un-gained* sub-terms, so those curves stay comparable across a gain change.
    """

    TERMS = ("r_int", "r_F", "dist", "contact_frac", "force")

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self.TERMS}
        self.counts = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log each sub-term's episode mean over *gated* frames only.

        The product collapses two very different failures into one number -- a palm that never arrives and a
        palm that arrives but never presses both drive the reward down -- so the sub-terms are what make a run
        readable. They are averaged over frames where the reference asks for contact, because elsewhere the
        term is a constant 1 and would just dilute the average.
        """
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})
        counts = self.counts[ids]
        # Average only over episodes that actually had gated frames. Counting an episode that never entered a
        # grasp window as a zero would drag every sub-term down by the same factor, which makes ``dist`` read
        # far lower than the real gated-mean distance and stops the terms being comparable to each other.
        seen = counts > 0
        for name in self.TERMS:
            if seen.any():
                log[f"Interaction/{name}"] = (self.sums[name][ids][seen] / counts[seen]).mean()
            self.sums[name][ids] = 0.0
        self.counts[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        contact_sensor_names: list[str],
        sigma_p: float = 0.3,
        sigma_f: float = 40.0,
        force_threshold: float = 10.0,
        gain: float = 5.0,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)

        local = command.contact_point_local_scaled(command.reference_frames(0))  # (N, P, 3), object frame
        quat = command.robot_object_quat_w[:, None, :].expand(-1, local.shape[1], -1)
        target = command.robot_object_pos_w[:, None, :] + quat_apply(quat, local)
        dist = torch.norm(command.robot_palm_pos_w - target, dim=-1)  # (N, P); a norm, so frame-free

        force = hand_object_normal_force(env, contact_sensor_names, reduce="last")
        r_int = torch.exp(-dist / sigma_p)
        r_f = torch.exp(torch.clamp(force - force_threshold, max=0.0) / sigma_f)

        gate = command.ref_contact  # (N, P), 0/1 from the capture
        absent = _object_absent(env)
        if absent is not None:
            # No object, nothing to grasp: score it like a frame outside the grasp windows.
            gate = gate * (~absent).float()[:, None]
        reward = (gate * (r_int * r_f) * gain + (1.0 - gate)).mean(dim=-1)

        # Episode means over the frames the gate is actually open. Deliberately un-gained, so a gain change
        # does not move these curves: they report behaviour, not the reward's scale. ``force`` is in Newtons,
        # which makes it the one sub-term comparable across a ``force_threshold`` change too.
        active = gate.sum(dim=-1)
        weighted = lambda value: (value * gate).sum(dim=-1)  # noqa: E731
        self.sums["r_int"] += weighted(r_int)
        self.sums["r_F"] += weighted(r_f)
        self.sums["dist"] += weighted(dist)
        self.sums["contact_frac"] += weighted((force > force_threshold).float())
        self.sums["force"] += weighted(force)
        self.counts += active
        return reward


# -- regularization (ULTRA) --


def joint_energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mechanical power, ``sum(|tau * qdot|)``, on the torque the joints actually applied."""
    asset = env.scene[asset_cfg.name]
    power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(power), dim=1)


def joint_torque_limit_ratio(
    env: ManagerBasedRLEnv, soft_ratio: float = 0.95, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """``sum(max(0, |tau| / tau_lim - soft_ratio))``, on the *computed* (pre-clip) PD torque.

    Not ``applied_torque``: the Booster actuators clip that to the effort limit (lower still at speed), so its
    ratio can never pass 1.0 and the term would cap at ``1 - soft_ratio`` per joint. The computed torque keeps
    growing with how hard the policy asks past saturation, which is the gradient worth having.
    """
    asset = env.scene[asset_cfg.name]
    tau = asset.data.computed_torque[:, asset_cfg.joint_ids]
    tau_lim = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(tau.abs() / tau_lim - soft_ratio, min=0.0), dim=1)


class _StateChangeL2(ManagerTermBase):
    """``||x_t - x_{t-1}||^2`` between consecutive env steps, for a quantity ``_read`` returns.

    The previous value is taken on the first step after a reset rather than in :meth:`reset`, because the reward
    manager resets before the new state has been stepped into ``asset.data``; that first step scores zero, so the
    reset teleport is never penalised.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev: torch.Tensor | None = None
        self.fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.fresh[slice(None) if env_ids is None else env_ids] = True

    def _read(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        x = self._read(env, asset_cfg)
        if self.prev is None:
            self.prev = x.clone()
        self.prev[self.fresh] = x[self.fresh]
        penalty = torch.sum(torch.square(x - self.prev), dim=1)
        self.prev[:] = x
        self.fresh[:] = False
        return penalty


class JointVelChangeL2(_StateChangeL2):
    """``||qdot_t - qdot_{t-1}||^2`` per control step."""

    def _read(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


class BaseAngVelChangeL2(_StateChangeL2):
    """``||omega_t - omega_{t-1}||^2`` per control step, in the base frame -- what the IMU measures."""

    def _read(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        return env.scene[asset_cfg.name].data.root_ang_vel_b






def feet_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """``sum ||g_foot_xy||^2``: foot tilt from gravity, from the gravity vector rotated into each foot's frame.

    Ungated, so it applies in swing as well as stance, the same form ULTRA uses.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, F, 4)
    gravity = asset.data.GRAVITY_VEC_W[:, None, :].expand(-1, quat.shape[1], -1)
    gravity_foot = quat_apply_inverse(quat, gravity)
    return torch.sum(torch.square(gravity_foot[..., :2]), dim=(1, 2))


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ratio: float = 4.0) -> torch.Tensor:
    """Number of feet whose contact force is mostly horizontal, ``||f_xy|| > ratio * |f_z|`` -- a foot kicking a
    step edge or scraping sideways rather than bearing load. A foot in the air has zero force and scores 0.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (N, F, 3)
    stumbling = torch.norm(force[..., :2], dim=-1) > ratio * torch.abs(force[..., 2])
    return torch.sum(stumbling.float(), dim=1)






def _debounce_stance(raw: np.ndarray, starts: list[int], lengths: list[int], min_run: int) -> np.ndarray:
    """Absorb stance/swing runs shorter than ``min_run`` into the run before them, per clip.

    Same logic as ``scripts/rsl_rl/play.py``'s ``CoPOverlay._debounce`` (a separate copy there, not a caller of
    this), so the render-time overlay and this training-time term build the reference-stance table the same way.
    """
    out = raw.copy()
    for start, length in zip(starts, lengths):
        seg = out[start : start + length]
        i = 0
        while i < length:
            j = i
            while j < length and seg[j] == seg[i]:
                j += 1
            if j - i < min_run and i > 0:
                seg[i:j] = seg[i - 1]
            i = j
    return out


STANCE_HEIGHT = 0.04  # m above that foot's lowest reference ankle height in the clip
STANCE_SPEED = 0.25  # m/s, reference foot horizontal speed
STANCE_MIN_RUN = 3  # frames


def reference_stance(command: MotionCommand, foot_body_names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame reference stance for each foot, ``(T, F)`` bool, and which frames were labelled, ``(T,)`` bool.

    Frames of a clip baked with stance labels (``MotionLoader.stance``) use those. Every other frame falls back to a
    kinematic guess from the reference: ankle within ``STANCE_HEIGHT`` of that foot's lowest point in its clip and
    horizontal speed below ``STANCE_SPEED``, with runs shorter than ``STANCE_MIN_RUN`` frames absorbed (the same
    heuristic as ``scripts/rsl_rl/play.py``'s ``CoPOverlay``). Height is clip-relative because the retargeted feet
    do not sit at one height across clips.
    """
    motion = command.motion
    device = motion.body_pos_w.device
    clip_of_frame = torch.repeat_interleave(torch.arange(motion.num_clips, device=device), motion.clip_lengths)
    starts, lengths = motion.clip_starts.tolist(), motion.clip_lengths.tolist()
    labelled = (
        motion.stance_labelled
        if getattr(motion, "stance_labelled", None) is not None
        else torch.zeros(motion.time_step_total, dtype=torch.bool, device=device)
    )
    columns = []
    for name in foot_body_names:
        k = command.cfg.body_names.index(name)
        z = motion.body_pos_w[:, k, 2]
        clip_min = torch.full((motion.num_clips,), float("inf"), device=device).scatter_reduce(
            0, clip_of_frame, z, reduce="amin"
        )
        speed = torch.norm(motion.body_lin_vel_w[:, k, :2], dim=-1)
        raw = ((z - clip_min[clip_of_frame] < STANCE_HEIGHT) & (speed < STANCE_SPEED)).cpu().numpy()
        guess = torch.tensor(_debounce_stance(raw, starts, lengths, STANCE_MIN_RUN), device=device)
        if labelled.any():
            label = motion.stance[:, motion.stance_names.index(name)] > 0.5
            guess = torch.where(labelled, label, guess)
        columns.append(guess)
    return torch.stack(columns, dim=1), labelled


def _weighted_contact_agg(
    forces: torch.Tensor, points: torch.Tensor, counts: torch.Tensor, starts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Force-weighted sum of contact points and total force, per (env, body) row, from a ragged contact buffer.

    ``forces``/``points`` are the flat per-contact arrays ``contact_physx_view.get_contact_data`` returns;
    ``counts``/``starts`` say which slice of them belongs to each row. Isaac Lab's own
    ``ContactSensor._unpack_contact_buffer_data`` does the analogous *unweighted* mean for ``contact_pos_w`` with
    ``repeat_interleave``, which needs the total count on the host (a GPU sync every step); this instead maps
    every buffer slot to its row with ``searchsorted`` over the sorted row starts and masks the unused tail, so
    it never leaves the GPU. Weighted by force, so the result is a genuine centre of pressure rather than a
    plain average of the contact points.
    """
    n_rows = counts.numel()
    starts, counts = starts.long(), counts.long()
    # Empty rows can share a start with a non-empty one; sorting on (start, non-empty) puts the non-empty row
    # last among ties, which is the one searchsorted(right=True) - 1 lands on.
    order = torch.argsort(starts * 2 + (counts > 0).long())
    sorted_starts = starts[order]
    slot = torch.arange(forces.shape[0], device=counts.device)
    row_ids = order[(torch.searchsorted(sorted_starts, slot, right=True) - 1).clamp(min=0)]
    valid = (slot >= starts[row_ids]) & (slot < starts[row_ids] + counts[row_ids])
    # where, not multiply: unused slots may hold garbage, and 0 * nan is still nan.
    f = torch.where(valid, forces.view(-1).abs(), 0.0)
    p = torch.where(valid[:, None], points, 0.0)
    sum_force = torch.zeros(n_rows, device=counts.device, dtype=points.dtype).index_add_(0, row_ids, f)
    sum_force_pos = torch.zeros(n_rows, 3, device=counts.device, dtype=points.dtype).index_add_(
        0, row_ids, f[:, None] * p
    )
    return sum_force, sum_force_pos


class CoPSupportCentering(ManagerTermBase):
    r"""Penalizes the global centre of pressure for wandering off the robot's own intended base of support.

    The reference contributes exactly one thing: a per-foot in-stance label (``c_ref``, from
    :func:`reference_stance`: the clip's baked labels where it has them, a kinematic guess otherwise), deciding *which* feet
    are expected to support the robot right now. Every position below -- the support geometry and the CoP
    itself -- comes from the *simulated* robot, never from the reference body poses, so this cannot simply
    duplicate ``motion_foot_pos``/``motion_foot_ori`` (which pull individual feet onto the reference's own foot
    poses): it is blind to where the reference thinks a foot should be, and only checks whether the robot's own
    ground reaction sits under its own stance foot/feet.

    Sibling of ``scripts/rsl_rl/play.py``'s ``CoPOverlay``, which visualizes exactly this pair (support centre
    vs global CoP) at render time; the reference-stance heuristic, sole geometry and force-weighted contact
    aggregation are all identical to it on purpose, so a checkpoint's training-time numbers and its rendered
    overlay agree.

    .. math::
        p_{support} &= \frac{\sum_f c_{ref,f}\, p_{foot,f}}{\sum_f c_{ref,f} + \epsilon} \\
        p_{cop}     &= \frac{\sum_f F_{z,f}\, p_{cop,f}}{\sum_f F_{z,f} + \epsilon},\quad
                       p_{cop,f} = \frac{\sum_j F_{z,j}\, p_j}{\sum_j F_{z,j} + \epsilon} \\
        d           &= \lVert p_{cop} - p_{support} \rVert \\
        \text{penalty} &= \frac{\mathrm{clamp}(d - r_{free},\, 0)}{scale}

    Zero inside ``free_radius`` -- ordinary sway near the support centre costs nothing -- and zero whenever
    there is no signal to judge: no foot in reference stance, or negligible total ground load (airborne).

    Not gated by ``_object_absent``: this is a general balance term, not an object-interaction one -- the robot
    still needs to keep its ground reaction under itself in drop/no-object scenarios, arguably more so.
    """

    SOLE_CENTRE = (0.0215, 0.0, -0.038)  # foot frame; same as CoPOverlay.SOLE_CENTRE
    BUDGET_CHECK_EVERY = 500  # calls; the overflow check is a host sync, so it only samples

    TERMS = ("dist", "excess", "penalty", "forward_offset", "lateral_offset")

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stance: torch.Tensor | None = None  # (T, F) bool, built lazily on first call
        self._foot_body_names: tuple[str, ...] | None = None
        self._robot_foot_ids: list[int] | None = None
        self._sensor_body_ids: list[int] | None = None
        self._sensor_num_bodies: int | None = None
        self._sole_local: torch.Tensor | None = None
        self._budget_warned = False
        self._calls = 0

        self.sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self.TERMS}
        self.counts = torch.zeros(self.num_envs, device=self.device)
        self.active_count = torch.zeros(self.num_envs, device=self.device)
        self.step_count = torch.zeros(self.num_envs, device=self.device)
        self.imbalance_sum = torch.zeros(self.num_envs, device=self.device)
        self.imbalance_count = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})

        counts = self.counts[ids]
        seen = counts > 0
        for name in self.TERMS:
            if seen.any():
                log[f"CoP/{name}"] = (self.sums[name][ids][seen] / counts[seen]).mean()
            self.sums[name][ids] = 0.0
        self.counts[ids] = 0.0

        steps = self.step_count[ids]
        seen_steps = steps > 0
        if seen_steps.any():
            log["CoP/active_frac"] = (self.active_count[ids][seen_steps] / steps[seen_steps]).mean()
        self.active_count[ids] = 0.0
        self.step_count[ids] = 0.0

        imb_n = self.imbalance_count[ids]
        seen_imb = imb_n > 0
        if seen_imb.any():
            log["CoP/load_imbalance"] = (self.imbalance_sum[ids][seen_imb] / imb_n[seen_imb]).mean()
        self.imbalance_sum[ids] = 0.0
        self.imbalance_count[ids] = 0.0

    def _setup(self, command: MotionCommand, sensor: ContactSensor, foot_body_names: tuple[str, ...]) -> None:
        """One-time build of the reference-stance table and the name->index lookups. Deferred past ``__init__``
        for the same reason ``events.ObjectScenario`` defers its setup: the command term's motion data is not
        guaranteed ready at manager-construction time, only once the env has actually reset/stepped.
        """
        device = self.device
        self._stance, _ = reference_stance(command, foot_body_names)

        self._foot_body_names = foot_body_names
        self._robot_foot_ids = [self._env.scene["robot"].body_names.index(n) for n in foot_body_names]
        self._sensor_body_ids = [sensor.body_names.index(n) for n in foot_body_names]
        self._sensor_num_bodies = sensor.num_bodies
        self._sole_local = torch.tensor(self.SOLE_CENTRE, device=device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_name: str,
        foot_body_names: tuple[str, ...] = ("left_foot_link", "right_foot_link"),
        free_radius: float = 0.05,
        scale: float = 0.05,
        min_load: float = 5.0,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        if self._stance is None:
            self._setup(command, sensor, foot_body_names)

        eps = 1e-6
        c_ref = self._stance[command.time_steps].float()  # (N, F); index 0 is "left" for load_imbalance's sign

        robot = env.scene["robot"]
        foot_pos_w = robot.data.body_pos_w[:, self._robot_foot_ids]  # (N, F, 3), simulated
        foot_quat_w = robot.data.body_quat_w[:, self._robot_foot_ids]
        sole_local = self._sole_local.view(1, 1, 3).expand_as(foot_pos_w)
        p_foot_xy = (foot_pos_w + quat_apply(foot_quat_w, sole_local))[..., :2]  # (N, F, 2)
        p_support = (c_ref[..., None] * p_foot_xy).sum(1) / (c_ref.sum(1, keepdim=True) + eps)  # (N, 2)

        forces, points, _, _, count, start = sensor.contact_physx_view.get_contact_data(dt=env.physics_dt)
        # The contact buffer is shared by all (env, foot) rows, sized max_contact_data_count_per_prim * rows;
        # PhysX truncates once the *total* fills it, so that -- not any one row's count -- is what to check.
        if not self._budget_warned and self._calls % self.BUDGET_CHECK_EVERY == 0:
            total = int(count.sum())
            if total >= forces.shape[0]:
                print(
                    f"[WARN] CoPSupportCentering: {total} foot-ground contact points filled the {forces.shape[0]}"
                    "-slot contact buffer -- contacts may be silently truncated, biasing the CoP. Consider raising"
                    " max_contact_data_count_per_prim on 'foot_ground_contact'."
                )
                self._budget_warned = True
        self._calls += 1
        sum_force, sum_force_pos = _weighted_contact_agg(forces, points, count[:, 0], start[:, 0])

        env_offset = torch.arange(env.num_envs, device=self.device) * self._sensor_num_bodies
        Fz = torch.stack([sum_force[env_offset + bid] for bid in self._sensor_body_ids], dim=1)  # (N, F)
        sum_pos_xy = torch.stack(
            [sum_force_pos[env_offset + bid, :2] for bid in self._sensor_body_ids], dim=1
        )  # (N, F, 2)

        Fz_total = Fz.sum(1)
        p_cop = sum_pos_xy.sum(1) / (Fz_total[:, None] + eps)  # (N, 2)

        d = torch.norm(p_cop - p_support, dim=-1)
        active = (c_ref.sum(1) > 0) & (Fz_total > min_load)
        excess = torch.clamp(d - free_radius, min=0.0)
        penalty = (excess / scale) * active.float()

        # Heading-frame decomposition, for logging only: forward = toe-ward, so a systematic bias there is the
        # direct diagnostic for the hardware heel/toe-lift issue this term is meant to address.
        anchor_quat = command.robot_anchor_quat_w
        w, x, y, z = anchor_quat.unbind(-1)
        heading = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        offset = p_cop - p_support
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        forward = offset[:, 0] * cos_h + offset[:, 1] * sin_h
        lateral = -offset[:, 0] * sin_h + offset[:, 1] * cos_h

        # Masked accumulation rather than boolean indexing: indexing needs the mask's count on the host.
        double = ((c_ref[:, 0] > 0) & (c_ref[:, 1] > 0) & active).float()
        imbalance = (Fz[:, 0] - Fz[:, 1]) / (Fz_total + eps)
        self.imbalance_sum += imbalance * double
        self.imbalance_count += double

        mask = active.float()
        self.step_count += 1.0
        self.active_count += mask
        for name, value in zip(self.TERMS, (d, excess, penalty, forward, lateral)):
            self.sums[name] += value * mask
        self.counts += mask

        return penalty


# Sole corners in the foot-link frame (m), from meshes/Left_Foot.STL: the sole's flat bottom sits at z = -0.0382 and
# spans x -0.065..0.094, y -0.039..0.041. The toe corners at x 0.0995 are ~6 mm past the end of the flat face, where
# the toe starts to curl, so they read a mm or two high on a flat foot -- well inside the dead zone below. Order:
# toe-outer, toe-inner, heel-outer, heel-inner; "outer" is +y on the left foot, so the right foot mirrors y.
SOLE_CORNERS_LEFT = ((0.0995, 0.04, -0.0382), (0.0995, -0.04, -0.0382), (-0.0657, 0.04, -0.0382), (-0.0657, -0.04, -0.0382))
SOLE_CORNER_NAMES = ("toe_out", "toe_in", "heel_out", "heel_in")


def sole_corner_heights(foot_pos_w: torch.Tensor, foot_quat_w: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
    """Height above the floor (z = 0, flat ground) of each sole corner: ``(..., F, 4)`` from ``(..., F, 3/4)`` foot
    poses and ``(F, 4, 3)`` corners in each foot's own frame."""
    quat = foot_quat_w[..., None, :].expand(*foot_quat_w.shape[:-1], corners.shape[-2], 4)
    points = foot_pos_w[..., None, :] + quat_apply(quat, corners.expand(*foot_pos_w.shape[:-1], -1, -1))
    return points[..., 2]


class FootFlatStance(ManagerTermBase):
    r"""Penalizes a stance foot that is not flat on the floor: any of its four sole corners lifted.

    Per foot :math:`f`, with sole-corner heights :math:`h_{f,c}` of the *simulated* foot and reference stance
    :math:`s_f` (:func:`reference_stance`):

    .. math::
        e_{f,c} &= \max(h_{f,c} - tol,\, 0) \\
        \text{score}_f &= \exp\Big(-\sum_c e_{f,c}^2 / \sigma^2\Big) \\
        \text{penalty} &= \frac{\sum_f s_f\,(1 - \text{score}_f)}{\max(\sum_f s_f,\, 1)}

    so it is 0 for flat stance feet, approaches 1 for a stance foot up on its heel, toe or edge, and is exactly 0
    on swing feet and in flight -- the stance gate is the whole point, so no separate gain is needed: the term's
    weight *is* the stance-frame gain, and unlike ``HandObjectInteraction`` there is no constant outside the gate
    for a gain to be measured against.

    The reference is deliberately not the target. Its own feet are rolled onto the inner edge (outer corners
    15-25 mm up) and hover ~1 cm on the standing frames, so the reference itself scores 0.1-0.3 here; this term asks
    for flat feet anyway, which is why ``motion_foot_ori`` is turned down alongside it.

    Not gated by ``_object_absent``: like ``CoPSupportCentering``, it is about the feet, not the object.
    """

    TERMS = ("score", "penalty") + tuple(f"h_{name}" for name in SOLE_CORNER_NAMES)

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stance: torch.Tensor | None = None
        self._robot_foot_ids: list[int] | None = None
        self._corners: torch.Tensor | None = None
        self.sums = {name: torch.zeros(self.num_envs, device=self.device) for name in self.TERMS}
        self.counts = torch.zeros(self.num_envs, device=self.device)  # stance foot-steps
        self.steps = torch.zeros(self.num_envs, device=self.device)
        self.stance_steps = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Log episode means over stance foot-steps only; swing feet are not judged and would just dilute them."""
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})
        counts = self.counts[ids]
        seen = counts > 0
        for name in self.TERMS:
            if seen.any():
                log[f"FootFlat/{name}"] = (self.sums[name][ids][seen] / counts[seen]).mean()
            self.sums[name][ids] = 0.0
        steps = self.steps[ids]
        if (steps > 0).any():
            log["FootFlat/stance_frac"] = (self.stance_steps[ids][steps > 0] / steps[steps > 0]).mean()
        self.counts[ids] = 0.0
        self.steps[ids] = 0.0
        self.stance_steps[ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        foot_body_names: tuple[str, str] = ("left_foot_link", "right_foot_link"),
        tol: float = 0.004,
        sigma: float = 0.015,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        if self._stance is None:
            self._stance, labelled = reference_stance(command, foot_body_names)
            self._robot_foot_ids = [env.scene["robot"].body_names.index(n) for n in foot_body_names]
            left = torch.tensor(SOLE_CORNERS_LEFT, device=self.device)
            self._corners = torch.stack([left, left * torch.tensor([1.0, -1.0, 1.0], device=self.device)])
            print(f"[FootFlatStance] stance from baked labels on {labelled.float().mean().item() * 100:.0f}% of reference"
                  " frames, kinematic guess on the rest")

        robot = env.scene["robot"]
        h = sole_corner_heights(
            robot.data.body_pos_w[:, self._robot_foot_ids], robot.data.body_quat_w[:, self._robot_foot_ids], self._corners
        )  # (N, F, 4)
        score = torch.exp(-(torch.clamp(h - tol, min=0.0) ** 2).sum(-1) / sigma**2)  # (N, F)
        stance = self._stance[command.time_steps].float()  # (N, F)
        n_stance = stance.sum(1)
        penalty = (stance * (1.0 - score)).sum(1) / n_stance.clamp(min=1.0)

        self.steps += 1.0
        self.stance_steps += (n_stance > 0).float()
        self.counts += n_stance
        self.sums["score"] += (stance * score).sum(1)
        self.sums["penalty"] += (stance * (1.0 - score)).sum(1)
        for c, name in enumerate(SOLE_CORNER_NAMES):
            self.sums[f"h_{name}"] += (stance * h[..., c]).sum(1)
        return penalty
