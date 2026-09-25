from __future__ import annotations

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

        local = command.contact_point_local[command.reference_frames(0)]  # (N, P, 3), object frame
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
