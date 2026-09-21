from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude

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
    env: ManagerBasedRLEnv, contact_sensor_names: list[str], reduce: str = "mean"
) -> torch.Tensor:
    """Normal contact force per hand against the filtered object, (N, len(contact_sensor_names)).

    ``force_matrix_w`` is already the *normal* force -- PhysX reports friction separately, in the
    ``friction_forces_w`` field -- so this is F_n directly, with no surface-normal approximation.

    Each sensor must cover a single body filtered against a single target, because PhysX filtered contact
    reporting is one-to-many, so each hand needs its own sensor. ``reduce`` picks how the physics substeps of
    one control step are collapsed:

    - ``"mean"`` for the reward: a hand touching for one substep out of four should not read as a full step of
      contact, which is exactly the graze-versus-grip distinction the interaction reward exists to draw.
    - ``"peak"`` for a presence flag, where catching a brief touch is the point.
    """
    out = []
    for name in contact_sensor_names:
        history = env.scene.sensors[name].data.force_matrix_w_history  # (N, H, 1, 1, 3), newest first
        assert history.shape[1] >= env.cfg.decimation, (
            f"Contact sensor '{name}' needs history_length >= decimation ({env.cfg.decimation})."
        )
        magnitude = torch.norm(history[:, : env.cfg.decimation, 0, 0], dim=-1)  # (N, decimation)
        out.append(magnitude.mean(dim=1) if reduce == "mean" else magnitude.amax(dim=1))
    return torch.stack(out, dim=1)


def _hand_object_contact_state(
    env: ManagerBasedRLEnv, contact_sensor_names: list[str], force_threshold: float, reduce: str = "peak"
) -> torch.Tensor:
    """0/1 contact per sensor, ordered like ``contact_sensor_names``."""
    return (hand_object_normal_force(env, contact_sensor_names, reduce) > force_threshold).float()


def motion_global_object_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str
) -> torch.Tensor:
    """World-frame object position tracking; the mirror of ``motion_global_anchor_position_error_exp``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.object_pos_w - command.robot_object_pos_w), dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_object_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str
) -> torch.Tensor:
    """World-frame object orientation tracking; the mirror of ``motion_global_anchor_orientation_error_exp``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.object_quat_w, command.robot_object_quat_w) ** 2
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


class HandObjectInteraction(ManagerTermBase):
    r"""Pays a hand for being on its contact point *and* pressing on the object.

    Per hand :math:`i`, with :math:`c_i` the reference contact label:

    .. math::
        r_{int,i} &= \exp(-d_i / \sigma_p),\quad d_i = \|p_{palm,i} - p_{target,i}\| \\
        r_{F,i}   &= \exp(\min(F_{n,i} - F_{thr}, 0) / \sigma_F) \\
        r_i       &= c_i\,(r_{int,i}\,r_{F,i}) + (1 - c_i)

    and the reward is the mean over hands.

    The target rides the *actual* object: it is the reference contact point carried onto wherever the object
    currently is, so if the object gets knocked out of place the target follows it instead of pointing at where
    the reference expected it to be.

    ``r_F`` saturates at 1.0 once the force reaches ``force_threshold``, so that threshold stays the definition
    of a successful grip, but it decays smoothly below rather than switching off. That matters: with a hard
    indicator, ``r_int`` would be multiplied by zero for the entire approach and the position term could never
    teach the hand to close the last few centimetres.

    Where the reference reports no contact the term is treated as already satisfied and contributes 1, so it
    applies no pressure outside the grasp windows.
    """

    TERMS = ("r_int", "r_F", "dist", "contact_frac")

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
        sigma_p: float = 0.1,
        sigma_f: float = 1.0,
        force_threshold: float = 2.0,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)

        local = command.contact_point_local[command.reference_frames(0)]  # (N, P, 3), object frame
        quat = command.robot_object_quat_w[:, None, :].expand(-1, local.shape[1], -1)
        target = command.robot_object_pos_w[:, None, :] + quat_apply(quat, local)
        dist = torch.norm(command.robot_palm_pos_w - target, dim=-1)  # (N, P); a norm, so frame-free

        force = hand_object_normal_force(env, contact_sensor_names, reduce="mean")
        r_int = torch.exp(-dist / sigma_p)
        r_f = torch.exp(torch.clamp(force - force_threshold, max=0.0) / sigma_f)

        gate = command.ref_contact  # (N, P), 0/1 from the capture
        reward = (gate * (r_int * r_f) + (1.0 - gate)).mean(dim=-1)

        # Episode means over the frames the gate is actually open.
        active = gate.sum(dim=-1)
        weighted = lambda value: (value * gate).sum(dim=-1)  # noqa: E731
        self.sums["r_int"] += weighted(r_int)
        self.sums["r_F"] += weighted(r_f)
        self.sums["dist"] += weighted(dist)
        self.sums["contact_frac"] += weighted((force > force_threshold).float())
        self.counts += active
        return reward
