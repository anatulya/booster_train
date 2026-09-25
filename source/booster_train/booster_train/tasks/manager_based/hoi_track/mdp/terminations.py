from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from booster_train.tasks.manager_based.hoi_track.mdp.commands import MotionCommand
from booster_train.tasks.manager_based.hoi_track.mdp.geometry import quat_in_heading, rotate_into_heading
from booster_train.tasks.manager_based.hoi_track.mdp.rewards import (
    _get_body_indexes,
    _object_absent,
    hand_object_normal_force,
)


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


class BadObjectPos(ManagerTermBase):
    r"""The object has been away from where the robot should be holding it for too many consecutive steps.

    Both sides are object-minus-anchor, each rotated into its own yaw-only frame::

        rel_now = R_heading(robot_anchor)^T (p_object      - p_robot_anchor)
        rel_ref = R_heading(ref_anchor)^T   (p_object_ref  - p_ref_anchor)
        bad     = ||rel_now - rel_ref|| > threshold

    so the robot's global position and yaw cancel and what is left is "is the box where you should be holding
    it". An earlier world-frame version fired whenever the *robot* drifted while carrying the object correctly,
    which made it the dominant termination at 40.6% of episodes while the object's mean position error
    (0.183 m) was indistinguishable from the robot's own anchor error (0.182 m).

    Heading frames rather than full orientation, matching ``object_pos_b``: the trunk pitches 40 degrees or
    more during a pick-up, and rotating the comparison by that pitch would manufacture error out of a posture
    difference that :class:`LostContact` already covers.

    The consecutive-step counter, zeroed on the falling edge exactly as in :class:`LostContact`, is what makes
    this survivable at reset. Envs are reset onto a random frame with the object written onto its reference
    pose, and 31% of frames of ``sub1_suitcase_029`` have the box more than 5 cm off the ground -- up to
    0.64 m, with the reference already demanding a two-handed grip. Without the counter a reset into the
    lift-off had roughly 16 control steps before free fall breached the bound, which killed the episode before
    the policy had done anything wrong; that section (bin 1, frames 47-94) was the worst-sampled bin on 100% of
    25k iterations. The counter buys another ``max_steps`` to close the hands and recover.

    Global object accuracy is still paid for by ``motion_global_object_position_error_exp`` (weight 10), which
    is the right place for it: a gradient, not a kill switch.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.bad_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.bad_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        threshold: float,
        max_steps: int = 25,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        rel_now = rotate_into_heading(
            command.robot_anchor_quat_w, command.robot_object_pos_w - command.robot_anchor_pos_w
        )
        rel_ref = rotate_into_heading(command.anchor_quat_w, command.object_pos_w - command.anchor_pos_w)
        bad = torch.norm(rel_now - rel_ref, dim=1) > threshold
        absent = _object_absent(env)
        if absent is not None:
            bad &= ~absent

        self.bad_steps = torch.where(bad, self.bad_steps + 1, torch.zeros_like(self.bad_steps))
        return self.bad_steps > max_steps


class BadObjectOri(ManagerTermBase):
    r"""The object has been badly mis-oriented relative to the robot for too many consecutive steps.

    The orientation counterpart of :class:`BadObjectPos`, built the same way: each side's object orientation is
    expressed in its own yaw-only frame, so the robot's heading drift cancels and what is left is how far the
    box is tipped relative to how the reference holds it::

        rel_now = quat_in_heading(robot_anchor_quat_w, robot_object_quat_w)
        rel_ref = quat_in_heading(anchor_quat_w,       object_quat_w)
        bad     = quat_error_magnitude(rel_now, rel_ref) > threshold

    Same frame convention as the ``object_ori_b`` / ``ref_object_ori_refroot`` observations, so the termination
    measures the quantity the policy is shown.

    At 1.2 rad (69 deg) the threshold clears the motion's own object rotation with room to spare: relative to
    its start in the root heading frame, the reference box turns at most 0.924 rad in ``sub1_suitcase_029`` and
    0.569 rad in ``sub15_suitcase_017``, and neither clip spends a single frame past 1.2 rad. So unlike the
    position bound -- whose 0.5 m sits *below* the reference's own 0.636 m displacement, making "never lifted
    it" indistinguishable from "displaced it" -- this one cannot fire on a policy that merely fails to rotate
    the box. It only catches genuine tipping.

    Consecutive-step counter and falling-edge reset exactly as in :class:`BadObjectPos`.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.bad_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.bad_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        threshold: float,
        max_steps: int = 25,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        rel_now = quat_in_heading(command.robot_anchor_quat_w, command.robot_object_quat_w)
        rel_ref = quat_in_heading(command.anchor_quat_w, command.object_quat_w)
        bad = math_utils.quat_error_magnitude(rel_now, rel_ref) > threshold
        absent = _object_absent(env)
        if absent is not None:
            bad &= ~absent

        self.bad_steps = torch.where(bad, self.bad_steps + 1, torch.zeros_like(self.bad_steps))
        return self.bad_steps > max_steps


class LostContact(ManagerTermBase):
    r"""A hand has been off its contact point for too many consecutive steps while the reference wants contact.

    Per hand :math:`i`, with :math:`c_i` the reference contact label:

    .. math::
        \text{lost}_i = c_i \land \big((F_{n,i} < F_{lost}) \lor (\|p_{palm,i} - p_{target,i}\| > d_{lost})\big)

    Either condition alone is enough, matching HDMI's ``cum_lost_contact_steps``, which builds
    ``in_contact = contact_pos & contact_frc`` and fires on its negation. A hand parked on the contact point
    without pressing, or pressing somewhere other than the contact point, both count as lost.

    Each hand carries its own consecutive-step counter, zeroed on the falling edge of its own condition --
    including when :math:`c_i` drops to 0 between grasp windows -- and either hand exceeding ``max_steps`` ends
    the episode. Because the reset is per hand, hands alternating between lost and regained never accumulate,
    and a clip with several short windows never carries a count from one window into the next. This is a
    deliberate divergence: HDMI keeps one counter fed by ``.any(dim=-1)``, which does accumulate across hands.

    Despite the name this tests "not in contact although the reference says it should be", whether or not
    contact was ever established in the first place.

    The 1.0 N / 0.2 m / 25-step thresholds are HDMI's (``hdmi-base.yaml:269``). The force bound stays well
    below the reward's grip threshold, so the gap between them is a deadband where a hand is neither paid in
    full nor killed -- but note that under the disjunction the distance bound now kills independently of
    force, so that deadband no longer covers a hand that is simply in the wrong place.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.lost_steps = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.lost_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        contact_sensor_names: list[str],
        force_threshold: float = 1.0,
        distance_threshold: float = 0.2,
        max_steps: int = 25,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)

        local = command.contact_point_local[command.reference_frames(0)]  # (N, P, 3), object frame
        quat = command.robot_object_quat_w[:, None, :].expand(-1, local.shape[1], -1)
        target = command.robot_object_pos_w[:, None, :] + math_utils.quat_apply(quat, local)
        dist = torch.norm(command.robot_palm_pos_w - target, dim=-1)  # (N, P)

        force = hand_object_normal_force(env, contact_sensor_names, reduce="last")
        lost = (command.ref_contact > 0.5) & ((force < force_threshold) | (dist > distance_threshold))
        absent = _object_absent(env)
        if absent is not None:
            lost &= ~absent[:, None]

        self.lost_steps = torch.where(lost, self.lost_steps + 1, torch.zeros_like(self.lost_steps))
        return (self.lost_steps > max_steps).any(dim=-1)
