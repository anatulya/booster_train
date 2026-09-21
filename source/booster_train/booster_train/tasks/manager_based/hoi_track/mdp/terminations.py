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
from booster_train.tasks.manager_based.hoi_track.mdp.geometry import rotate_into_heading
from booster_train.tasks.manager_based.hoi_track.mdp.rewards import _get_body_indexes, hand_object_normal_force


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


def bad_object_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """The object is not where the robot should be holding it, measured in the robot's own heading frame.

    Both sides are object-minus-anchor, each rotated into its own yaw-only frame::

        rel_now = R_heading(robot_anchor)^T (p_object      - p_robot_anchor)
        rel_ref = R_heading(ref_anchor)^T   (p_object_ref  - p_ref_anchor)
        fire if ||rel_now - rel_ref|| > threshold

    so the robot's global position and yaw cancel and what is left is "is the box where you should be holding
    it". This replaces an earlier world-frame check, which measured the object against a world-anchored
    reference and so fired whenever the *robot* drifted while carrying the object correctly. That made it the
    dominant termination -- 40.6% of episodes at iteration 5.7k -- while the object's mean position error
    (0.183 m) was indistinguishable from the robot's own anchor error (0.182 m), i.e. the object was riding
    along with root drift rather than being dropped.

    It also restores consistency with ``bad_anchor_pos_z_only``, which tolerates horizontal robot drift by
    design. Global object accuracy is still paid for by ``motion_global_object_position_error_exp`` (weight
    10), which is the right place for it: a gradient, not a kill switch.

    Heading frames rather than full orientation, matching ``object_pos_b``: the trunk pitches 40 degrees or
    more during a pick-up, and rotating the comparison by that pitch would manufacture error out of a posture
    difference that ``LostContact`` already covers.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    rel_now = rotate_into_heading(command.robot_anchor_quat_w, command.robot_object_pos_w - command.robot_anchor_pos_w)
    rel_ref = rotate_into_heading(command.anchor_quat_w, command.object_pos_w - command.anchor_pos_w)
    return torch.norm(rel_now - rel_ref, dim=1) > threshold


class LostContact(ManagerTermBase):
    r"""A hand has been off its contact point for too many consecutive steps while the reference wants contact.

    Per hand :math:`i`, with :math:`c_i` the reference contact label:

    .. math::
        \text{lost}_i = c_i \land (F_{n,i} < F_{lost}) \land (\|p_{palm,i} - p_{target,i}\| > d_{lost})

    Each hand carries its own consecutive-step counter, zeroed on the falling edge of its own condition --
    including when :math:`c_i` drops to 0 between grasp windows -- and either hand exceeding ``max_steps`` ends
    the episode. Because the reset is per hand, hands alternating between lost and regained never accumulate,
    and a clip with several short windows never carries a count from one window into the next.

    Despite the name this tests "not in contact although the reference says it should be", whether or not
    contact was ever established in the first place.

    The thresholds are deliberately looser than the reward's: 2 N defines a successful grip there, 1 N and
    0.2 m define failure here, and the gap between them is a deadband where a hand is neither paid in full nor
    killed.
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

        force = hand_object_normal_force(env, contact_sensor_names, reduce="mean")
        lost = (command.ref_contact > 0.5) & (force < force_threshold) & (dist > distance_threshold)

        self.lost_steps = torch.where(lost, self.lost_steps + 1, torch.zeros_like(self.lost_steps))
        return (self.lost_steps > max_steps).any(dim=-1)
