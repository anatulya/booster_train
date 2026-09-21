from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms

from booster_train.tasks.manager_based.hoi_track.mdp.geometry import (
    quat_in_heading,
    rotate_into_heading,
    tan_norm,
)

from booster_train.tasks.manager_based.hoi_track.mdp.commands import MotionCommand
from booster_train.tasks.manager_based.hoi_track.mdp.rewards import _hand_object_contact_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _command(env: ManagerBasedEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


##
# Object-aware terms.
#
# Reference terms are a pure function of the motion file and the horizon offset ``k``: no simulator state
# enters, so they are deployable as-is (they are table lookups, like the reference joint angles already are).
# They are expressed in the *reference* root's heading frame, which stands in for the robot's unknown world
# frame -- legitimate because the result is only ever used to form body-relative differences.
#
# Actual terms read the simulated object and are privileged for now. Note that each one has the robot's world
# pose cancel out (object minus root, both in world), so they become deployable the moment the object's pose
# in the body frame comes from perception -- unlike ``motion_anchor_pos_b``, which never can.
##


def _ref_root(command: MotionCommand, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference anchor position and orientation at ``frames``, straight from the motion file (no env origins).

    Everything built on this stays inside raw motion space, where the reference object positions also live, so
    env origins never enter and cannot cancel incorrectly.
    """
    return (
        command.motion.body_pos_w[frames, command.motion_anchor_body_index],
        command.motion.body_quat_w[frames, command.motion_anchor_body_index],
    )


def _object_points_w(pos: torch.Tensor, quat: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Points (N, P, 3) given in an object's frame, mapped to world by that object's pose (N, 3) / (N, 4)."""
    return pos[:, None, :] + quat_apply(quat[:, None, :].expand(-1, local.shape[1], -1), local)


def ref_joint_pos(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    command = _command(env, command_name)
    return command.motion.joint_pos[command.reference_frames(k)]


def ref_joint_vel(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    command = _command(env, command_name)
    return command.motion.joint_vel[command.reference_frames(k)]


def ref_anchor_pos_b(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Reference anchor position at +k in the robot anchor frame; ``motion_anchor_pos_b`` at a future frame."""
    command = _command(env, command_name)
    frames = command.reference_frames(k)
    pos_w = command.motion.body_pos_w[frames, command.motion_anchor_body_index] + env.scene.env_origins
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, pos_w, command.robot_anchor_quat_w
    )
    return pos.view(env.num_envs, -1)


def ref_anchor_ori_b(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Reference anchor orientation at +k in the robot anchor frame (6D); ``motion_anchor_ori_b`` at +k."""
    command = _command(env, command_name)
    quat_w = command.motion.body_quat_w[command.reference_frames(k), command.motion_anchor_body_index]
    return tan_norm(quat_mul(quat_inv(command.robot_anchor_quat_w), quat_w))


def ref_object_pos_refroot(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    command = _command(env, command_name)
    frames = command.reference_frames(k)
    root_pos, root_quat = _ref_root(command, frames)
    return rotate_into_heading(root_quat, command.motion.object_pos_w[frames] - root_pos)


def ref_object_ori_refroot(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    command = _command(env, command_name)
    frames = command.reference_frames(k)
    _, root_quat = _ref_root(command, frames)
    return tan_norm(quat_in_heading(root_quat, command.motion.object_quat_w[frames]))


def ref_object_lin_vel_refroot(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    command = _command(env, command_name)
    frames = command.reference_frames(k)
    _, root_quat = _ref_root(command, frames)
    return rotate_into_heading(root_quat, command.motion.object_lin_vel_w[frames])


def ref_object_ang_vel_refroot(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    command = _command(env, command_name)
    frames = command.reference_frames(k)
    _, root_quat = _ref_root(command, frames)
    return rotate_into_heading(root_quat, command.motion.object_ang_vel_w[frames])


def ref_contact_point_objlocal(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Where each hand should meet the object at +k, in the object's own frame: 3 per hand.

    Out of contact this holds the next window's grasp point, so it is an aim point rather than a dead signal.
    """
    command = _command(env, command_name)
    return command.contact_point_local[command.reference_frames(k)].reshape(env.num_envs, -1)


def ref_contact_point_refroot(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """The same contact points, carried onto the reference object pose and expressed in the reference root."""
    command = _command(env, command_name)
    frames = command.reference_frames(k)
    root_pos, root_quat = _ref_root(command, frames)
    points = _object_points_w(
        command.motion.object_pos_w[frames],
        command.motion.object_quat_w[frames],
        command.contact_point_local[frames],
    )
    return rotate_into_heading(root_quat, points - root_pos[:, None, :]).reshape(env.num_envs, -1)


def ref_contact_flag(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Whether each hand should be touching the object at +k: 0/1 per hand."""
    command = _command(env, command_name)
    return command.motion.contact[command.reference_frames(k)]


def motion_phase(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    return _command(env, command_name).motion_phase


def object_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    rel = command.robot_object_pos_w - command.robot_anchor_pos_w
    return rotate_into_heading(command.robot_anchor_quat_w, rel)


def object_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return tan_norm(quat_in_heading(command.robot_anchor_quat_w, command.robot_object_quat_w))


def object_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return rotate_into_heading(command.robot_anchor_quat_w, command.object.data.root_lin_vel_w)


def object_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return rotate_into_heading(command.robot_anchor_quat_w, command.object.data.root_ang_vel_w)


def actual_contact_point_objlocal(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Where each palm actually is, in the simulated object's frame: 3 per hand.

    Directly comparable to ``ref_contact_point_objlocal`` -- their difference is the grasp error in the frame
    the grasp is actually defined in.
    """
    command = _command(env, command_name)
    rel = command.robot_palm_pos_w - command.robot_object_pos_w[:, None, :]
    conj = quat_inv(command.robot_object_quat_w)[:, None, :].expand(-1, rel.shape[1], -1)
    return quat_apply(conj, rel).reshape(env.num_envs, -1)


def contact_target_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """The reference contact point carried onto the *actual* object pose, in the robot heading frame.

    This is what closes the loop for grasping: if the object has been knocked out of place, the target moves
    with it instead of pointing at where the reference thought it would be.
    """
    command = _command(env, command_name)
    points = _object_points_w(
        command.robot_object_pos_w,
        command.robot_object_quat_w,
        command.contact_point_local[command.reference_frames(0)],
    )
    rel = points - command.robot_anchor_pos_w[:, None, :]
    return rotate_into_heading(command.robot_anchor_quat_w, rel).reshape(env.num_envs, -1)


def contact_residual_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Contact target minus the actual palm, in the robot heading frame: the reach error, 3 per hand."""
    command = _command(env, command_name)
    rel = command.robot_palm_pos_w - command.robot_anchor_pos_w[:, None, :]
    palm_b = rotate_into_heading(command.robot_anchor_quat_w, rel).reshape(env.num_envs, -1)
    return contact_target_b(env, command_name) - palm_b


def hand_object_contact(env: ManagerBasedEnv, contact_sensor_names: list[str], force_threshold: float) -> torch.Tensor:
    """Measured 0/1 contact per hand, ordered like ``contact_sensor_names``.

    Peak-reduced over the control step's substeps: as an observation this is a presence flag, where catching a
    brief touch is the point. The interaction *reward* uses the substep mean instead, so that a graze cannot
    read the same as a grip.
    """
    return _hand_object_contact_state(env, contact_sensor_names, force_threshold, reduce="peak")


##
# Deployable stand-ins for the privileged terms.
#
# Each of these is a pure function of the motion file and the horizon offset, so it is computable on the real
# robot from the reference alone. They exist so a policy can be trained with the same observation *layout* as
# the privileged one while carrying only information the hardware can actually produce -- see
# ``FlatRefOnlyEnvCfg``. The trade is real: the policy learns where the object is *supposed* to be, and is
# blind to where it actually went.
##


def ref_anchor_lin_vel_refroot(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Reference anchor linear velocity at +k, in the reference root's heading frame.

    The deployable stand-in for ``base_lin_vel``, which needs a state estimator the K1 does not have.
    """
    command = _command(env, command_name)
    frames = command.reference_frames(k)
    _, root_quat = _ref_root(command, frames)
    vel = command.motion.body_lin_vel_w[frames, command.motion_anchor_body_index]
    return rotate_into_heading(root_quat, vel[:, None, :]).squeeze(1)


def ref_anchor_pos_delta_refroot(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Where the reference root will be at +k, relative to where it is now, in the current heading frame.

    The deployable stand-in for ``ref_anchor_pos_b``. That term needs the robot's own world position, which is
    not estimable; this one asks only "how far does the reference travel over the next k frames", which is a
    table lookup. It is identically zero at k=0, kept so the observation layout matches the privileged one.
    """
    command = _command(env, command_name)
    now, root_quat = _ref_root(command, command.reference_frames(0))
    ahead = command.motion.body_pos_w[command.reference_frames(k), command.motion_anchor_body_index]
    return rotate_into_heading(root_quat, (ahead - now)[:, None, :]).squeeze(1)


def ref_contact_residual_refroot(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Reference contact target minus the reference palm, in the reference root's heading frame: 3 per hand.

    The deployable stand-in for ``contact_residual_b``. Inside a grasp window the target *is* the reference
    palm, so this is ~0; outside one it is the vector from where the palm is to where the next grasp will
    happen, which is the approach signal the real residual carries.
    """
    command = _command(env, command_name)
    frames = command.reference_frames(k)
    root_pos, root_quat = _ref_root(command, frames)
    target = _object_points_w(
        command.motion.object_pos_w[frames],
        command.motion.object_quat_w[frames],
        command.contact_point_local[frames],
    )
    palm = command._palm_positions(command.motion.body_pos_w[frames], command.motion.body_quat_w[frames])
    return rotate_into_heading(root_quat, target - palm).reshape(env.num_envs, -1)
