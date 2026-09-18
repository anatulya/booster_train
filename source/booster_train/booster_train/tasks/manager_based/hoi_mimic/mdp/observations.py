"""Observation terms for the ULTRA-style asymmetric actor-critic.

Actor terms (``ref_*``) depend only on the reference motion (and, for ``ref_anchor_ori_b``, the IMU orientation
booster_deploy already uses), so they are computable on the robot. Critic terms (``robot_*``, ``object_*``,
``*_residual*``, ``*_ig*``) use privileged simulator state.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_inv, quat_mul

from booster_train.tasks.manager_based.hoi_mimic.mdp.commands import MotionCommand
from booster_train.tasks.manager_based.hoi_mimic.mdp.geometry import (
    ig_feature,
    nearest_surface_vectors,
    quat_in_heading,
    rotate_into_heading,
    tan_norm,
)
from booster_train.tasks.manager_based.hoi_mimic.mdp.rewards import _hand_object_contact_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _command(env: ManagerBasedEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


##
# Actor (deployable): reference-derived
##


def ref_joint_pos(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    return _command(env, command_name).reference(k)["joint_pos"]


def ref_joint_vel(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    return _command(env, command_name).reference(k)["joint_vel"]


def ref_anchor_ori_b(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Reference anchor orientation at +k in the robot anchor frame (6D); same quantity as booster_deploy's
    ``motion_anchor_ori_b``, just at a future frame."""
    command = _command(env, command_name)
    ori = quat_mul(quat_inv(command.robot_anchor_quat_w), command.reference(k)["anchor_quat_raw"])
    return tan_norm(ori)


def ref_contact(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    return _command(env, command_name).reference(k)["contact"]


def _raw_reference_heading(command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """Current reference anchor position and orientation straight from the motion file (no env origins)."""
    t = command.time_steps
    return (
        command.motion.body_pos_w[t, command.motion_anchor_body_index],
        command.motion.body_quat_w[t, command.motion_anchor_body_index],
    )


def ref_object_pose_ref_b(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Reference object pose at +k in the *current reference* root heading frame: xy relative to the reference
    root, absolute height (InterMimic convention), and 6D rotation. A pure function of the motion and time."""
    command = _command(env, command_name)
    frames = command.reference(k)["frames"]
    root_pos, root_quat = _raw_reference_heading(command)
    rel = command.motion.object_pos_w[frames] - root_pos
    rel[:, 2] = command.motion.object_pos_w[frames, 2]
    pos = rotate_into_heading(root_quat, rel)
    rot = tan_norm(quat_in_heading(root_quat, command.motion.object_quat_w[frames]))
    return torch.cat([pos, rot], dim=-1)


def ref_palm_object_ig_ref_b(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Reference palm-to-nearest-object-surface features at +k, in the current reference root heading frame."""
    command = _command(env, command_name)
    frames = command.reference(k)["frames"]
    root_pos, root_quat = _raw_reference_heading(command)
    palms = command._palm_positions(command.motion.body_pos_w[frames], command.motion.body_quat_w[frames])
    surface = command.object_points_w(command.motion.object_pos_w[frames], command.motion.object_quat_w[frames])
    vectors = nearest_surface_vectors(palms, surface)
    return ig_feature(rotate_into_heading(root_quat, vectors)).reshape(env.num_envs, -1)


##
# Critic (privileged)
##


def robot_root_height(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    return (command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def robot_link_state_heading(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Tracked links in the robot heading frame: local pos (3), 6D rot (6), lin vel (3), ang vel (3) per link."""
    command = _command(env, command_name)
    anchor_quat = command.robot_anchor_quat_w
    pos = rotate_into_heading(anchor_quat, command.robot_body_pos_w - command.robot_anchor_pos_w[:, None, :])
    rot = tan_norm(quat_in_heading(anchor_quat, command.robot_body_quat_w))
    lin = rotate_into_heading(anchor_quat, command.robot_body_lin_vel_w)
    ang = rotate_into_heading(anchor_quat, command.robot_body_ang_vel_w)
    return torch.cat([pos, rot, lin, ang], dim=-1).reshape(env.num_envs, -1)


def link_contact_flags(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return (sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=-1) > threshold).float()


def hand_object_contact(env: ManagerBasedEnv, contact_sensor_names: list[str], force_threshold: float) -> torch.Tensor:
    return _hand_object_contact_state(env, contact_sensor_names, force_threshold)


def _object_in_robot_heading(command: MotionCommand, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    rel = pos - command.robot_anchor_pos_w
    return torch.cat(
        [rotate_into_heading(command.robot_anchor_quat_w, rel), tan_norm(quat_in_heading(command.robot_anchor_quat_w, quat))],
        dim=-1,
    )


def object_state_heading(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Simulated object pose (3 + 6D) and lin/ang velocity in the robot heading frame."""
    command = _command(env, command_name)
    obj = command.object.data
    return torch.cat(
        [
            _object_in_robot_heading(command, obj.root_pos_w, obj.root_quat_w),
            rotate_into_heading(command.robot_anchor_quat_w, obj.root_lin_vel_w),
            rotate_into_heading(command.robot_anchor_quat_w, obj.root_ang_vel_w),
        ],
        dim=-1,
    )


def ref_link_pose_heading(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Re-anchored reference link pose at +k in the robot heading frame: pos (3) + 6D rot (6) per link."""
    command = _command(env, command_name)
    ref = command.reference(k)
    anchor_quat = command.robot_anchor_quat_w
    pos = rotate_into_heading(anchor_quat, ref["body_pos"] - command.robot_anchor_pos_w[:, None, :])
    rot = tan_norm(quat_in_heading(anchor_quat, ref["body_quat"]))
    return torch.cat([pos, rot], dim=-1).reshape(env.num_envs, -1)


def ref_object_state_heading(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    command = _command(env, command_name)
    ref = command.reference(k)
    return torch.cat(
        [
            _object_in_robot_heading(command, ref["object_pos"], ref["object_quat"]),
            rotate_into_heading(command.robot_anchor_quat_w, ref["object_lin_vel"]),
            rotate_into_heading(command.robot_anchor_quat_w, ref["object_ang_vel"]),
        ],
        dim=-1,
    )


def link_residual_heading(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    """Reference (+k) minus simulated link state in the robot heading frame: pos (3), rot (6D of
    sim^-1 * ref, 6), lin vel (3), ang vel (3) per link."""
    command = _command(env, command_name)
    ref = command.reference(k)
    anchor_quat = command.robot_anchor_quat_w
    dpos = rotate_into_heading(anchor_quat, ref["body_pos"] - command.robot_body_pos_w)
    drot = tan_norm(quat_mul(quat_inv(command.robot_body_quat_w), ref["body_quat"]))
    dlin = rotate_into_heading(anchor_quat, ref["body_lin_vel"] - command.robot_body_lin_vel_w)
    dang = rotate_into_heading(anchor_quat, ref["body_ang_vel"] - command.robot_body_ang_vel_w)
    return torch.cat([dpos, drot, dlin, dang], dim=-1).reshape(env.num_envs, -1)


def object_residual_heading(env: ManagerBasedEnv, command_name: str, k: int) -> torch.Tensor:
    command = _command(env, command_name)
    ref = command.reference(k)
    obj = command.object.data
    anchor_quat = command.robot_anchor_quat_w
    return torch.cat(
        [
            rotate_into_heading(anchor_quat, ref["object_pos"] - obj.root_pos_w),
            tan_norm(quat_mul(quat_inv(obj.root_quat_w), ref["object_quat"])),
            rotate_into_heading(anchor_quat, ref["object_lin_vel"] - obj.root_lin_vel_w),
            rotate_into_heading(anchor_quat, ref["object_ang_vel"] - obj.root_ang_vel_w),
        ],
        dim=-1,
    )


def _ig_keypoints(command: MotionCommand, body_pos: torch.Tensor, body_quat: torch.Tensor, body_names: list[str]):
    idx = [command.cfg.body_names.index(name) for name in body_names]
    return torch.cat([command._palm_positions(body_pos, body_quat), body_pos[:, idx]], dim=1)


def robot_ig(env: ManagerBasedEnv, command_name: str, body_names: list[str]) -> torch.Tensor:
    """Simulated interaction graph: palms + ``body_names`` to nearest object surface, in the robot heading frame."""
    command = _command(env, command_name)
    keypoints = _ig_keypoints(command, command.robot_body_pos_w, command.robot_body_quat_w, body_names)
    vectors = nearest_surface_vectors(keypoints, command.robot_object_points_w)
    return ig_feature(rotate_into_heading(command.robot_anchor_quat_w, vectors)).reshape(env.num_envs, -1)


def ig_residual(env: ManagerBasedEnv, command_name: str, body_names: list[str], k: int) -> torch.Tensor:
    """Reference (+k) minus simulated interaction-graph features."""
    command = _command(env, command_name)
    ref = command.reference(k)
    keypoints = _ig_keypoints(command, ref["body_pos"], ref["body_quat"], body_names)
    surface = command.object_points_w(ref["object_pos"], ref["object_quat"])
    ref_feat = ig_feature(rotate_into_heading(command.robot_anchor_quat_w, nearest_surface_vectors(keypoints, surface)))
    return ref_feat.reshape(env.num_envs, -1) - robot_ig(env, command_name, body_names)
