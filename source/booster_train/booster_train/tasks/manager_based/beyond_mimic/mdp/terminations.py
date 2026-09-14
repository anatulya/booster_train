from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand
from booster_train.tasks.manager_based.beyond_mimic.mdp.rewards import (
    _body_object_vectors,
    _get_body_indexes,
    _hand_object_contact_state,
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


def bad_motion_object_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.object_pos_relative_w - command.object.data.root_pos_w, dim=-1) > threshold


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


def bad_body_object_relative_position(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str]
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot, reference = _body_object_vectors(command, body_names)
    return torch.any(torch.norm(robot - reference, dim=-1) > threshold, dim=-1)


class BadHandContactMismatchStreak(ManagerTermBase):
    """Terminates when any tracked hand's contact state disagrees with the reference for too many steps.

    A class term because the per-hand streak must persist across steps; the termination manager
    zeroes it for resetting envs through :meth:`reset`.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.mismatch_streak = torch.zeros(
            self.num_envs, len(cfg.params["contact_sensor_names"]), device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.mismatch_streak[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        contact_sensor_names: list[str],
        force_threshold: float,
        threshold_steps: int,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        assert command.has_contact, "Motion file has no 'contact' key; re-bake it with scripts/pt_to_npz.py."
        current = _hand_object_contact_state(env, contact_sensor_names, force_threshold)
        mismatched = current != command.contact
        self.mismatch_streak = torch.where(mismatched, self.mismatch_streak + 1, 0.0)
        return torch.any(self.mismatch_streak > threshold_steps, dim=-1)
