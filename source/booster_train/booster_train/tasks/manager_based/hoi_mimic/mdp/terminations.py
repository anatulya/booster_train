"""Early terminations: falls, excessive deviation, and contact loss (ULTRA; thresholds from InterMimic)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from booster_train.tasks.manager_based.hoi_mimic.mdp.commands import MotionCommand
from booster_train.tasks.manager_based.hoi_mimic.mdp.geometry import interaction_offsets
from booster_train.tasks.manager_based.hoi_mimic.mdp.rewards import _hand_object_contact_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def bad_body_pos_mean(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """InterMimic ``human_reset``: mean tracked-link position error above ``threshold``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = (command.reference(0)["body_pos"] - command.robot_body_pos_w).norm(dim=-1)
    return error.mean(dim=-1) > threshold


def bad_object_points_mean(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """InterMimic ``object_reset``: mean object surface point error above ``threshold``."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref = command.reference(0)
    ref_points = command.object_points_w(ref["object_pos"], ref["object_quat"])
    return (command.robot_object_points_w - ref_points).norm(dim=-1).mean(dim=-1) > threshold


def bad_interaction_ratio(
    env: ManagerBasedRLEnv, command_name: str, threshold: float = 2.0, min_dist: float = 0.5, grace_steps: int = 2
) -> torch.Tensor:
    """InterMimic ``reset_ig``: palm-to-surface offset error relative to max(|offset|, min_dist), in either the
    reference or simulated normalization, exceeds ``threshold`` for any palm/point pair."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref = command.reference(0)
    sim = interaction_offsets(command.robot_palm_pos_w, command.robot_object_points_w)
    target = interaction_offsets(ref["palm_pos"], command.object_points_w(ref["object_pos"], ref["object_quat"]))
    err = (sim - target).norm(dim=-1)
    ratio_ref = err / target.norm(dim=-1).clamp(min=min_dist)
    ratio_sim = err / sim.norm(dim=-1).clamp(min=min_dist)
    bad = (ratio_ref.amax(dim=(-1, -2)) > threshold) | (ratio_sim.amax(dim=(-1, -2)) > threshold)
    return bad & (env.episode_length_buf > grace_steps)


class HandContactLossStreak(ManagerTermBase):
    """Terminates when a hand the reference has in contact is not in contact for more than ``threshold_steps``
    consecutive steps (ULTRA: contact mismatch for 20 frames; InterMimic counts reference-on mismatches)."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.streak = torch.zeros(self.num_envs, len(cfg.params["contact_sensor_names"]), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.streak[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        contact_sensor_names: list[str],
        force_threshold: float,
        threshold_steps: int = 20,
        grace_steps: int = 2,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        contact = _hand_object_contact_state(env, contact_sensor_names, force_threshold)
        lost = (command.reference(0)["contact"] > 0.5) & (contact < 0.5)
        self.streak = torch.where(lost, self.streak + 1.0, torch.zeros_like(self.streak))
        return torch.any(self.streak > threshold_steps, dim=-1) & (env.episode_length_buf > grace_steps)
