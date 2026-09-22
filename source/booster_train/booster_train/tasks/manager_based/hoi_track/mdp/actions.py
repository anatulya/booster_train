"""Residual joint-position action: the policy outputs a correction to the reference, not an absolute pose.

The stock :class:`JointPositionAction` commands ``default_joint_pos + a * scale``, so a freshly initialised
policy -- whose mean output is near zero -- explores around the robot's default standing pose. On these clips
that is the wrong place to be looking: measured on ``sub1_suitcase_029``, the reference sits a median 0.25
action units from the default but ``Elbow_Yaw``'s *median* is 1.34 and hip pitch reaches 3.6, so for much of
the motion the policy has to learn a large constant bias per joint before it can track anything at all.

Replacing the offset with the current reference pose makes a zero action command the reference exactly, so the
policy starts by roughly tracking and only has to learn the correction. This is the same composition HDMI uses
(``pos_target = ref_joint_pos + net_out * action_scaling``); the difference is that HDMI adds the reference to
the distribution mean inside the network, while we do it here. Doing it in the action term means the env's
action *is* the residual, so ``action_rate_l2`` stops fining the policy for the reference's own fast joint
motion and only penalises residual roughness.

Frame consistency comes for free from the manager ordering in ``ManagerBasedRLEnv.step``: the command advances
in ``command_manager.compute()`` *after* ``action_manager.process_action()``, so the reference read here is
still the frame the policy observed when it chose this action.
"""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands import MotionCommand


class ResidualJointPositionAction(JointPositionAction):
    """``JointPositionAction`` whose offset is the reference pose of the current motion frame."""

    cfg: ResidualJointPositionActionCfg

    def process_actions(self, actions: torch.Tensor):
        command: MotionCommand = self._env.command_manager.get_term(self.cfg.command_name)
        # motion.joint_pos is stored in asset joint order, and joint_names=[".*"] collapses _joint_ids to
        # slice(None), so these line up 1:1 with no reindexing. Deliberately *not* clipped to the soft joint
        # limits: the reference overshoots them on some frames (Right_Shoulder_Roll by 0.157 rad, Elbow_Yaw by
        # 0.122) and we stay faithful to the motion file here. Note MotionCommand._resample_command does clip
        # on the reset path, so the two disagree on exactly those frames.
        self._offset = command.motion.joint_pos[command.reference_frames(0)]
        super().process_actions(actions)

        if self.cfg.clip_to_soft_limits:
            # Clamp the *composed* target, so the reference's own overshoot is clipped along with the policy's
            # residual. Off during training by design (see above); a play/render-time toggle for seeing what the
            # motion looks like when the commanded pose is kept physically reachable.
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            self._processed_actions = torch.clamp(
                self._processed_actions, min=limits[..., 0], max=limits[..., 1]
            )


@configclass
class ResidualJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`ResidualJointPositionAction`."""

    class_type: type = ResidualJointPositionAction

    command_name: str = MISSING
    """Name of the :class:`MotionCommand` term supplying the reference pose."""

    clip_to_soft_limits: bool = False
    """Clamp the composed position target to the asset's soft joint limits.

    Off for training, deliberately: the reference overshoots the soft limits on some frames and the action term
    stays faithful to the motion file, with ``joint_limit`` paying for the consequences. ``scripts/rsl_rl/play.py``
    turns it on with ``--clip_joint_limits`` so a render can be compared against the unclipped one. Note this
    makes the action path agree with ``MotionCommand._resample_command``, which already clips on reset.
    """
