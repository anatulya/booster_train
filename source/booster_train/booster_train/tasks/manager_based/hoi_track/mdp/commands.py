from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """Loads one or more motion npz files onto a single flat timeline.

    Several clips are concatenated along the time axis, so every reference lookup stays a plain index into
    one tensor; ``clip_starts`` / ``clip_lengths`` say which clip a global frame belongs to. Clips must agree
    on fps, joint names and body names, since the policy tracks them with one action space.
    """

    def __init__(self, motion_file: str | Sequence[str],
                 track_body_names: Sequence[str],
                 track_joint_names: Sequence[str],
                 *,
                 default_motion_body_names: Sequence[str] | None = None,
                 default_motion_joint_names: Sequence[str] | None = None,
                 tail_len: int = 0, device: str = "cpu"):
        motion_files = [motion_file] if isinstance(motion_file, str) else list(motion_file)
        assert motion_files, "No motion files given."
        for path in motion_files:
            assert os.path.isfile(path), f"Invalid file path: {path}"
        datas = [np.load(path) for path in motion_files]
        data = datas[0]
        for path, other in zip(motion_files[1:], datas[1:]):
            for key in ("fps", "joint_names", "body_names"):
                if key in data and not np.array_equal(data[key], other[key]):
                    raise ValueError(f"{path}: {key} differs from {motion_files[0]}; clips cannot be combined.")
        self.clip_names = [os.path.basename(path).replace(".npz", "") for path in motion_files]
        self.fps = data["fps"]
        if "body_names" in data:
            self._body_names = data["body_names"].tolist()
        else:
            assert default_motion_body_names is not None, "Motion file missing body_names, and no default_body_names provided."
            self._body_names = default_motion_body_names
        if "joint_names" in data:
            self._joint_names = data["joint_names"].tolist()
        else:
            assert default_motion_joint_names is not None, "Motion file missing joint_names, and no default_joint_names provided."
            self._joint_names = default_motion_joint_names
        self._body_indexes = torch.tensor(
            [self._body_names.index(name) for name in track_body_names], dtype=torch.long, device=device
        )
        self._joint_indexes = torch.tensor(
            [self._joint_names.index(name) for name in track_joint_names], dtype=torch.long, device=device
        )
        cat = lambda key: torch.tensor(np.concatenate([d[key] for d in datas]), dtype=torch.float32, device=device)
        self.joint_pos = cat("joint_pos")[:, self._joint_indexes]
        self.joint_vel = cat("joint_vel")[:, self._joint_indexes]
        self._body_pos_w = cat("body_pos_w")
        self._body_quat_w = cat("body_quat_w")
        self._body_lin_vel_w = cat("body_lin_vel_w")
        self._body_ang_vel_w = cat("body_ang_vel_w")
        # Object trajectory and reference hand-contact labels, present in every baked npz. Tracking-only tasks
        # ignore them; tasks that put the object in the scene reset it onto object_pos_w/object_quat_w.
        self.has_object = all("object_pos_w" in d for d in datas)
        if self.has_object:
            self.object_pos_w = cat("object_pos_w")
            self.object_quat_w = cat("object_quat_w")
            self.object_lin_vel_w = cat("object_lin_vel_w")
            self.object_ang_vel_w = cat("object_ang_vel_w")
            self.contact = cat("contact") if all("contact" in d for d in datas) else None
        self.has_contact = self.has_object and getattr(self, "contact", None) is not None
        self.time_step_total = self.joint_pos.shape[0]
        self.tail_len = tail_len

        # Per-clip bounds on the shared timeline.
        lengths = [int(d["joint_pos"].shape[0]) for d in datas]
        self.clip_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.clip_starts = torch.cumsum(self.clip_lengths, dim=0) - self.clip_lengths
        self.clip_ends = self.clip_starts + self.clip_lengths
        self.num_clips = len(lengths)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def max_reset_frame(self) -> int:
        """Last resettable frame of the first clip; see ``clip_reset_lengths`` for the multi-clip form."""
        return int(self.clip_lengths[0].item()) - self.tail_len

    @property
    def clip_reset_lengths(self) -> torch.Tensor:
        """Resettable frame count per clip (its length minus the un-resettable tail)."""
        return (self.clip_lengths - self.tail_len).clamp(min=1)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        print(f'{self.robot.body_names=}')
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        default_motion_body_names = self.cfg.default_motion_body_names or self.robot.body_names
        default_motion_joint_names = self.cfg.default_motion_joint_names or self.robot.joint_names

        self.motion = MotionLoader(
            self.cfg.motion_file, self.cfg.body_names, self.robot.joint_names,
            default_motion_body_names=default_motion_body_names,
            default_motion_joint_names=default_motion_joint_names,
            tail_len=self.cfg.tail_len, device=self.device
        )
        # Optional object in the scene: reset onto the reference object pose of whichever frame an env starts at.
        self.object: RigidObject | None = None
        if self.cfg.object_asset_name is not None:
            if not self.motion.has_object:
                raise ValueError(f"{self.cfg.object_asset_name=} but the motion files carry no object trajectory.")
            self.object = env.scene[self.cfg.object_asset_name]

        self.has_contact = self.motion.has_contact

        # Palm keypoints. The K1's *_hand_link origin sits at the forearm end, ~0.2 m from the palm, so a
        # contact point taken from the link origin alone is off by more than the object is wide.
        self.palm_body_indexes = [self.cfg.body_names.index(name) for name in self.cfg.palm_body_names]
        self.palm_offsets = torch.tensor(self.cfg.palm_offsets, dtype=torch.float32, device=self.device).view(-1, 3)
        assert len(self.palm_body_indexes) == self.palm_offsets.shape[0], "One palm offset per palm body."

        # Reference frame indexes at each horizon offset, refreshed once per step.
        self._ref_frames: dict[int, torch.Tensor] = {}

        self.contact_point_local: torch.Tensor | None = None
        if self.object is not None and self.has_contact:
            self.contact_point_local = self._compute_contact_targets()

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Adaptive-sampling bins, laid end to end over every clip: bin i covers global frames
        # [bin_frame_start[i], bin_frame_end[i]) of clip bin_clip[i]. With one clip this is the original
        # single-motion layout, so single-clip tasks are unaffected.
        frames_per_bin = 1 / (env.cfg.decimation * env.cfg.sim.dt)  # 1 second of motion
        bin_clip, bin_start, bin_end = [], [], []
        for clip in range(self.motion.num_clips):
            start = int(self.motion.clip_starts[clip].item())
            resettable = int(self.motion.clip_reset_lengths[clip].item())
            count = int(resettable // frames_per_bin) + 1
            edges = torch.linspace(0, resettable, count + 1, device=self.device).long()
            for b in range(count):
                bin_clip.append(clip)
                bin_start.append(start + int(edges[b].item()))
                bin_end.append(start + max(int(edges[b + 1].item()), int(edges[b].item()) + 1))
        self.bin_clip = torch.tensor(bin_clip, dtype=torch.long, device=self.device)
        self.bin_frame_start = torch.tensor(bin_start, dtype=torch.long, device=self.device)
        self.bin_frame_end = torch.tensor(bin_end, dtype=torch.long, device=self.device)
        self.bin_count = len(bin_clip)
        # Reverse map so a running env's current frame can be charged to the bin it started in.
        self.frame_to_bin = torch.zeros(self.motion.time_step_total, dtype=torch.long, device=self.device)
        for b in range(self.bin_count):
            self.frame_to_bin[self.bin_frame_start[b] : self.bin_frame_end[b]] = b
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_clip"] = torch.zeros(self.num_envs, device=self.device)

    def _palm_positions(self, body_pos: torch.Tensor, body_quat: torch.Tensor) -> torch.Tensor:
        """Palm keypoints (N, P, 3) from tracked-body poses (N, B, 3) / (N, B, 4)."""
        pos = body_pos[:, self.palm_body_indexes]
        quat = body_quat[:, self.palm_body_indexes]
        return pos + quat_apply(quat, self.palm_offsets[None].expand(pos.shape[0], -1, -1))

    @property
    def robot_palm_pos_w(self) -> torch.Tensor:
        return self._palm_positions(self.robot_body_pos_w, self.robot_body_quat_w)

    def _compute_contact_targets(self) -> torch.Tensor:
        """Contact target per frame and hand, in the object's own frame: (T, P, 3).

        While the reference reports contact this is the live palm keypoint expressed in the object frame. While
        it does not, it holds the value the palm will have at the *start of the next* contact window, so during
        the approach the target is an aim point rigidly attached to the object that the hand can converge on;
        after the final window it holds that window's last value. Windows are found per clip, because the clips
        are laid end to end on one timeline and the next clip's grasp has nothing to do with this one's.
        """
        palms = self._palm_positions(self.motion.body_pos_w, self.motion.body_quat_w)
        rel = palms - self.motion.object_pos_w[:, None, :]
        conj = quat_inv(self.motion.object_quat_w)[:, None, :].expand(-1, palms.shape[1], -1)
        local = quat_apply(conj, rel)

        target = local.clone()
        contact = self.motion.contact > 0.5
        for clip in range(self.motion.num_clips):
            lo, hi = int(self.motion.clip_starts[clip].item()), int(self.motion.clip_ends[clip].item())
            for side in range(palms.shape[1]):
                flag = contact[lo:hi, side]
                held = torch.nonzero(flag).flatten()
                free = torch.nonzero(~flag).flatten()
                if held.numel() == 0 or free.numel() == 0:
                    continue  # never contacts, or always does: the live point already covers every frame
                first = torch.cat([torch.ones(1, dtype=torch.bool, device=held.device), held[1:] != held[:-1] + 1])
                starts = held[first]
                # For each free frame, the first window start after it; past the last window, that window's end.
                nxt = torch.searchsorted(starts, free, right=True)
                src = torch.where(nxt < starts.numel(), starts[nxt.clamp(max=starts.numel() - 1)], held[-1])
                target[lo + free, side] = local[lo + src, side]
        return target

    def reference_frames(self, k: int = 0) -> torch.Tensor:
        """Motion frame index at horizon offset ``k``, clamped to the end of each env's own clip."""
        if k not in self._ref_frames:
            self._refresh_reference_frames()
        return self._ref_frames[k]

    def _refresh_reference_frames(self):
        # Clamp per clip, not to the end of the concatenated timeline: a +k lookahead near a clip boundary
        # would otherwise read frames belonging to the next motion.
        last = self.motion.clip_ends[self.motion_ids] - 1
        for k in (0, *self.cfg.future_steps):
            self._ref_frames[k] = torch.minimum(self.time_steps + k, last)

    @property
    def motion_phase(self) -> torch.Tensor:
        """Progress through the current clip, in [0, 1]: (N, 1)."""
        elapsed = (self.time_steps - self.motion.clip_starts[self.motion_ids]).float()
        return (elapsed / self.motion.clip_lengths[self.motion_ids].float()).unsqueeze(-1)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self.motion.object_pos_w[self.time_steps] + self._env.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.motion.object_quat_w[self.time_steps]

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self.motion.object_lin_vel_w[self.time_steps]

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return self.motion.object_ang_vel_w[self.time_steps]

    @property
    def ref_contact(self) -> torch.Tensor:
        """Reference hand-object contact labels, [num_envs, 2] (left, right)."""
        return self.motion.contact[self.time_steps]

    @property
    def robot_object_pos_w(self) -> torch.Tensor:
        return self.object.data.root_pos_w

    @property
    def robot_object_quat_w(self) -> torch.Tensor:
        return self.object.data.root_quat_w

    @property
    def robot_object_lin_vel_w(self) -> torch.Tensor:
        return self.object.data.root_lin_vel_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = self.frame_to_bin[self.time_steps.clamp(0, self.motion.time_step_total - 1)]
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        # Smooth within each clip only: neighbouring bins are adjacent in time, but the last bin of one clip
        # has nothing to do with the first bin of the next, so failures must not bleed across that seam.
        smoothed = torch.empty_like(sampling_probabilities)
        for clip in range(self.motion.num_clips):
            mask = self.bin_clip == clip
            segment = sampling_probabilities[mask]
            padded = torch.nn.functional.pad(
                segment.view(1, 1, -1),
                (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
                mode="replicate",
            )
            smoothed[mask] = torch.nn.functional.conv1d(padded, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = smoothed

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # sampling_probabilities = (
        #     (1 - self.cfg.adaptive_uniform_ratio) * sampling_probabilities
        #     + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        # )

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        # Land anywhere inside the sampled bin, not only on its first frame, and adopt that bin's clip.
        span = (self.bin_frame_end - self.bin_frame_start)[sampled_bins]
        offset = (sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) * span).long().clamp(max=span - 1)
        self.time_steps[env_ids] = self.bin_frame_start[sampled_bins] + offset
        self.motion_ids[env_ids] = self.bin_clip[sampled_bins]

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
        self.metrics["sampling_top1_clip"][:] = self.bin_clip[imax].float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        if self.cfg.play:
            # Deal the clips out across envs so one play run shows every motion from its first frame.
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            self.motion_ids[ids] = ids % self.motion.num_clips
            self.time_steps[ids] = self.motion.clip_starts[self.motion_ids[ids]]

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        if self.object is not None:
            # Same frame the robot was just placed on, so object and robot start consistent mid-clip too.
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            frames = self.time_steps[ids]
            object_state = torch.cat(
                [
                    self.motion.object_pos_w[frames] + self._env.scene.env_origins[ids],
                    self.motion.object_quat_w[frames],
                    self.motion.object_lin_vel_w[frames],
                    self.motion.object_ang_vel_w[frames],
                ],
                dim=-1,
            )
            self.object.write_root_state_to_sim(object_state, env_ids=ids)

        # time_steps just moved, so the cached horizon frames are stale. This runs on the reset path too, where
        # _update_command does not, and the first observation after a reset would otherwise read the frame the
        # env was on *before* it was reset.
        self._refresh_reference_frames()

    def _update_command(self):
        self.time_steps += 1
        # Each env ends at the end of its own clip, not of the concatenated timeline.
        env_ids = torch.where(self.time_steps >= self.motion.clip_ends[self.motion_ids])[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self._refresh_reference_frames()

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    play: bool = False

    asset_name: str = MISSING
    object_asset_name: str | None = None
    """Scene asset to reset onto the motion's object trajectory. None leaves the scene object-free (tracking only)."""

    motion_file: str | Sequence[str] = MISSING
    """One motion npz, or several to train a single policy across all of them. Multiple clips are laid end to
    end on one timeline; they must share fps, joint names and body names."""
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    default_motion_body_names: list[str] | None = None
    default_motion_joint_names: list[str] | None = None
    tail_len: int = 0

    future_steps: tuple[int, ...] = (1, 2, 8, 16)
    """Reference offsets in motion frames, besides the current frame 0, exposed through ``reference_frames``."""
    palm_body_names: list[str] = []
    """Bodies carrying a palm keypoint; must be a subset of ``body_names``."""
    palm_offsets: list[tuple[float, float, float]] = []
    """Palm keypoint offset in the frame of each body in ``palm_body_names``."""

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
