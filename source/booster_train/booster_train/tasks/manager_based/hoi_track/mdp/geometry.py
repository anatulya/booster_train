"""Frame helpers shared by the HOI observations.

Everything the object-aware terms expose lives in a *heading frame*: the yaw-only part of some reference
orientation, so roll and pitch of the trunk do not rotate the observation. That matters on this task because
the trunk pitches 40 degrees or more during a pick-up; in the full trunk frame the object vector would swing
with the bend, while in the heading frame "0.35 m ahead and 0.4 m below" stays interpretable throughout.

It also keeps the terms deployable: converting a body-frame vector measured by a camera into the heading frame
needs only roll and pitch, which gravity gives, and never the drifting yaw.

Copied from ``hoi_mimic/mdp/geometry.py``, minus the interaction-graph helpers, which need a sampled mesh.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul, yaw_quat


def heading_inv(quat: torch.Tensor) -> torch.Tensor:
    """Inverse of the yaw-only part of ``quat`` (..., 4): rotates world vectors into a heading-aligned frame."""
    return quat_inv(yaw_quat(quat))


def rotate_into_heading(quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate world vectors (N, ..., 3) into the heading frame of ``quat`` (N, 4)."""
    q = yaw_quat(quat)
    shape = vectors.shape
    q = q.view(shape[0], *([1] * (len(shape) - 2)), 4).expand(*shape[:-1], 4)
    return quat_apply_inverse(q, vectors)


def quat_in_heading(quat_ref: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Orientations (N, ..., 4) expressed in the heading frame of ``quat_ref`` (N, 4)."""
    h = heading_inv(quat_ref)
    shape = quat.shape
    h = h.view(shape[0], *([1] * (len(shape) - 2)), 4).expand(*shape)
    return quat_mul(h, quat)


def tan_norm(quat: torch.Tensor) -> torch.Tensor:
    """6D rotation (first two rotation-matrix columns), flattened over the last axis: (..., 4) -> (..., 6).

    Quaternions double-cover, so ``q`` and ``-q`` are the same rotation and a network fed raw quaternions has to
    learn that invariance for nothing. Every orientation observation in this package goes through here.
    """
    mat = matrix_from_quat(quat)
    return mat[..., :2].reshape(*quat.shape[:-1], 6)
