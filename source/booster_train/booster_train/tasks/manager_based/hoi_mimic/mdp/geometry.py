"""Frame and interaction-geometry helpers shared by observations, rewards and terminations."""

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
    """6D rotation (first two rotation-matrix columns), flattened over the last axis: (..., 4) -> (..., 6)."""
    mat = matrix_from_quat(quat)
    return mat[..., :2].reshape(*quat.shape[:-1], 6)


def nearest_surface_vectors(points: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
    """Vector from the nearest surface point to each query point (InterMimic ``compute_sdf``).

    Args:
        points: (N, K, 3) query points.
        surface: (N, M, 3) object surface points.
    Returns:
        (N, K, 3) vectors ``point - nearest_surface_point``.
    """
    nearest = torch.cdist(points, surface).argmin(dim=-1)  # (N, K)
    gathered = torch.gather(surface, 1, nearest.unsqueeze(-1).expand(-1, -1, 3))
    return points - gathered


def ig_feature(vectors: torch.Tensor) -> torch.Tensor:
    """InterMimic interaction-graph feature: unit direction scaled by exp(-5 |v|) (near = large, far -> 0)."""
    norm = vectors.norm(dim=-1, keepdim=True)
    return vectors / (norm + 1e-6) * torch.exp(-5.0 * norm)


def interaction_offsets(keypoints: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
    """All keypoint-to-surface-point offsets delta_ij = p_i - s_j: (N, K, 3), (N, M, 3) -> (N, K, M, 3)."""
    return keypoints.unsqueeze(2) - surface.unsqueeze(1)
