from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op, push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg

from booster_train.tasks.manager_based.hoi_track.mdp.rewards import hand_object_normal_force

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. attention::
        **Articulation only.** ``coms[:, body_ids, :3]`` assumes the (num_envs, num_bodies, 7) layout an
        ``ArticulationView`` returns. A ``RigidBodyView`` returns (num_envs, 7) -- see
        ``rigid_object_data.py``, which slices ``pose[:, 3:7]`` where ``articulation_data.py`` slices
        ``pose[..., 3:7]`` -- so indexing it here raises. Use :func:`randomize_rigid_object_com` for a
        :class:`RigidObject` such as the manipulated box.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_object_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Offset a :class:`RigidObject`'s centre of mass by a uniform sample per environment.

    The :class:`RigidObject` counterpart of :func:`randomize_rigid_body_com`, which cannot be used here: a
    ``RigidBodyView`` reports CoM poses as (num_envs, 7), not the (num_envs, num_bodies, 7) an articulation
    gives, so the body axis that function indexes does not exist.

    Only the position block is touched. The view's quaternion is **xyzw** (``rigid_object_data`` converts it
    to wxyz on read), but leaving orientation alone means no conversion is needed here.

    Startup only: the offset is applied to the *current* CoM, so calling this repeatedly compounds.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")

    coms = asset.root_physx_view.get_coms().clone()  # (num_envs, 7), xyz + xyzw quat
    coms[env_ids, :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_rigid_object_inertia_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Scale a :class:`RigidObject`'s inertia tensor by a uniform factor per environment.

    Isaac Lab 2.3 has no inertia randomisation term of its own -- ``events.py`` offers mass, material, CoM,
    scale, collider offsets and gravity, but nothing for inertia -- so this fills that gap.

    This is *independent* of the mass-coupled scaling ``randomize_rigid_body_mass(recompute_inertia=True)``
    already performs. That term sets ``inertia = default_inertia * mass_ratio``, overwriting whatever is
    there, so this one must be declared **after** it in the event config; the composition is then
    ``default_inertia * mass_ratio * scale``. Physically the mass term keeps the box's inertia consistent
    with its new mass under a uniform-density assumption, while this term perturbs how that mass is
    distributed.

    Startup only: it multiplies the *current* tensor, so repeated calls compound.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    scales = math_utils.sample_uniform(scale_range[0], scale_range[1], (len(env_ids), 1), device="cpu")

    inertias = asset.root_physx_view.get_inertias().clone()  # (num_envs, 9) for a rigid object
    inertias[env_ids] *= scales
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_object_material_with_ratio(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    dynamic_friction_range: tuple[float, float],
    static_dynamic_friction_ratio_range: tuple[float, float],
    restitution_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    num_buckets: int = 64,
):
    """Randomize physics materials with static friction derived from dynamic friction by a ratio.

    The physical constraint is ``static >= dynamic``. Isaac Lab's
    :func:`~isaaclab.envs.mdp.events.randomize_rigid_body_material` enforces it with ``make_consistent=True``,
    which *clamps* ``dynamic = min(static, dynamic)``. That satisfies the constraint but distorts the
    distribution: drawing both from U(a, b) and taking the min gives a dynamic friction whose mean is
    ``a + (b - a) / 3`` rather than the midpoint -- for (0.2, 1.2), 0.533 instead of 0.700, with the bias
    concentrated exactly where the clamp bites.

    This follows HDMI's ``object_body_randomization`` instead: draw dynamic friction, draw a ratio
    ``>= 1``, and set ``static = dynamic * ratio``. The constraint then holds *by construction*, and the
    dynamic friction marginal stays exactly uniform over its range.

    Note the consequence: static friction spans ``dynamic_range * ratio_range``, which is wider than the
    dynamic range. Narrow ``static_dynamic_friction_ratio_range`` if that upper end is not wanted.

    Bucketed like the Isaac Lab term, because PhysX caps the scene at 64000 unique materials; ``num_buckets``
    distinct materials are sampled and then assigned randomly across shapes.

    Startup only: buckets are re-drawn on every call.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    dynamic = math_utils.sample_uniform(*dynamic_friction_range, (num_buckets,), device="cpu")
    ratio = math_utils.sample_uniform(*static_dynamic_friction_ratio_range, (num_buckets,), device="cpu")
    restitution = math_utils.sample_uniform(*restitution_range, (num_buckets,), device="cpu")
    # Column order is the PhysX material layout: static friction, dynamic friction, restitution.
    material_buckets = torch.stack([dynamic * ratio, dynamic, restitution], dim=1)

    total_num_shapes = asset.root_physx_view.max_shapes
    bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")

    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids] = material_buckets[bucket_ids]
    asset.root_physx_view.set_material_properties(materials, env_ids)


def push_object_during_contact(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    contact_sensor_names: list[str],
    force_threshold: float,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Kick the object's velocity, but only while the policy is actually holding it as the reference asks.

    An env is pushed when every hand the reference labels in contact reads at least ``force_threshold`` against
    the object, and at least one hand is labelled. Outside grasp windows a push would just move a box nobody is
    holding; with a hand missing the grasp has already failed, which LostContact deals with. A timer that fires
    while this gate is closed is skipped, not deferred.

    A velocity kick rather than a force impulse, so its effect does not scale with the randomized object mass.
    """
    command = env.command_manager.get_term(command_name)
    ref = command.ref_contact[env_ids] > 0.5
    held = hand_object_normal_force(env, contact_sensor_names, reduce="last")[env_ids] >= force_threshold
    gate = ref.any(dim=-1) & (held | ~ref).all(dim=-1)
    if gate.any():
        push_by_setting_velocity(env, env_ids[gate], velocity_range, asset_cfg)
