from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op, push_by_setting_velocity
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from booster_train.tasks.manager_based.hoi_track.mdp.rewards import hand_object_normal_force

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


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


class ObjectScenario(ManagerTermBase):
    r"""Deal each episode one of three object scenarios, and run the two that take the object away.

    * ``0`` regular -- the task as it has always been. This term does nothing to it.
    * ``1`` forced drop -- at a reference contact frame chosen at reset, a downward world-frame force of
      :math:`k\,m\,g` pushes the box until the simulated hands have been off it for ``release_steps``
      consecutive steps. Only then does the grasp count as dropped.
    * ``2`` no object -- the box is parked ``park_offset`` above the env origin for the whole episode.

    Scenarios 1 and 2 bypass adaptive sampling. The command calls :meth:`presample` right after its own bin
    draw and before it places robot and box, and that overwrites the start frame. No-object episodes start
    uniformly over all resettable frames. Drop episodes start uniformly over the resettable frames that
    still have a drop frame ahead within one episode. Adaptive sampling is about where the *grasp* fails,
    which says nothing about where these two should start. It would also pull drop starts toward failure-heavy
    bins, which often sit after the last drop frame.

    The drop frame is drawn from the reference alone, uniformly over the frames the episode can still reach:
    from the start to the clip end or ``max_episode_length`` steps, whichever is sooner. It must be a frame
    where the reference has a hand in contact *and* the box is lifted at least ``min_lift`` above that clip's
    lowest box height. Waiting for simulated contact would under-sample exactly the grasps the policy is
    still bad at. The lift filter keeps drops out of frames where the box already rests on the floor and a
    push down does nothing.

    ``drop_triggered`` only starts the force. ``grasp_dropped`` is set once contact is actually gone, and it
    is what the rest of the task keys off, via ``env.object_absent = grasp_dropped | (scenario == 2)``:
    object rewards are masked, object terminations are disabled, and the termination penalty applies only to
    genuine falls. So a box the force fails to eject -- capped after ``max_force_s`` and logged as
    ``Scenario/drop_cap_hit_frac`` -- leaves the episode running as a regular one.

    Runs as an interval event with ``interval_range_s=(0, 0)``, i.e. once per control step. Its reset runs
    after the command's, so the new start frame is known and the park write is not undone by the command
    placing the box on the reference. Parking is re-written every step since a parked box would otherwise fall.

    The flags are shared through ``env.object_scenario``, ``env.grasp_dropped`` and ``env.object_absent``,
    written in place so the references stay valid.
    """

    REGULAR, DROP, NO_OBJECT = 0, 1, 2

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        long = dict(dtype=torch.long, device=self.device)
        flag = dict(dtype=torch.bool, device=self.device)
        self.scenario = torch.zeros(n, **long)
        self.drop_frame = torch.zeros(n, **long)
        self.trigger_step = torch.zeros(n, **long)
        self.release_step = torch.zeros(n, **long)
        self.no_contact_steps = torch.zeros(n, **long)
        self.drop_triggered = torch.zeros(n, **flag)
        self.grasp_dropped = torch.zeros(n, **flag)
        self.force_capped = torch.zeros(n, **flag)
        self.object_absent = torch.zeros(n, **flag)
        self.force_scale = torch.zeros(n, device=self.device)
        self.next_scenario = torch.zeros(n, **long)
        self.err_sums = {name: torch.zeros(n, device=self.device) for name in self.TRACKED_ERRORS}
        self.err_counts = torch.zeros(n, device=self.device)

        env.object_scenario = self.scenario
        env.grasp_dropped = self.grasp_dropped
        env.object_absent = self.object_absent
        # MotionCommand calls presample() through this, so scenarios 1 and 2 can pick their own start frame.
        env.object_scenario_sampler = self
        self.presampled = torch.zeros(n, **flag)

        probs = torch.tensor(cfg.params["probabilities"], dtype=torch.float, device=self.device)
        self.probs = probs / probs.sum()
        # Render / eval only: env i always runs fixed_scenarios[i % len], so one rollout shows every scenario.
        fixed = cfg.params.get("fixed_scenarios")
        self.fixed = None if fixed is None else torch.tensor(fixed, dtype=torch.long, device=self.device)
        self.asset: RigidObject = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("object")).name]
        # Built on first use: the command manager is created after the event manager.
        self._eligible_cum: torch.Tensor | None = None
        self._mass: torch.Tensor | None = None
        self._force_written = False

    def _setup(self, command) -> None:
        motion = command.motion
        clip_of_frame = torch.repeat_interleave(
            torch.arange(motion.num_clips, device=self.device), motion.clip_lengths
        )
        z = motion.object_pos_w[:, 2]
        clip_min = torch.full((motion.num_clips,), float("inf"), device=self.device)
        clip_min = clip_min.scatter_reduce(0, clip_of_frame, z, reduce="amin")
        lifted = z - clip_min[clip_of_frame] > self.cfg.params["min_lift"]
        if motion.has_contact:
            eligible = (motion.contact > 0.5).any(dim=-1) & lifted
        else:
            eligible = torch.zeros_like(lifted)
        # cum[i] = number of eligible frames in [0, i), so [s, e) holds cum[e] - cum[s] of them.
        self._eligible_cum = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.device), torch.cumsum(eligible.long(), dim=0)]
        )

        # Start-frame tables for the scenarios that skip adaptive sampling: every resettable frame, and the
        # subset from which a drop frame is still reachable within one episode.
        reset_lengths = motion.clip_reset_lengths
        reset_clips = torch.repeat_interleave(torch.arange(motion.num_clips, device=self.device), reset_lengths)
        offsets = torch.arange(len(reset_clips), device=self.device) - torch.repeat_interleave(
            torch.cumsum(reset_lengths, dim=0) - reset_lengths, reset_lengths
        )
        reset_frames = motion.clip_starts[reset_clips] + offsets
        reachable = self._drop_window_end(reset_frames, reset_clips, motion)
        can_drop = self._eligible_cum[reachable] - self._eligible_cum[reset_frames] > 0
        self._reset_frames, self._reset_clips = reset_frames, reset_clips
        self._drop_start_frames, self._drop_start_clips = reset_frames[can_drop], reset_clips[can_drop]

    def _drop_window_end(self, start: torch.Tensor, clip: torch.Tensor, motion) -> torch.Tensor:
        """Exclusive end of the frames an episode starting at ``start`` can reach."""
        return torch.minimum(motion.clip_ends[clip], start + int(self._env.max_episode_length))

    def _deal(self, ids: torch.Tensor) -> torch.Tensor:
        """Scenario for each of ``ids``: drawn from ``probabilities``, or fixed per env index in render / eval."""
        if self.fixed is not None:
            return self.fixed[ids % len(self.fixed)]
        return torch.multinomial(self.probs, len(ids), replacement=True)

    def presample(self, env_ids: Sequence[int], command) -> None:
        """Deal the next episode's scenario and move non-regular starts off the adaptive draw.

        Called by :class:`MotionCommand` after ``_adaptive_sampling`` has set ``time_steps`` / ``motion_ids``,
        and before it writes the robot and object onto that frame.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(ids) == 0:
            return
        if self._eligible_cum is None:
            self._setup(command)
        scenario = self._deal(ids)
        if len(self._drop_start_frames) == 0:
            scenario = torch.where(scenario == self.DROP, self.REGULAR, scenario)

        for kind, frames, clips in (
            (self.DROP, self._drop_start_frames, self._drop_start_clips),
            (self.NO_OBJECT, self._reset_frames, self._reset_clips),
        ):
            mask = scenario == kind
            if mask.any():
                pick = torch.randint(len(frames), (int(mask.sum()),), device=self.device)
                command.time_steps[ids[mask]] = frames[pick]
                command.motion_ids[ids[mask]] = clips[pick]

        self.next_scenario[ids] = scenario
        self.presampled[ids] = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        params = self.cfg.params
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(ids) == 0:
            return
        command = self._env.command_manager.get_term(params["command_name"])
        if self._eligible_cum is None:
            self._setup(command)

        self._log(ids, params)

        # -- the next episode's scenario: dealt by presample() when the command ran its adaptive draw, and
        # drawn here otherwise (play mode, where starts are dealt deterministically and left alone).
        presampled = self.presampled[ids]
        scenario = torch.where(presampled, self.next_scenario[ids], self._deal(ids))
        self.presampled[ids] = False
        # -- and for drops, the frame: uniform over eligible frames this episode can still reach
        cum = self._eligible_cum
        start = command.time_steps[ids]
        end = self._drop_window_end(start, command.motion_ids[ids], command.motion)
        count = cum[end] - cum[start]
        rank = torch.minimum((torch.rand(len(ids), device=self.device) * count).long(), (count - 1).clamp(min=0))
        frame = torch.searchsorted(cum, (cum[start] + rank + 1).contiguous()) - 1
        scenario = torch.where((scenario == self.DROP) & (count == 0), self.REGULAR, scenario)

        lo, hi = params["force_scale_range"]
        self.scenario[ids] = scenario
        self.drop_frame[ids] = frame
        log = self._env.extras.setdefault("log", {})
        for kind, name in enumerate(self.SCENARIO_NAMES):
            log[f"Scenario/{name}_share"] = (scenario == kind).float().mean()
        self.force_scale[ids] = torch.rand(len(ids), device=self.device) * (hi - lo) + lo
        self.trigger_step[ids] = 0
        self.release_step[ids] = 0
        self.no_contact_steps[ids] = 0
        self.drop_triggered[ids] = False
        self.grasp_dropped[ids] = False
        self.force_capped[ids] = False
        for err in self.TRACKED_ERRORS:
            self.err_sums[err][ids] = 0.0
        self.err_counts[ids] = 0.0
        self._update_absent()
        self._park(params["park_offset"])

    # Tracking errors accumulated per scenario, from MotionCommand.metrics. Drops count only after the release,
    # so the number describes tracking without the box rather than being diluted by the carry before it.
    TRACKED_ERRORS = ("error_body_pos", "error_anchor_rot", "error_joint_pos")
    SCENARIO_NAMES = ("regular", "drop", "noobj")

    def _log(self, ids: torch.Tensor, params: dict) -> None:
        """Per-scenario episode metrics under ``Scenario/``, logged in one batch per reset call.

        Every scenario gets:
        - ``share``: its fraction of the episodes *dealt* in this batch, logged from :meth:`reset`. Counting
          the episodes that end would over-weight whichever scenario fails soonest.
        - ``len_s``: episode length
        - ``timeout_frac``, ``fell_frac`` (``fall_term_names``), ``ee_body_frac``
        - tracking errors

        For a drop, those are over episodes whose grasp was actually released, since that is the scenario
        being trained; a drop that never released is a regular episode in all but name. Regular episodes also
        get ``object_term_frac``, the baseline the other two no longer have. Drops add
        ``triggered_frac`` / ``released_frac`` / ``cap_hit_frac`` for the mechanism itself, ``release_s``
        (trigger to release) and ``post_drop_s`` (release to episode end).
        """
        env = self._env
        scenario = self.scenario[ids]
        terminations = env.termination_manager
        active = terminations.active_terms

        def any_term(names) -> torch.Tensor:
            hit = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
            for name in names:
                if name in active:
                    hit |= terminations.get_term(name)[ids]
            return hit

        fell = any_term(params["fall_term_names"])
        ee = any_term([params["ee_term_name"]])
        object_term = any_term(params["object_term_names"])
        timeout = terminations.time_outs[ids]
        length_s = env.episode_length_buf[ids].float() * env.step_dt
        counts = self.err_counts[ids]

        log = env.extras.setdefault("log", {})
        drop = scenario == self.DROP
        released = self.grasp_dropped[ids]
        groups = {
            "regular": scenario == self.REGULAR,
            "drop": released,
            "noobj": scenario == self.NO_OBJECT,
        }
        for name in self.SCENARIO_NAMES:
            mask = groups[name]
            if not mask.any():
                continue
            log[f"Scenario/{name}_len_s"] = length_s[mask].mean()
            log[f"Scenario/{name}_timeout_frac"] = timeout[mask].float().mean()
            log[f"Scenario/{name}_fell_frac"] = fell[mask].float().mean()
            log[f"Scenario/{name}_ee_body_frac"] = ee[mask].float().mean()
            seen = mask & (counts > 0)
            if seen.any():
                for err in self.TRACKED_ERRORS:
                    log[f"Scenario/{name}_{err}"] = (self.err_sums[err][ids][seen] / counts[seen]).mean()
        if groups["regular"].any():
            log["Scenario/regular_object_term_frac"] = object_term[groups["regular"]].float().mean()

        if drop.any():
            triggered = self.drop_triggered[ids] & drop
            log["Scenario/drop_triggered_frac"] = triggered[drop].float().mean()
            if triggered.any():
                log["Scenario/drop_released_frac"] = released[triggered].float().mean()
                log["Scenario/drop_cap_hit_frac"] = self.force_capped[ids][triggered].float().mean()
            if released.any():
                release_s = (self.release_step[ids] - self.trigger_step[ids]).float() * env.step_dt
                log["Scenario/drop_release_s"] = release_s[released].mean()
                post_drop = (env.episode_length_buf[ids] - self.release_step[ids]).float() * env.step_dt
                log["Scenario/drop_post_drop_s"] = post_drop[released].mean()

    def _update_absent(self) -> None:
        torch.logical_or(self.grasp_dropped, self.scenario == self.NO_OBJECT, out=self.object_absent)

    def _park(self, park_offset: tuple[float, float, float]) -> None:
        ids = (self.scenario == self.NO_OBJECT).nonzero().flatten()
        if len(ids) == 0:
            return
        offset = torch.tensor(park_offset, dtype=torch.float, device=self.device)
        quat = torch.zeros(len(ids), 4, device=self.device)
        quat[:, 0] = 1.0
        pose = torch.cat([self._env.scene.env_origins[ids] + offset, quat], dim=-1)
        self.asset.write_root_pose_to_sim(pose, env_ids=ids)
        self.asset.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

    def _write_force(self, active: torch.Tensor) -> None:
        any_active = bool(active.any())
        if not any_active:
            if self._force_written:
                self.asset.permanent_wrench_composer.reset()
                self._force_written = False
            return
        if self._mass is None:
            # Read after the startup mass randomisation, which is why this is lazy.
            self._mass = self.asset.root_physx_view.get_masses().to(self.device).view(self.num_envs, -1).sum(-1)
        gravity = abs(self._env.sim.cfg.gravity[2])
        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        forces[:, 0, 2] = -(active.float() * self.force_scale * self._mass * gravity)
        self.asset.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torch.zeros_like(forces), is_global=True
        )
        self._force_written = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        command_name: str,
        contact_sensor_names: list[str],
        force_threshold: float,
        probabilities: tuple[float, float, float],
        fixed_scenarios: tuple[int, ...] | None = None,
        force_scale_range: tuple[float, float] = (2.0, 4.0),
        release_steps: int = 5,
        max_force_s: float = 1.0,
        min_lift: float = 0.1,
        park_offset: tuple[float, float, float] = (0.0, 0.0, 10.0),
        fall_term_names: tuple[str, ...] = ("anchor_pos", "anchor_ori"),
        ee_term_name: str = "ee_body_pos",
        object_term_names: tuple[str, ...] = ("object_pos", "object_ori", "lost_contact"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> None:
        command = env.command_manager.get_term(command_name)
        step = env.episode_length_buf

        trigger = (self.scenario == self.DROP) & ~self.drop_triggered & (command.time_steps >= self.drop_frame)
        self.drop_triggered |= trigger
        self.trigger_step = torch.where(trigger, step, self.trigger_step)

        pending = self.drop_triggered & ~self.grasp_dropped & ~self.force_capped
        if pending.any():
            held = (hand_object_normal_force(env, contact_sensor_names, reduce="last") >= force_threshold).any(dim=-1)
            self.no_contact_steps = torch.where(
                pending & ~held, self.no_contact_steps + 1, torch.zeros_like(self.no_contact_steps)
            )
            released = pending & (self.no_contact_steps >= release_steps)
            self.grasp_dropped |= released
            self.release_step = torch.where(released, step, self.release_step)
            self.force_capped |= pending & ~released & ((step - self.trigger_step) * env.step_dt >= max_force_s)
            pending &= ~self.grasp_dropped & ~self.force_capped

        self._write_force(pending)
        self._update_absent()
        self._park(park_offset)

        # Tracking errors: all of a regular or no-object episode, and a drop only once the box is gone.
        counted = (self.scenario != self.DROP) | self.grasp_dropped
        for err in self.TRACKED_ERRORS:
            self.err_sums[err] += command.metrics[err] * counted
        self.err_counts += counted.float()
