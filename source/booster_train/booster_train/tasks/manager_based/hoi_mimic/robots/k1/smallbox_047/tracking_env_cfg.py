"""ULTRA-teacher-style HOI tracking MDP (arXiv:2603.03279) for K1 carrying smallbox_047.

Asymmetric actor-critic: the actor sees only onboard proprioception plus reference-derived targets (deployable);
the critic sees ULTRA's privileged teacher state. Domain randomization is intentionally absent for now.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import booster_train.tasks.manager_based.hoi_mimic.mdp as mdp

##
# Shared constants
##

# Hand-object contact. Order must match the npz "contact" channels (scripts/pt_to_npz.py CONTACT_CHANNELS).
HAND_CONTACT_SENSOR_NAMES = ["left_hand_object_contact", "right_hand_object_contact"]
# The box's rigid body is a child link; filtering against ".../Object" itself registers no force.
OBJECT_CONTACT_FILTER = "{ENV_REGEX_NS}/Object/smallbox_0539923_link"
HAND_CONTACT_FORCE_THRESHOLD = 0.1
FOOT_BODY_NAMES = ["left_foot_link", "right_foot_link"]
KNEE_BODY_NAMES = ["Left_Shank", "Right_Shank"]
# Links (besides the palms) whose nearest-surface vectors form the critic's interaction graph.
IG_BODY_NAMES = [
    "left_hand_link", "right_hand_link", "Left_Arm_3", "Right_Arm_3", "Trunk", "Head_2",
    "Left_Shank", "Right_Shank", "left_foot_link", "right_foot_link",
]  # fmt: skip
FUTURE_STEPS = (1, 16)
PROPRIO_HISTORY = 10


@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True, force_threshold=10.0
    )
    # Filtered contact is one sensor body to many targets, so each hand gets its own sensor.
    # history_length covers every physics substep of a control step (decimation 4, or 8 at low frequency).
    left_hand_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_link", filter_prim_paths_expr=[OBJECT_CONTACT_FILTER], history_length=8
    )
    right_hand_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_link", filter_prim_paths_expr=[OBJECT_CONTACT_FILTER], history_length=8
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        object_asset_name="object",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        # no reset jitter while DR is off: robot and box start exactly on the sampled reference frame
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        future_steps=FUTURE_STEPS,
        palm_body_names=["left_hand_link", "right_hand_link"],
        # centroid of the last 5 cm of Left_Arm_4.STL / Right_Arm_4.STL (forearm end)
        palm_offsets=[(0.0, 0.20, 0.0), (0.0, -0.20, 0.0)],
        foot_body_names=FOOT_BODY_NAMES,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable: onboard proprioception with history + reference-derived targets at +1 and +16 frames."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=PROPRIO_HISTORY)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=PROPRIO_HISTORY)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=PROPRIO_HISTORY)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=PROPRIO_HISTORY)
        actions = ObsTerm(func=mdp.last_action, history_length=PROPRIO_HISTORY)

        ref_joint_pos_1 = ObsTerm(func=mdp.ref_joint_pos, params={"command_name": "motion", "k": 1})
        ref_joint_pos_16 = ObsTerm(func=mdp.ref_joint_pos, params={"command_name": "motion", "k": 16})
        ref_joint_vel_1 = ObsTerm(func=mdp.ref_joint_vel, params={"command_name": "motion", "k": 1})
        ref_joint_vel_16 = ObsTerm(func=mdp.ref_joint_vel, params={"command_name": "motion", "k": 16})
        ref_anchor_ori_1 = ObsTerm(func=mdp.ref_anchor_ori_b, params={"command_name": "motion", "k": 1})
        ref_anchor_ori_16 = ObsTerm(func=mdp.ref_anchor_ori_b, params={"command_name": "motion", "k": 16})
        ref_contact_1 = ObsTerm(func=mdp.ref_contact, params={"command_name": "motion", "k": 1})
        ref_contact_16 = ObsTerm(func=mdp.ref_contact, params={"command_name": "motion", "k": 16})
        ref_object_pose_1 = ObsTerm(func=mdp.ref_object_pose_ref_b, params={"command_name": "motion", "k": 1})
        ref_object_pose_16 = ObsTerm(func=mdp.ref_object_pose_ref_b, params={"command_name": "motion", "k": 16})
        ref_palm_ig_1 = ObsTerm(func=mdp.ref_palm_object_ig_ref_b, params={"command_name": "motion", "k": 1})
        ref_palm_ig_16 = ObsTerm(func=mdp.ref_palm_object_ig_ref_b, params={"command_name": "motion", "k": 16})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """ULTRA teacher state: sim, reference and residuals at +1 and +16 frames, interaction graph."""

        # simulated humanoid
        root_height = ObsTerm(func=mdp.robot_root_height, params={"command_name": "motion"})
        link_state = ObsTerm(func=mdp.robot_link_state_heading, params={"command_name": "motion"})
        link_contact = ObsTerm(
            func=mdp.link_contact_flags, params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 0.1}
        )
        hand_contact = ObsTerm(
            func=mdp.hand_object_contact,
            params={"contact_sensor_names": HAND_CONTACT_SENSOR_NAMES, "force_threshold": HAND_CONTACT_FORCE_THRESHOLD},
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=mdp.joint_effort)
        actions = ObsTerm(func=mdp.last_action)
        # simulated object
        object_state = ObsTerm(func=mdp.object_state_heading, params={"command_name": "motion"})
        # reference (same targets the actor gets, plus full link/object state)
        ref_joint_pos_1 = ObsTerm(func=mdp.ref_joint_pos, params={"command_name": "motion", "k": 1})
        ref_joint_pos_16 = ObsTerm(func=mdp.ref_joint_pos, params={"command_name": "motion", "k": 16})
        ref_joint_vel_1 = ObsTerm(func=mdp.ref_joint_vel, params={"command_name": "motion", "k": 1})
        ref_joint_vel_16 = ObsTerm(func=mdp.ref_joint_vel, params={"command_name": "motion", "k": 16})
        ref_anchor_ori_1 = ObsTerm(func=mdp.ref_anchor_ori_b, params={"command_name": "motion", "k": 1})
        ref_anchor_ori_16 = ObsTerm(func=mdp.ref_anchor_ori_b, params={"command_name": "motion", "k": 16})
        ref_contact_1 = ObsTerm(func=mdp.ref_contact, params={"command_name": "motion", "k": 1})
        ref_contact_16 = ObsTerm(func=mdp.ref_contact, params={"command_name": "motion", "k": 16})
        ref_link_pose_1 = ObsTerm(func=mdp.ref_link_pose_heading, params={"command_name": "motion", "k": 1})
        ref_link_pose_16 = ObsTerm(func=mdp.ref_link_pose_heading, params={"command_name": "motion", "k": 16})
        ref_object_state_1 = ObsTerm(func=mdp.ref_object_state_heading, params={"command_name": "motion", "k": 1})
        ref_object_state_16 = ObsTerm(func=mdp.ref_object_state_heading, params={"command_name": "motion", "k": 16})
        # residuals
        link_residual_1 = ObsTerm(func=mdp.link_residual_heading, params={"command_name": "motion", "k": 1})
        link_residual_16 = ObsTerm(func=mdp.link_residual_heading, params={"command_name": "motion", "k": 16})
        object_residual_1 = ObsTerm(func=mdp.object_residual_heading, params={"command_name": "motion", "k": 1})
        object_residual_16 = ObsTerm(func=mdp.object_residual_heading, params={"command_name": "motion", "k": 16})
        # interaction graph
        ig = ObsTerm(func=mdp.robot_ig, params={"command_name": "motion", "body_names": IG_BODY_NAMES})
        ig_residual_1 = ObsTerm(
            func=mdp.ig_residual, params={"command_name": "motion", "body_names": IG_BODY_NAMES, "k": 1}
        )
        ig_residual_16 = ObsTerm(
            func=mdp.ig_residual, params={"command_name": "motion", "body_names": IG_BODY_NAMES, "k": 16}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """No domain randomization yet. The collider offsets are a fixed physics setting, not DR: left unauthored,
    PhysX sizes them from the mesh (~2.5 mm / 0 for this box), and the URDF importer's instanced collision mesh
    cannot take collision_props, so they are written through PhysX here."""

    object_collider_offsets = EventTerm(
        func=mdp.randomize_rigid_body_collider_offsets,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "contact_offset_distribution_params": (0.02, 0.02),
            "rest_offset_distribution_params": (0.001, 0.001),
        },
    )


@configclass
class RewardsCfg:
    # -- r_track: product of exponential tracking terms (ULTRA Table B)
    tracking = RewTerm(
        func=mdp.UltraTrackingReward,
        weight=1.0,
        params={
            "command_name": "motion",
            "contact_sensor_names": HAND_CONTACT_SENSOR_NAMES,
            "force_threshold": HAND_CONTACT_FORCE_THRESHOLD,
        },
    )

    # -- smoothness / regularization (ULTRA Table C)
    base_lin_vel = RewTerm(func=mdp.base_lin_vel_norm, weight=-0.1)
    base_ang_vel = RewTerm(func=mdp.base_ang_vel_sq, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-4.0e-4)
    action_rate = RewTerm(func=mdp.action_rate_norm, weight=-0.1)
    joint_vel_change = RewTerm(func=mdp.joint_vel_change_sq, weight=-2.0e-5)
    base_ang_vel_change = RewTerm(func=mdp.base_ang_vel_change_sq, weight=-5.0e-4)
    torque = RewTerm(func=mdp.torque_norm, weight=-1.0e-3)
    energy = RewTerm(func=mdp.energy_norm, weight=-1.0e-4)
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    torque_limits = RewTerm(func=mdp.torque_limit_ratio, weight=-1.0)
    feet_orientation = RewTerm(
        func=mdp.feet_orientation, weight=-0.35, params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES)}
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-10.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES)},
    )
    # K1 is smaller than G1: ranges bracket the reference motion (feet 0.18-0.38 m, knees 0.18-0.29 m)
    # instead of ULTRA's G1 range [0.25, 0.65].
    feet_distance = RewTerm(
        func=mdp.body_pair_distance_outside,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES), "min_dist": 0.15, "max_dist": 0.45},
    )
    knee_distance = RewTerm(
        func=mdp.body_pair_distance_outside,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=KNEE_BODY_NAMES), "min_dist": 0.15, "max_dist": 0.35},
    )
    stand_on_feet = RewTerm(
        func=mdp.stand_on_feet,
        weight=-1.0,
        params={"command_name": "motion", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES)},
    )
    swing_clearance = RewTerm(
        func=mdp.swing_clearance,
        weight=-0.6,
        params={"command_name": "motion", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES)},
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-50.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # falls
    anchor_pos = DoneTerm(func=mdp.bad_anchor_pos_z_only, params={"command_name": "motion", "threshold": 0.25})
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    # excessive deviation (InterMimic thresholds)
    body_pos = DoneTerm(func=mdp.bad_body_pos_mean, params={"command_name": "motion", "threshold": 0.5})
    object_points = DoneTerm(func=mdp.bad_object_points_mean, params={"command_name": "motion", "threshold": 0.5})
    interaction = DoneTerm(func=mdp.bad_interaction_ratio, params={"command_name": "motion", "threshold": 2.0})
    # contact loss for 20 frames
    hand_contact_loss = DoneTerm(
        func=mdp.HandContactLossStreak,
        params={
            "command_name": "motion",
            "contact_sensor_names": HAND_CONTACT_SENSOR_NAMES,
            "force_threshold": HAND_CONTACT_FORCE_THRESHOLD,
            "threshold_steps": 20,
        },
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.origin_type = "world"
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
