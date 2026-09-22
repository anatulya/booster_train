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

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import booster_train.tasks.manager_based.hoi_track.mdp as mdp

##
# Scene definition
##

# The object's rigid body is a child link of the spawned asset; filtering against ".../Object" registers no force.
# env_cfg.py rewrites this to the link name of whichever object the clip used.
OBJECT_CONTACT_FILTER = "{ENV_REGEX_NS}/Object/largebox_0539923_link"
HAND_CONTACT_SENSOR_NAMES = ["left_hand_object_contact", "right_hand_object_contact"]
HAND_CONTACT_FORCE_THRESHOLD = 0.1
# Reward-side contact, taken verbatim from HDMI's eef_contact_exp (cfg/task/base/hdmi-base.yaml:220). Their
# direct analogue to this task -- cfg/task/G1/hdmi/move_suitcase.yaml, same OMOMO dataset, sub1_suitcase_011
# against our sub1_suitcase_029 -- inherits these unchanged.
#
# These are not per-object numbers. frc_thres=10 is a single global constant in HDMI, applied across objects
# from 0.9 kg to 40 kg; the only task-level override repeats it, and the one config with different values is
# commented out. It is better read as a robot-scale quantity -- achievable normal force comes from how hard
# the arm squeezes and how stiff the contact is, not from the object's weight -- so it is a fact about the G1
# rather than something portable to the K1 by arithmetic. Whether the K1 can reach 10 N here is an open
# question; Interaction/force logs it in Newtons so the answer shows up in the first few hundred iterations.
#
# The previous 0.1 / 1.0 pair made this a near-gate rather than a shaped reward: a hand at zero force scored
# exp(-2/1) = 0.135, and at the 25k run's measured 0.158 m gated palm distance r_int scored 0.400. Their
# product left almost no gradient during the approach, which is the phase that needs it most. At 10/40 the
# force factor instead spans only 0.779 to 1.0, so force nudges and position shapes.
CONTACT_REWARD_FORCE_THRESHOLD = 10.0
CONTACT_SIGMA_P = 0.3
CONTACT_SIGMA_F = 40.0
# Scales only the gated part of the interaction reward. Without it a single weight sets both the grasping
# gradient and the flat (1 - gate) survival constant, and the constant wins: at 25k iterations 9.24 of the
# 10.05 earned was the constant and 0.83 was grasping. At gain 5 the gated-frame weight becomes 100 against
# 139 for all body tracking combined -- close to HDMI, which runs grasping at 1:1 with its tracking group.
CONTACT_GAIN = 5.0

# Early termination. LOST_CONTACT_* are HDMI's cum_lost_contact_steps values (hdmi-base.yaml:269), and the
# force bound stays far below the reward's grip threshold on purpose: between 1 N and CONTACT_REWARD_FORCE_-
# THRESHOLD is a deadband where a hand is neither paid in full nor killed. LostContact now fires on either
# condition rather than both, so that deadband no longer shelters a hand that is simply in the wrong place.
#
# OBJECT_POS_TERMINATION is measured in the robot's heading frame (see BadObjectPos), not in world, so it is
# a budget for "the box is misplaced relative to me" and not for global drift. The world-frame version of this
# check fired on 40.6% of episodes at 5.7k iterations, almost all of it root drift rather than dropping.
#
# OBJECT_POS_STEPS mirrors LOST_CONTACT_STEPS: an env reset into the lift-off spawns the box in mid-air with
# the reference already asking for a grip, and had ~16 steps before free fall breached the bound. The counter
# gives it time to close the hands instead of dying on the reset condition.
OBJECT_POS_TERMINATION = 0.5
OBJECT_POS_STEPS = 25
# 1.2 rad (69 deg) clears the motion's own object rotation: measured from its start in the root heading frame,
# the reference box turns at most 0.924 rad (sub1) / 0.569 rad (sub15) and never spends a frame past 1.2. So
# this cannot fire on a policy that simply fails to rotate the box -- it only catches genuine tipping.
OBJECT_ORI_TERMINATION = 1.2
OBJECT_ORI_STEPS = 25
LOST_CONTACT_FORCE = 1.0
LOST_CONTACT_DISTANCE = 0.2
LOST_CONTACT_STEPS = 25

# Reference horizon, in motion frames at 50 Hz: now, +20 ms, +40 ms, +160 ms, +320 ms. The policy is a plain
# MLP with no history, so without a horizon it cannot anticipate -- which matters most for pre-shaping a grasp.
HORIZON = (0, 1, 2, 8, 16)
FUTURE_STEPS = tuple(k for k in HORIZON if k != 0)

PALM_BODY_NAMES = ["left_hand_link", "right_hand_link"]
# The K1's *_hand_link origin is the forearm end, ~0.2 m short of the palm. Verified against the captures with
# pt_to_npz_with_offline_video.py --show_contact_points: the keypoint then sits ~3 cm off the object face
# throughout the grasp, a constant standoff rather than an error to tune away.
PALM_OFFSETS = [(0.0, 0.20, 0.0), (0.0, -0.20, 0.0)]

# Reference terms evaluated at every horizon offset. Generated in HoiObsCfg.__post_init__ rather than written
# out, because 11 terms x 5 offsets is 55 near-identical declarations.
REFERENCE_TERM_FUNCS = (
    "ref_joint_pos",
    "ref_joint_vel",
    "ref_anchor_pos_b",
    "ref_anchor_ori_b",
    "ref_object_pos_refroot",
    "ref_object_ori_refroot",
    "ref_object_lin_vel_refroot",
    "ref_object_ang_vel_refroot",
    "ref_contact_point_objlocal",
    "ref_contact_point_refroot",
    "ref_contact_flag",
)

# Reset jitter only: added to the reference root velocity when an env is placed on its start frame, in
# MotionCommand._resample_command.
#
# Softened to match PUSH_VELOCITY_RANGE below. The old +-0.7/0.7/0.4 m/s was wildly out of scale with the clip:
# the reference Trunk moves at 0.071 m/s median and 0.409 m/s at its fastest, so a max jitter magnitude of
# 1.07 m/s spawned envs travelling 2.6x faster than anything the motion ever does, and ~15x the median. That is
# not initial-state diversity, it is noise unrelated to the task.
#
# Kept as its own constant rather than aliased to the push range: the two happen to coincide now, but they
# answer different questions (how varied a start, versus how hard a shove) and should stay independently
# tunable.
VELOCITY_RANGE = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (-0.05, 0.05),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
}

# Push disturbance. Max magnitude 0.43 m/s and 0.41 rad/s, against 1.07 and 1.34 under the old range -- 2.5x
# smaller linearly, 3.2x angularly, with the vertical kick 8x smaller. The old range was breaking grasps rather
# than teaching recovery.
PUSH_VELOCITY_RANGE = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "z": (-0.05, 0.05),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
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
    # robots
    robot: ArticulationCfg = MISSING
    # The clip's captured object, reset onto the reference object pose of whichever frame an env starts at.
    object: RigidObjectCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )
    # Filtered contact is one sensor body to many targets, so each hand needs its own sensor. history_length
    # covers every physics substep of a control step, so a short contact is not missed between steps.
    left_hand_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_link",
        filter_prim_paths_expr=[OBJECT_CONTACT_FILTER],
        history_length=8,
    )
    right_hand_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_link",
        filter_prim_paths_expr=[OBJECT_CONTACT_FILTER],
        history_length=8,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        object_asset_name="object",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        future_steps=FUTURE_STEPS,
        palm_body_names=PALM_BODY_NAMES,
        palm_offsets=PALM_OFFSETS,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Residual on the reference pose rather than on the default stance: a zero action commands the reference
    # exactly, so the policy starts by roughly tracking and only learns the correction. See mdp/actions.py.
    joint_pos = mdp.ResidualJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True, command_name="motion"
    )


@configclass
class HoiObsCfg(ObsGroup):
    """Proprioception + a reference horizon + object and interaction state.

    Phase 1 is deliberately diagnostic: the actor gets the same privileged observation as the critic, including
    the object state and the root-position terms a real K1 cannot estimate. The point is to find out whether a
    pickup is achievable at all, without the confound of the policy being unable to perceive the object or its
    own drift. ``FlatEnvCfg.make_actor_deployable()`` strips those back off the actor for phase 2.
    """

    # -- proprioception (current frame only; no history stacking) --
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)

    # -- where we are in the clip --
    motion_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})

    # -- simulated object, in the robot heading frame (privileged) --
    object_pos_b = ObsTerm(func=mdp.object_pos_b, params={"command_name": "motion"})
    object_ori_b = ObsTerm(func=mdp.object_ori_b, params={"command_name": "motion"})
    object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_b, params={"command_name": "motion"})
    object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_b, params={"command_name": "motion"})

    # -- interaction with the simulated object (privileged) --
    actual_contact_point_objlocal = ObsTerm(
        func=mdp.actual_contact_point_objlocal, params={"command_name": "motion"}
    )
    contact_target_b = ObsTerm(func=mdp.contact_target_b, params={"command_name": "motion"})
    contact_residual_b = ObsTerm(func=mdp.contact_residual_b, params={"command_name": "motion"})
    measured_hand_contact = ObsTerm(
        func=mdp.hand_object_contact,
        params={"contact_sensor_names": HAND_CONTACT_SENSOR_NAMES, "force_threshold": HAND_CONTACT_FORCE_THRESHOLD},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        # Reference terms at every horizon offset. ObservationManager reads a group's terms from __dict__, so
        # these land after the declared terms above, in the order they are set here.
        for k in HORIZON:
            for name in REFERENCE_TERM_FUNCS:
                setattr(self, f"{name}_{k}", ObsTerm(func=getattr(mdp, name), params={"command_name": "motion", "k": k}))


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: HoiObsCfg = HoiObsCfg()
    critic: HoiObsCfg = HoiObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            # "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    #
    # Additive, despite the name: Isaac Lab's docstring says it "sets the root velocity to a random value", but
    # events.py:1069 does ``vel_w += sample_uniform(...)``, so the sample lands on top of whatever the trunk is
    # already doing rather than replacing it. A push therefore perturbs momentum instead of erasing it.
    #
    # EventManager resamples each env's timer at reset and again after each push, so the first push arrives 3-6 s
    # in and an episode ending before 3 s is never pushed at all. A full 10 s episode now sees ~1.8 pushes against
    # ~4.5 under (1.0, 3.0).
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": PUSH_VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=5.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=5.0,
        params={"command_name": "motion", "std": 0.4},
    )
    # The four motion_body_* terms average over all 14 tracked bodies. Position std is 0.15, not 0.3: at 0.3 the
    # 25k run's 4.7 cm mean body error left this term at kernel ~0.98, almost flat across the errors the policy
    # actually makes, so a weight of 20 would have bought a constant. At 0.15 it sits near 0.87, the operating
    # point the dedicated hand and foot terms already have, and is about four times as sensitive to error.
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=20.0,
        params={"command_name": "motion", "std": 0.15},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=15.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=2.5,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=2.5,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-5.0)
    # sum(raw_action ** 2) -- isaaclab's built-in, operating on env.action_manager.action, the same raw network
    # output action_rate_l2 reads and what ResidualJointPositionAction's docstring calls "the residual" ("the
    # env's action *is* the residual"). Deliberately not the physical joint-space delta (raw_action *
    # K1_ACTION_SCALE, in radians): that would penalise joints unevenly since scale ranges 0.27 (ankles) to
    # 0.89 (arms), an ~11x spread once squared.
    action_l2 = RewTerm(func=mdp.action_l2, weight=-5.0)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # Off for the HOI task. It penalises contact on every body but the feet, using the *unfiltered*
    # contact_forces sensor, so a hand pressing on the object earns the interaction reward and is fined -10 for
    # touching anything at the same time -- the two terms directly cancel. Re-enable with the hands excluded
    # from body_names if the robot needs a penalty for faceplanting.
    undesired_contacts = None
    motion_foot_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=15.0,
        params={"command_name": "motion", "std": 0.2, "body_names": ["left_foot_link", "right_foot_link"]},
    )

    motion_foot_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=15.0,
        params={"command_name": "motion", "std": 0.2, "body_names": ["left_foot_link", "right_foot_link"]},
    )

    motion_hand_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=15.0,
        params={"command_name": "motion", "std": 0.2, "body_names": ["left_hand_link", "right_hand_link"]},
    )

    motion_hand_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=15.0,
        params={"command_name": "motion", "std": 0.2, "body_names": ["left_hand_link", "right_hand_link"]},
    )

    motion_trunk_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=30.0,
        params={"command_name": "motion", "std": 0.2, "body_names": ["Trunk"]},
    )

    # Off. The trunk is the anchor body, so its relative-frame position error is height only (xy is the robot's
    # own by construction) and motion_global_anchor_pos already covers that in 3D. It was saturated at kernel
    # 0.989 in the 25k run -- about 12 of free reward. motion_trunk_ori above stays: it is the tightest roll/pitch
    # signal there is (std 0.2, against 0.4 for anchor_ori and body_ori), and the reference pitches the torso
    # ~40 degrees during the pick-up.
    motion_trunk_pos = None

    motion_trunk_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=5.0,
        params={"command_name": "motion", "std": 3.14, "body_names": ["Trunk"]},
    )

    # -- object interaction --
    # Object and grasp weight, against 145 for all body tracking: pose + lin-vel 35, and the interaction term at 25
    # with gain 5 is 125 on frames where the reference asks for contact and 25 on the rest.
    motion_global_object_pos = RewTerm(
        func=mdp.motion_global_object_position_error_exp,
        weight=20.0,
        params={"command_name": "motion", "std": 0.3},
    )

    motion_global_object_ori = RewTerm(
        func=mdp.motion_global_object_orientation_error_exp,
        weight=10.0,
        params={"command_name": "motion", "std": 0.4},
    )

    # std 0.5 m/s: the reference box's own speed is p90 0.43 / peak 1.7 m/s, so 0.5 gives full credit while
    # it is near-still, 0.37 at half a metre per second of error, and ~0 for a box in free fall.
    motion_global_object_lin_vel = RewTerm(
        func=mdp.motion_global_object_linear_velocity_error_exp,
        weight=5.0,
        params={"command_name": "motion", "std": 0.5},
    )

    hand_object_interaction = RewTerm(
        func=mdp.HandObjectInteraction,
        weight=25.0,
        params={
            "command_name": "motion",
            "contact_sensor_names": HAND_CONTACT_SENSOR_NAMES,
            "sigma_p": CONTACT_SIGMA_P,
            "sigma_f": CONTACT_SIGMA_F,
            "force_threshold": CONTACT_REWARD_FORCE_THRESHOLD,
            "gain": CONTACT_GAIN,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_hand_link",
                "right_hand_link",
                "left_foot_link",
                "right_foot_link",
            ],
        },
    )
    object_pos = DoneTerm(
        func=mdp.BadObjectPos,
        params={"command_name": "motion", "threshold": OBJECT_POS_TERMINATION, "max_steps": OBJECT_POS_STEPS},
    )
    object_ori = DoneTerm(
        func=mdp.BadObjectOri,
        params={"command_name": "motion", "threshold": OBJECT_ORI_TERMINATION, "max_steps": OBJECT_ORI_STEPS},
    )
    lost_contact = DoneTerm(
        func=mdp.LostContact,
        params={
            "command_name": "motion",
            "contact_sensor_names": HAND_CONTACT_SENSOR_NAMES,
            "force_threshold": LOST_CONTACT_FORCE,
            "distance_threshold": LOST_CONTACT_DISTANCE,
            "max_steps": LOST_CONTACT_STEPS,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.origin_type = "world"   # for rotation the view by mouse and keyboard
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
