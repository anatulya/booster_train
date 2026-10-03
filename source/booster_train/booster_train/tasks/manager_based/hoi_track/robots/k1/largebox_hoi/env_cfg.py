import glob
import os

import numpy as np

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.objects.boxes import CLIP_TOKEN_TO_OBJECT, OBJECT_CFGS, OBJECT_NAMES, mesh_bounds, object_link
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.hoi_track import mdp

from .tracking_env_cfg import HORIZON, TrackingEnvCfg

# Same clips and tracking setup as largebox_tracker, but with the captured object simulated in the scene: the
# command resets it onto the reference object pose of whichever frame an env starts at, and each hand has a
# filtered contact sensor against it. Observations and rewards are still tracking-only -- object-aware terms
# come next.
MOTION_DIR = f"{BOOSTER_ASSETS_DIR}/motions/K1/hoi_track/npz"
ALL_MOTION_FILES = sorted(glob.glob(f"{MOTION_DIR}/*_hold.npz"))


def _clip_object(path: str) -> str:
    """The object asset one clip was captured with.

    The npz's own ``object_name`` (written by pt_to_npz_with_offline_video.py from --object) wins. Clips converted
    before that key existed fall back to their file name, matched on whole tokens against CLIP_TOKEN_TO_OBJECT.
    Anything else is an error: guessing is how smallbox and BEHAVE clips used to be handed the largebox.
    """
    with np.load(path) as data:
        if "object_name" in data.files:
            name = str(data["object_name"])
            if name not in OBJECT_CFGS:
                raise ValueError(f"{path}: object_name {name!r} has no asset; known objects: {OBJECT_NAMES}")
            return name
    tokens = set(os.path.basename(path).removesuffix(".npz").split("_"))
    matches = {CLIP_TOKEN_TO_OBJECT[t] for t in tokens & CLIP_TOKEN_TO_OBJECT.keys()}
    if len(matches) != 1:
        found = f"matches {sorted(matches)}" if matches else "names no known object"
        raise ValueError(
            f"{path}: no object_name key and the file name {found}. Reconvert it with"
            f" pt_to_npz_with_offline_video.py --object <name> (one of {OBJECT_NAMES}), which records it."
        )
    return matches.pop()


def object_name_for(motion_file: str | list[str]) -> str:
    """The object asset a clip, or a set of clips that must share one scene, was captured with."""
    paths = [motion_file] if isinstance(motion_file, str) else list(motion_file)
    names = {_clip_object(p) for p in paths}
    if len(names) != 1:
        raise ValueError(f"clips disagree on the captured object ({names}); train one object at a time.")
    return names.pop()


# One scene holds one object, so the multi-clip task takes whichever object has the most clips here; the
# single-clip tasks below cover every clip, each with the object it was captured with.
_BY_OBJECT: dict[str, list[str]] = {}
for _p in ALL_MOTION_FILES:
    _BY_OBJECT.setdefault(_clip_object(_p), []).append(_p)
MOTION_FILES = max(_BY_OBJECT.values(), key=len) if _BY_OBJECT else []

K1_TRACK_BODY_NAMES = [
    "Trunk", "Head_2",
    "Left_Hip_Roll", "Left_Shank", "left_foot_link",
    "Right_Hip_Roll", "Right_Shank", "right_foot_link",
    "Left_Arm_2", "Left_Arm_3", "left_hand_link",
    "Right_Arm_2", "Right_Arm_3", "right_hand_link",
]  # fmt: skip


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.commands.motion.motion_file = MOTION_FILES
        self.commands.motion.anchor_body_name = "Trunk"
        self.commands.motion.body_names = K1_TRACK_BODY_NAMES
        # After motion_file: the object asset and the hands' contact filter follow from which clip is loaded.
        self.set_object_from_motion()

    def set_object_from_motion(self):
        name = object_name_for(self.commands.motion.motion_file)
        self.scene.object = OBJECT_CFGS[name].replace(prim_path="{ENV_REGEX_NS}/Object")
        for sensor in (self.scene.left_hand_object_contact, self.scene.right_hand_object_contact):
            sensor.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Object/" + object_link(name)]


@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    """Phase 1: the actor sees everything the critic does, object state included.

    The deployable cut lives in :meth:`make_actor_deployable` and is deliberately not called yet -- see the
    docstring on ``HoiObsCfg``. The class keeps its name so the registered task ids do not churn between
    phases; only what ``__post_init__`` does changes.
    """

    def __post_init__(self):
        super().__post_init__()

    def make_actor_deployable(self):
        """Phase 2: drop from the actor everything a real K1 cannot produce.

        Two different reasons, worth keeping straight. ``ref_anchor_pos_b`` and ``base_lin_vel`` need a state
        estimator the robot does not have, and no sensor will ever fix that -- the reference root is an
        abstract target that exists only in world coordinates. The object and interaction terms are privileged
        only because the object pose currently comes from the simulator; each one has the robot's world pose
        cancel out, so they become legitimately deployable as soon as that pose comes from a camera. Drop them
        here for the ablation, not because they are unobtainable.
        """
        policy = self.observations.policy
        policy.base_lin_vel = None
        for k in HORIZON:
            setattr(policy, f"ref_anchor_pos_b_{k}", None)
        for name in (
            "object_pos_b",
            "object_ori_b",
            "object_lin_vel_b",
            "object_ang_vel_b",
            "actual_contact_point_objlocal",
            "contact_target_b",
            "contact_residual_b",
            "measured_hand_contact",
        ):
            setattr(policy, name, None)


@configclass
class FlatRefOnlyEnvCfg(FlatWoStateEstimationEnvCfg):
    """Same observation layout, but every privileged slot carries a reference-derived value instead.

    The privileged terms are replaced rather than dropped: the actor still sees an object position, an object
    orientation, contact flags and so on, in the same slots and at the same widths -- they are just the values
    the *reference* says, read out of the motion file, rather than what the simulator's object is doing. Every
    one is a table lookup, so the whole actor observation is computable on the real robot.

    The point is to find out whether the pickup survives that substitution. A policy trained this way is blind
    to the object having moved: it knows where the suitcase is *meant* to be and must grasp there. If it can,
    the task is deployable without object perception; if it cannot, perception is genuinely required and that
    is worth knowing before building around it.

    The critic is untouched and keeps full privileged state.
    """

    def __post_init__(self):
        super().__post_init__()
        self.make_actor_reference_only()

    def make_actor_reference_only(self):
        policy = self.observations.policy
        motion = {"command_name": "motion"}
        policy.base_lin_vel = ObsTerm(func=mdp.ref_anchor_lin_vel_refroot, params={**motion, "k": 0})
        for k in HORIZON:
            setattr(
                policy,
                f"ref_anchor_pos_b_{k}",
                ObsTerm(func=mdp.ref_anchor_pos_delta_refroot, params={**motion, "k": k}),
            )
        for name, func in (
            ("object_pos_b", mdp.ref_object_pos_refroot),
            ("object_ori_b", mdp.ref_object_ori_refroot),
            ("object_lin_vel_b", mdp.ref_object_lin_vel_refroot),
            ("object_ang_vel_b", mdp.ref_object_ang_vel_refroot),
            ("actual_contact_point_objlocal", mdp.ref_contact_point_objlocal),
            ("contact_target_b", mdp.ref_contact_point_refroot),
            ("contact_residual_b", mdp.ref_contact_residual_refroot),
            ("measured_hand_contact", mdp.ref_contact_flag),
        ):
            setattr(policy, name, ObsTerm(func=func, params={**motion, "k": 0}))


@configclass
class PlayFlatRefOnlyEnvCfg(FlatRefOnlyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None
        # Nominal actuators (gains x1, full strength, no joint friction, default armature); play.py
        # --actuator_stress restores actuator_dr with fixed worst-case values.
        self.events.actuator_dr = None
        self.events.joint_armature = None
        self.events.push_object = None
        self.events.object_scenario = None
        # No reset jitter either: a re-placed robot would otherwise start each rollout with a random kick and offset.
        self.commands.motion.velocity_range = {}
        self.commands.motion.pose_range = {}


@configclass
class PlayFlatWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # In play mode each env starts at the first frame of a different clip, so one run shows every motion.
        self.commands.motion.play = True
        self.events.push_robot = None
        # Nominal actuators (gains x1, full strength, no joint friction, default armature); play.py
        # --actuator_stress restores actuator_dr with fixed worst-case values.
        self.events.actuator_dr = None
        self.events.joint_armature = None
        self.events.push_object = None
        self.events.object_scenario = None
        # No reset jitter either: a re-placed robot would otherwise start each rollout with a random kick and offset.
        self.commands.motion.velocity_range = {}
        self.commands.motion.pose_range = {}


# ---- Single-clip variants, for overfitting one policy per sequence ----------------------------------------------
# One pair of cfg classes per clip, e.g. FlatWoStateEstimationEnvCfg_sub10_075 / PlayFlatWoStateEstimationEnvCfg_sub10_075,
# created at module level (not in a closure) so the training scripts can resolve and serialize them by name.
def clip_tag(path: str) -> str:
    """sub10_largebox_075_hold.npz -> sub10_075"""
    return os.path.basename(path).replace("_hold.npz", "").replace("_largebox", "")


SINGLE_CLIP_FILES = {clip_tag(path): path for path in ALL_MOTION_FILES}

# Per-clip ObjectScenario overrides, merged over the tracking_env_cfg defaults. Grip strength is a property of
# the policy trained on a clip, so the drop force has to be tuned per sequence.
#
# sub3_003_stand, from a sweep on its gain-10 fine-tune (model_35000, ~150 drops per config). The default
# 3-5 x / 1.5 s left the grip holding on 23.8% of drops (median release 0.54 s). 5-8 x / 1.5 s: 14.5%.
# 4-6 x / 2.5 s: 7.4%. 5-8 x / 2.5 s: 6.3%. 7-10 x / 1.5 s: 7.3%, releasing fastest (median 0.20 s, p90
# 1.03 s). Falls after release stayed at 3-5% throughout, so the heavier push does not destabilise the robot.
#
# sub03_behave_boxlarge_000_stand, calibrated on the boxlarge policy: 2-4 x weight plus 40-80 N absolute (grip
# friction does not scale with the box's weight), 1.0 s cap, drops 0.5-2.0 s into the episode. 82% of drop
# episodes trigger, 97% of triggered drops release (median 0.14 s, p90 0.20 s), 0% hit the cap. The earlier
# 3-5 x weight-only setting capped 17% of drops (40% on boxes under 0.5 kg).
OBJECT_SCENARIO_OVERRIDES = {
    "sub3_003_stand": {"force_scale_range": (7.0, 10.0), "max_force_s": 1.5},
    "sub03_behave_boxlarge_000_stand": {
        "force_scale_range": (2.0, 4.0),
        "force_abs_range": (40.0, 80.0),
        "release_steps": 5,
        "max_force_s": 1.0,
        "drop_delay_s": (0.5, 2.0),
        "min_lift": 0.10,
    },
}

# Real-world measured size (+/- jitter, applied per axis via a runtime USD scale before the sim starts) for a
# clip's object, where the sim mesh's own baked size differs from the box that will actually be grasped on
# hardware. See mdp.events.randomize_rigid_object_scale_and_store and MotionCommand.contact_point_local_scaled,
# which together keep the hand-contact reward/termination's target on the resized surface rather than where the
# original-size mesh's surface used to be.
#
# Both target_size and jitter are per object-local axis (x, y, z), in metres.
#
# sub1_suitcase_029_stand: real box measured 11.25 x 8.5 x 8.25 in, stood on end and gripped across the 8.5 in
# side. suitcase_0539923's local axes, read off the reference: y is vertical while the box rests (native 28.7
# cm -> 11.25 in), z is the axis between the two palms (native 22.3 cm -> 8.5 in), x is the remaining horizontal
# side (native 22.1 cm -> 8.25 in). Every axis is within ~1 cm of native. Grip gets the widest jitter: a
# narrower box than the reference's forces the hands to squeeze past their reference poses, which is the case
# worth making robust; height and depth get less, height because it also shifts where the hands meet the box.
#
# Disabled: no clip currently resizes its object. Re-enable by restoring the entry below.
OBJECT_SIZE_OVERRIDES = {
    # "sub1_suitcase_029_stand": {
    #     "target_size": (8.25 * 0.0254, 11.25 * 0.0254, 8.5 * 0.0254),
    #     "jitter": (0.015, 0.015, 0.02),
    # },
}


# Per-clip overrides of the object_pos termination, merged into its params. sub7_041_stand is a one-handed push
# along the floor: the box travels ~0.85 m and the robot ~1 m. The default heading-frame check compares the box
# with the robot, and both stand still together just as well as they move together, so a policy that grips the
# box and stays put was never terminated (nor by lost_contact: the reference's right-hand contact comes in bursts
# of at most 26 frames, each gap resetting its 75-step counter). In world frame, a robot that never pushes passes
# 0.3 m around frame 105 and is cut 0.5 s later.
OBJECT_POS_TERMINATION_OVERRIDES = {
    "sub7_041_stand": {"world_frame": True, "threshold": 0.3, "max_steps": 25},
}


def apply_clip_overrides(cfg, tag: str) -> None:
    """Merge per-clip overrides into ``cfg``: ``OBJECT_SCENARIO_OVERRIDES`` into the drop event,
    ``OBJECT_POS_TERMINATION_OVERRIDES`` into the object_pos termination, and ``OBJECT_SIZE_OVERRIDES`` into a new
    per-axis object-scale event, all keyed by clip tag. Call after
    ``set_object_from_motion()``, which this relies on for the object's identity."""
    event = getattr(cfg.events, "object_scenario", None)
    if event is not None and tag in OBJECT_SCENARIO_OVERRIDES:
        event.params = {**event.params, **OBJECT_SCENARIO_OVERRIDES[tag]}

    term = getattr(cfg.terminations, "object_pos", None)
    if term is not None and tag in OBJECT_POS_TERMINATION_OVERRIDES:
        term.params = {**term.params, **OBJECT_POS_TERMINATION_OVERRIDES[tag]}

    if tag in OBJECT_SIZE_OVERRIDES:
        override = OBJECT_SIZE_OVERRIDES[tag]
        target, jitter = override["target_size"], override["jitter"]
        name = object_name_for(cfg.commands.motion.motion_file)
        mins, maxs = mesh_bounds(name)
        scale_range = {
            axis: ((t - j) / (hi - lo), (t + j) / (hi - lo))
            for axis, t, j, lo, hi in zip(("x", "y", "z"), target, jitter, mins, maxs)
        }
        # Per-env-different scale requires replicate_physics=False (isaaclab's own randomize_rigid_body_scale
        # docstring); scoped to this clip's own scene cfg, so no other clip/task is affected.
        cfg.scene.replicate_physics = False
        cfg.events.object_scale = EventTerm(
            func=mdp.randomize_rigid_object_scale_and_store,
            mode="prestartup",
            params={
                "scale_range": scale_range,
                "asset_cfg": SceneEntityCfg("object"),
                "relative_child_path": object_link(name),
                "mesh_bounds": (mins, maxs),
            },
        )


for _tag, _path in SINGLE_CLIP_FILES.items():
    for _base in (FlatWoStateEstimationEnvCfg, PlayFlatWoStateEstimationEnvCfg, FlatRefOnlyEnvCfg, PlayFlatRefOnlyEnvCfg):
        _name = f"{_base.__name__}_{_tag}"

        def _post_init(self, _base=_base, _path=_path, _tag=_tag):
            _base.__post_init__(self)
            self.commands.motion.motion_file = _path
            self.set_object_from_motion()  # this clip may use a different object than the multi-clip default
            apply_clip_overrides(self, _tag)

        _cls = type(_name, (_base,), {"__post_init__": _post_init, "__module__": __name__, "__qualname__": _name})
        globals()[_name] = configclass(_cls)
