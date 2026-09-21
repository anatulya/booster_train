import glob
import os

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.objects.boxes import LARGEBOX_0539923_CFG, SUITCASE_0539923_CFG
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.hoi_track import mdp

from .tracking_env_cfg import HORIZON, TrackingEnvCfg

# Same clips and tracking setup as largebox_tracker, but with the captured object simulated in the scene: the
# command resets it onto the reference object pose of whichever frame an env starts at, and each hand has a
# filtered contact sensor against it. Observations and rewards are still tracking-only -- object-aware terms
# come next.
MOTION_DIR = f"{BOOSTER_ASSETS_DIR}/motions/K1/hoi_track/npz"
ALL_MOTION_FILES = sorted(glob.glob(f"{MOTION_DIR}/*_hold.npz"))
# One scene holds one object, so the multi-clip task takes whichever object has the most clips here; the
# single-clip tasks below cover every clip, each with the object it was captured with.
_BY_OBJECT: dict[str, list[str]] = {}
for _p in ALL_MOTION_FILES:
    _BY_OBJECT.setdefault("suitcase_0539923" if "suitcase" in os.path.basename(_p) else "largebox_0539923", []).append(_p)
MOTION_FILES = max(_BY_OBJECT.values(), key=len) if _BY_OBJECT else []

# Which captured object a clip interacted with, taken from its file name.
OBJECT_CFGS = {"suitcase_0539923": SUITCASE_0539923_CFG, "largebox_0539923": LARGEBOX_0539923_CFG}


def object_name_for(motion_file: str | list[str]) -> str:
    """The object asset a clip was captured with: suitcase clips carry 'suitcase' in their file name."""
    paths = [motion_file] if isinstance(motion_file, str) else list(motion_file)
    names = {"suitcase_0539923" if "suitcase" in os.path.basename(p) else "largebox_0539923" for p in paths}
    if len(names) != 1:
        raise ValueError(f"clips disagree on the captured object ({names}); train one object at a time.")
    return names.pop()

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
            sensor.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Object/" + f"{name}_link"]


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


@configclass
class PlayFlatWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # In play mode each env starts at the first frame of a different clip, so one run shows every motion.
        self.commands.motion.play = True
        self.events.push_robot = None


# ---- Single-clip variants, for overfitting one policy per sequence ----------------------------------------------
# One pair of cfg classes per clip, e.g. FlatWoStateEstimationEnvCfg_sub10_075 / PlayFlatWoStateEstimationEnvCfg_sub10_075,
# created at module level (not in a closure) so the training scripts can resolve and serialize them by name.
def clip_tag(path: str) -> str:
    """sub10_largebox_075_hold.npz -> sub10_075"""
    return os.path.basename(path).replace("_hold.npz", "").replace("_largebox", "")


SINGLE_CLIP_FILES = {clip_tag(path): path for path in ALL_MOTION_FILES}

for _tag, _path in SINGLE_CLIP_FILES.items():
    for _base in (FlatWoStateEstimationEnvCfg, PlayFlatWoStateEstimationEnvCfg, FlatRefOnlyEnvCfg, PlayFlatRefOnlyEnvCfg):
        _name = f"{_base.__name__}_{_tag}"

        def _post_init(self, _base=_base, _path=_path):
            _base.__post_init__(self)
            self.commands.motion.motion_file = _path
            self.set_object_from_motion()  # this clip may use a different object than the multi-clip default

        _cls = type(_name, (_base,), {"__post_init__": _post_init, "__module__": __name__, "__qualname__": _name})
        globals()[_name] = configclass(_cls)
