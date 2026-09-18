import glob
import os

from isaaclab.utils import configclass

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE

from .tracking_env_cfg import TrackingEnvCfg

# All 12 InterMimic tracker rollouts of K1 picking up the largebox, baked with scripts/pt_to_npz.py and given
# a 3 s hold by scripts/append_hold_npz.py. One policy tracks all of them: the clips are laid end to end on a
# single timeline by MotionLoader, and each env is reset onto a (clip, frame) pair by the adaptive sampler.
# There is no object in this scene -- this is whole-body tracking only; the box interaction lives in hoi_mimic.
MOTION_FILES = sorted(glob.glob(f"{BOOSTER_ASSETS_DIR}/motions/K1/tracker/npz/*_hold.npz"))

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


@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    """Deployable observation set: no anchor position, no base linear velocity."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


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


SINGLE_CLIP_FILES = {clip_tag(path): path for path in MOTION_FILES}

for _tag, _path in SINGLE_CLIP_FILES.items():
    for _base in (FlatWoStateEstimationEnvCfg, PlayFlatWoStateEstimationEnvCfg):
        _name = f"{_base.__name__}_{_tag}"

        def _post_init(self, _base=_base, _path=_path):
            _base.__post_init__(self)
            self.commands.motion.motion_file = _path

        _cls = type(_name, (_base,), {"__post_init__": _post_init, "__module__": __name__, "__qualname__": _name})
        globals()[_name] = configclass(_cls)
