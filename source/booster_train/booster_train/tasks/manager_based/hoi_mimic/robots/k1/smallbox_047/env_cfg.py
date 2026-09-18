from isaaclab.utils import configclass

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.objects.boxes import SMALLBOX_0539923_CFG as OBJECT_CFG
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE

from .tracking_env_cfg import TrackingEnvCfg

# The reference was retargeted by InterMimic and baked with scripts/pt_to_npz.py (+ append_hold_npz.py):
# 499 frames at 50 Hz, one frame per control step.
K1_BODY_NAMES = [
    "Trunk", "Head_1", "Left_Arm_1", "Right_Arm_1", "Left_Hip_Pitch", "Right_Hip_Pitch", "Head_2",
    "Left_Arm_2", "Right_Arm_2", "Left_Hip_Roll", "Right_Hip_Roll", "Left_Arm_3", "Right_Arm_3",
    "Left_Hip_Yaw", "Right_Hip_Yaw", "left_hand_link", "right_hand_link", "Left_Shank", "Right_Shank",
    "Left_Ankle_Cross", "Right_Ankle_Cross", "left_foot_link", "right_foot_link",
]  # fmt: skip


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.object = OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Object")
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.commands.motion.motion_file = f"{BOOSTER_ASSETS_DIR}/motions/K1/sub4_smallbox_047_hold_from_pt.npz"
        self.commands.motion.object_mesh_path = (
            f"{BOOSTER_ASSETS_DIR}/motions/K1/smallbox_0539923/smallbox_0539923.obj"
        )
        self.commands.motion.anchor_body_name = "Trunk"
        # ULTRA tracks every link; K1 has 23 bodies.
        self.commands.motion.body_names = K1_BODY_NAMES


@configclass
class PlayFlatEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
