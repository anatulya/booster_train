from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.objects.boxes import SMALLBOX_0539923_CFG as OBJECT_CFG
from booster_train.assets.robots.booster import BOOSTER_K1_CFG as ROBOT_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from .tracking_env_cfg import TrackingEnvCfg

# The reference was retargeted by InterMimic and baked with scripts/pt_to_npz.py.
# 349 frames at 50 Hz; the command advances one frame per control step, so the motion
# lasts 349 / 50 = 6.98 s.
MOTION_FRAMES = 349
MOTION_FPS = 50


@configclass
class FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.commands.motion.motion_file = f"{BOOSTER_ASSETS_DIR}/motions/K1/sub4_smallbox_047_hold_from_pt.npz"
        self.commands.motion.anchor_body_name = "Trunk"
        # Keep the episode within one pass of the motion. The base cfg uses 10.0 s, which is
        # longer than this reference and would wrap the playhead mid-episode.
        # self.episode_length_s = MOTION_FRAMES / MOTION_FPS
        # Must stay well below MOTION_FRAMES: max_reset_frame = time_step_total - tail_len,
        # so a large value (mj_dance_004 uses 400) would drive it negative on a 349 frame
        # motion and break the adaptive sampler's bin count.
        # self.commands.motion.tail_len = 0
        # self.commands.motion.adaptive_uniform_ratio = 0.1
        self.commands.motion.body_names = [
            'Trunk',
            'Head_2',
            'Left_Hip_Roll',
            'Left_Shank',
            'left_foot_link',
            'Right_Hip_Roll',
            'Right_Shank',
            'right_foot_link',
            'Left_Arm_2',
            'Left_Arm_3',
            'left_hand_link',
            'Right_Arm_2',
            'Right_Arm_3',
            'right_hand_link',
        ]


@configclass
class FlatWoStateEstimationEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class RoughWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False        # 设为True可视化地形分布
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),            # 每个地形块尺寸（米）
            border_width=20.0,            # 边界宽度（米）
            num_rows=5,                   # 地形网格行数
            num_cols=10,                  # 地形网格列数
            horizontal_scale=0.1,         # 水平分辨率
            vertical_scale=0.005,         # 垂直分辨率
            slope_threshold=0.75,         # 网格简化阈值
            use_cache=False,              # 每次重新生成地形
            curriculum=False,              # 启用课程学习
            sub_terrains={
                # 80%接近平面的地形（非常平滑）
                "nearly_flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.8,
                    noise_range=(0.0, 0.005),    # 高度波动0-0.5cm（几乎平坦）
                    noise_step=0.005,            # 噪声步长0.5cm
                    border_width=0.25,
                ),
                # 20%随机粗糙地形
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(-0.015, 0.015),    # 高度波动±1.5cm
                    noise_step=0.005,               # 噪声步长0.5cm
                    border_width=0.25,
                ),
            },
        )


@configclass
class PlayFlatWoStateEstimationEnvCfg(FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None


@configclass
class FlatLowFreqEnvCfg(FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
