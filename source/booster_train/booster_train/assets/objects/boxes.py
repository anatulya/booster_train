import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from booster_assets import BOOSTER_ASSETS_DIR

# Retargeted from the InterMimic sub4_smallbox_047 capture; mass/inertia/friction come
# straight from the URDF (scanned box, ~0.1 kg).
SMALLBOX_0539923_CFG = RigidObjectCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{BOOSTER_ASSETS_DIR}/motions/K1/smallbox_0539923/smallbox_0539923.urdf",
        joint_drive=None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
)
