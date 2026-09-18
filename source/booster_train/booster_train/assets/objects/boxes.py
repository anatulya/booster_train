import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from booster_assets import BOOSTER_ASSETS_DIR

# Retargeted from the InterMimic sub4_smallbox_047 capture. Mass and inertia come from the URDF;
# surface properties do not -- Isaac Lab's URDF converter ignores the PyBullet <contact> block, so
# friction/restitution are set by the object_physics_material event instead.
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

# The OMOMO largebox at the same 0.539923 scale, used by the sub*_largebox_* references. Same caveats as
# above: the URDF's <contact> block is dropped by the converter, and collision_props cannot be authored on
# the spawn cfg because the importer makes the collision prim instanceable -- write offsets through PhysX.
# The URDF declares 0.1 kg; callers that need the captured 0.5 kg override mass/inertia at runtime.
LARGEBOX_0539923_CFG = RigidObjectCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{BOOSTER_ASSETS_DIR}/motions/K1/largebox_0539923/largebox_0539923.urdf",
        joint_drive=None,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.11)),
)
