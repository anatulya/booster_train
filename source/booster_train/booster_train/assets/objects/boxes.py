import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from booster_assets import BOOSTER_ASSETS_DIR

# Every manipulated object lives at OBJECT_DIR/<name>/<name>.urdf with a single link <name>_link and its mesh at
# <name>.obj, so the name alone identifies the asset. Adding an object means dropping its directory in here.
#
# Mass and inertia come from the URDF, which declares a placeholder 0.1 kg -- callers that need the captured mass
# randomize or override it at runtime. Surface properties do not: Isaac Lab's URDF converter ignores the PyBullet
# <contact> block, so friction/restitution are set by the object_physics_material event instead. collision_props
# cannot be authored on the spawn cfg either, because the importer makes the collision prim instanceable -- write
# offsets through PhysX.
OBJECT_DIR = f"{BOOSTER_ASSETS_DIR}/motions/K1"

OBJECT_NAMES = sorted(
    name for name in os.listdir(OBJECT_DIR) if os.path.isfile(os.path.join(OBJECT_DIR, name, f"{name}.urdf"))
)

# Object tokens as they appear in capture file names, for clips converted before the npz recorded object_name.
# Matched against whole "_"-separated tokens, since the substring "box" is in every box name. One asset per
# token: where a token covers several assets (suitcase_0539923 / suitcase_0674904 / suitcase_tall15), this is
# the one the captures used; the others only ever appear through an explicit --object.
CLIP_TOKEN_TO_OBJECT = {
    "suitcase": "suitcase_0539923",
    "largebox": "largebox_0539923",
    "smallbox": "smallbox_0539923",
    "boxlarge": "behave_boxlarge_0569850",
}


def object_link(name: str) -> str:
    return f"{name}_link"


def object_cfg(name: str) -> RigidObjectCfg:
    return RigidObjectCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            asset_path=f"{OBJECT_DIR}/{name}/{name}.urdf",
            joint_drive=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.11)),
    )


OBJECT_CFGS = {name: object_cfg(name) for name in OBJECT_NAMES}

# Named handles kept for the scripts that import them. Retargeted from the InterMimic sub4_smallbox_047 capture.
SMALLBOX_0539923_CFG = OBJECT_CFGS["smallbox_0539923"].replace(
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1))
)
OBJECT_CFGS["smallbox_0539923"] = SMALLBOX_0539923_CFG
# The OMOMO largebox and suitcase at the 0.539923 scale, used by the sub*_largebox_* / sub*_suitcase_* references.
LARGEBOX_0539923_CFG = OBJECT_CFGS["largebox_0539923"]
SUITCASE_0539923_CFG = OBJECT_CFGS["suitcase_0539923"]
# The same suitcase at a larger 0.674904 scale, for testing whether a bigger object is easier to grip.
SUITCASE_0674904_CFG = OBJECT_CFGS["suitcase_0674904"]
# The 0.539923 suitcase stretched 1.5x / 1.8x along its own up axis (base kept in place), so its sides reach the
# height the reference hands close at; the _w125 / _w140 variants widen tall15 1.25x / 1.4x in plan.
SUITCASE_TALL15_CFG = OBJECT_CFGS["suitcase_tall15"]
SUITCASE_TALL18_CFG = OBJECT_CFGS["suitcase_tall18"]
SUITCASE_TALL15_W125_CFG = OBJECT_CFGS["suitcase_tall15_w125"]
SUITCASE_TALL15_W140_CFG = OBJECT_CFGS["suitcase_tall15_w140"]
