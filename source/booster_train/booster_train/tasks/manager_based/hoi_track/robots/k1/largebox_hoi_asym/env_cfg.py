"""Environment cfgs for the asymmetric actor-critic HOI task.

Everything about the scene and the motion -- robot asset, action scale, clip list, anchor body, tracked bodies,
which object each clip was captured with -- is inherited from ``largebox_hoi.FlatEnvCfg``. The only thing these
classes change is which observation group the actor reads.
"""

from isaaclab.utils import configclass

from ..largebox_hoi.env_cfg import ALL_MOTION_FILES, FlatEnvCfg, clip_tag
from .tracking_env_cfg import AsymObservationsCfg


@configclass
class FlatAsymEnvCfg(FlatEnvCfg):
    """Deployable actor, privileged critic. Everything else is ``largebox_hoi``'s multi-clip setup."""

    observations: AsymObservationsCfg = AsymObservationsCfg()


@configclass
class PlayFlatAsymEnvCfg(FlatAsymEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # In play mode each env starts at the first frame of a different clip, so one run shows every motion.
        self.commands.motion.play = True
        self.events.push_robot = None
        self.events.push_object = None
        self.events.object_scenario = None
        # No reset jitter either: a re-placed robot would otherwise start each rollout with a random kick and offset.
        self.commands.motion.velocity_range = {}
        self.commands.motion.pose_range = {}
        # The actor group carries observation noise during training. Leaving it on here would make a render
        # show the noise as much as the policy, and would make repeated rollouts non-comparable.
        self.observations.policy.enable_corruption = False

        # Clean render. The command's debug vis draws a FRAME_MARKER axis triad per tracked body, twice over
        # (current and goal) plus the two anchors -- 30 sets of RGB arrows sitting on the joints, which hides
        # the thing we are actually trying to look at. The contact sensor adds force arrows on top.
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False

        # Side-on view that follows the robot. Both suitcase clips have the robot facing roughly -y (yaw -90
        # and -84 deg) with the object 0.2-0.3 m in front of it, and the root barely translates -- it is a
        # stationary pick-up. So the informative angle is perpendicular to the robot->object line, i.e. along
        # x, which is also what eval_tracker_with_box.py's "side" view picks.
        #
        # origin_type "asset_root" offsets eye and lookat by the robot's root *position* only (no yaw), so the
        # framing tracks the robot without the camera swinging as it turns.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        # 2.0 m out puts the ~1.35 m robot across roughly 70% of the frame with headroom for the lift; the
        # lookat sits on the root->object midpoint, which is why its y matches the eye's.
        self.viewer.eye = (2.0, -0.12, 0.42)
        self.viewer.lookat = (0.0, -0.12, -0.12)
        self.viewer.resolution = (1280, 720)


# ---- Single-clip variants, for overfitting one policy per sequence ----------------------------------------------
# Two cfg classes per clip (the privileged task in largebox_hoi generates four), created at module level so the
# training scripts can resolve and serialize them by name.
SINGLE_CLIP_FILES = {clip_tag(path): path for path in ALL_MOTION_FILES}

for _tag, _path in SINGLE_CLIP_FILES.items():
    for _base in (FlatAsymEnvCfg, PlayFlatAsymEnvCfg):
        _name = f"{_base.__name__}_{_tag}"

        def _post_init(self, _base=_base, _path=_path):
            _base.__post_init__(self)
            self.commands.motion.motion_file = _path
            self.set_object_from_motion()  # this clip may use a different object than the multi-clip default

        _cls = type(_name, (_base,), {"__post_init__": _post_init, "__module__": __name__, "__qualname__": _name})
        globals()[_name] = configclass(_cls)
