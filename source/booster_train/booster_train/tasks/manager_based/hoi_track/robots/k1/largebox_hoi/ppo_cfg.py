from isaaclab.utils import configclass

from booster_train.tasks.manager_based.hoi_track.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 50000
    experiment_name = "k1_largebox_hoi"



# Per-clip runner cfgs (experiment k1_largebox_tracker_<clip>), so single-clip runs log to their own folders.
def _register_single_clip_runners():
    from .env_cfg import SINGLE_CLIP_FILES

    for tag in SINGLE_CLIP_FILES:
        name = f"PPORunnerCfg_{tag}"
        attrs = {"experiment_name": f"k1_largebox_hoi_{tag}", "__module__": __name__, "__qualname__": name}
        globals()[name] = configclass(type(name, (PPORunnerCfg,), attrs))


_register_single_clip_runners()
