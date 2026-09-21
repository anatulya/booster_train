from isaaclab.utils import configclass

from booster_train.tasks.manager_based.hoi_track.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 50000
    experiment_name = "k1_largebox_hoi_asym"
    # Set explicitly rather than leaning on resolve_obs_groups' name-matching fallback, which warns. With the
    # actor and critic now different widths, this is what sizes each network (ActorCritic.__init__ reads it).
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()
        # Per-set normalization, instead of the deprecated runner-level empirical_normalization. rsl_rl keeps
        # one normalizer per observation set, so the asymmetric widths are handled for free.
        self.empirical_normalization = None
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True


# Per-clip runner cfgs (experiment k1_largebox_hoi_asym_<clip>), so single-clip runs log to their own folders.
def _register_single_clip_runners():
    from .env_cfg import SINGLE_CLIP_FILES

    for tag in SINGLE_CLIP_FILES:
        name = f"PPORunnerCfg_{tag}"
        attrs = {"experiment_name": f"k1_largebox_hoi_asym_{tag}", "__module__": __name__, "__qualname__": name}
        globals()[name] = configclass(type(name, (PPORunnerCfg,), attrs))


_register_single_clip_runners()
