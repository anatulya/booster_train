from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BoundsLossPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with the rl_games bounds loss (see ``agents/modules.py``)."""

    class_name: str = "BoundsLossPPO"
    bounds_loss_coef: float = 10.0
    bounds_soft_bound: float = 1.1


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """ULTRA teacher PPO (arXiv:2603.03279, Table E) on rsl_rl.

    Minibatch 16384 with 4096 envs x horizon 32 = 131072 samples -> 8 mini-batches.
    """

    num_steps_per_env = 32
    max_iterations = 30000
    save_interval = 1000
    experiment_name = "hoi_mimic"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        # Learnable std as in BeyondMimic: ULTRA's fixed log std -2.9 is only ~1-3 deg of noise with K1_ACTION_SCALE.
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 1024, 512],
        critic_hidden_dims=[1024, 1024, 512],
        activation="relu",
    )
    algorithm = BoundsLossPpoAlgorithmCfg(
        value_loss_coef=5.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=6,
        num_mini_batches=8,
        learning_rate=2.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,  # unused with the fixed schedule
        max_grad_norm=1.0,
        bounds_loss_coef=10.0,
        bounds_soft_bound=1.1,
    )
