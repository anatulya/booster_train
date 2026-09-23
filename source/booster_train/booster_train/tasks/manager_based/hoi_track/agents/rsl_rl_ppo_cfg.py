from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 1000
    experiment_name = "hoi_track"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        # 0.35, not the usual 1.0: the residual action means the mean already sits on the reference, and our
        # arm action scale (0.887) is 2x HDMI's, so std 1.0 would shake the hands with 51 deg per arm joint
        # while they are meant to be holding the box. 0.5 matched HDMI's 25.4 deg on the arms, but a from-scratch
        # run left to itself fell from 0.5 to 0.34 within 200 iterations and settled at ~0.36 for 20k, so 0.5
        # only bought early noise -- which the smoothness penalties then pay for, in proportion to std^2. 0.35
        # (17.8 deg on the arms) starts where the policy ends up anyway.
        init_noise_std=0.35,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


LOW_FREQ_SCALE = 0.5


@configclass
class BaseLowFreqPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
