"""Asymmetric actor-critic variant of the largebox HOI task.

The critic keeps the full privileged observation (``HoiObsCfg``). The actor is cut down to what a real K1 can
produce: proprioception it has sensors for, a clock, and the reference horizon -- which is a table lookup out
of the motion file and so is available on hardware exactly as it is in sim.

Two terms are dropped relative to the critic for the same reason beyond_mimic drops them: ``base_lin_vel`` and
``ref_anchor_pos_b`` both need the robot's world pose, which requires a state estimator the K1 does not have.
The object and interaction terms are dropped for a different reason -- they need object perception the robot
does not currently carry.

What that leaves is an actor whose only closed-loop error signal is ``ref_anchor_ori_b`` plus whatever it
infers from ``ref_joint_pos - joint_pos``. Every object and contact term is feedforward: the policy learns
where the suitcase is *meant* to be relative to the *reference* root and has to assume it is the reference
root. It cannot see the object move and cannot feel whether it is gripping. Whether a pickup survives that is
the question this task exists to answer.

Everything below the observation layer -- scene, commands, actions, rewards, terminations, events -- is
inherited from ``largebox_hoi`` rather than copied, so reward and termination work lands in one place.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import booster_train.tasks.manager_based.hoi_track.mdp as mdp

from ..largebox_hoi.tracking_env_cfg import HORIZON, REFERENCE_TERM_FUNCS, HoiObsCfg, TrackingEnvCfg

# The actor's reference block, i.e. everything the critic gets at each horizon offset except the one term that
# is not computable on hardware. ``ref_anchor_pos_b`` is the robot-to-reference position error, which needs
# ``robot_anchor_pos_w``; beyond_mimic drops its equivalent (``motion_anchor_pos_b``) for the same reason.
ACTOR_REFERENCE_TERM_FUNCS = tuple(name for name in REFERENCE_TERM_FUNCS if name != "ref_anchor_pos_b")

# Noise on the generated per-offset terms, applied inside the loop so it rides along with each ``k``. Only the
# anchor orientation gets any: the rest of the reference block is a table lookup, exact on the real robot.
# The level matches beyond_mimic's ``motion_anchor_ori_b``.
ACTOR_TERM_NOISE = {"ref_anchor_ori_b": Unoise(n_min=-0.05, n_max=0.05)}


@configclass
class ActorObsCfg(ObsGroup):
    """The deployable half of the observation: proprioception, a clock, and the reference horizon."""

    # -- proprioception. Noise levels are beyond_mimic's, term for term. base_lin_vel is absent: it needs the
    # state estimator the K1 does not have, and no amount of noise makes an unavailable signal available. --
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
    actions = ObsTerm(func=mdp.last_action)

    # -- where we are in the clip. A clock, exact on hardware, so no noise. --
    motion_phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        # Same generation rule as HoiObsCfg, minus ref_anchor_pos_b: ObservationManager reads a group's terms
        # from __dict__, so these land after the declared terms above, in the order they are set here.
        for k in HORIZON:
            for name in ACTOR_REFERENCE_TERM_FUNCS:
                setattr(
                    self,
                    f"{name}_{k}",
                    ObsTerm(
                        func=getattr(mdp, name),
                        params={"command_name": "motion", "k": k},
                        noise=ACTOR_TERM_NOISE.get(name),
                    ),
                )


@configclass
class AsymObservationsCfg:
    """Deployable actor, privileged critic."""

    policy: ActorObsCfg = ActorObsCfg()
    critic: HoiObsCfg = HoiObsCfg()


@configclass
class AsymTrackingEnvCfg(TrackingEnvCfg):
    """``TrackingEnvCfg`` with the observation groups split. Everything else is inherited unchanged."""

    observations: AsymObservationsCfg = AsymObservationsCfg()
