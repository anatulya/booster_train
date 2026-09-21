# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Booster-K1-Largebox-Hoi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Largebox-Hoi-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

# Reference-only actor: same observation layout, privileged slots filled from the motion file instead of the
# simulator, so the resulting policy is deployable. See FlatRefOnlyEnvCfg.
gym.register(
    id="Booster-K1-Largebox-HoiRef-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatRefOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Largebox-HoiRef-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatRefOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

# One task per clip for single-sequence overfitting, e.g. Booster-K1-Largebox-Hoi-sub10_075-v0 (+ -Play).
from .env_cfg import SINGLE_CLIP_FILES  # noqa: E402

for _tag in SINGLE_CLIP_FILES:
    gym.register(
        id=f"Booster-K1-Largebox-Hoi-{_tag}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:FlatWoStateEstimationEnvCfg_{_tag}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg_{_tag}",
        },
    )
    gym.register(
        id=f"Booster-K1-Largebox-Hoi-{_tag}-v0-Play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatWoStateEstimationEnvCfg_{_tag}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg_{_tag}",
        },
    )
    gym.register(
        id=f"Booster-K1-Largebox-HoiRef-{_tag}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:FlatRefOnlyEnvCfg_{_tag}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg_{_tag}",
        },
    )
    gym.register(
        id=f"Booster-K1-Largebox-HoiRef-{_tag}-v0-Play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatRefOnlyEnvCfg_{_tag}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg_{_tag}",
        },
    )
