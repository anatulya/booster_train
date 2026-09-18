"""ULTRA teacher PPO pieces that rsl_rl 3.0.1 does not provide.

* :class:`FixedStdActorCritic` -- action std held constant (ULTRA/rl_games ``fixed_sigma``, log std -2.9).
* :class:`BoundsLossPPO` -- PPO with the rl_games bounds loss on the action mean.

rsl_rl resolves ``class_name`` with ``eval`` inside :mod:`rsl_rl.runners.on_policy_runner`, so both classes are
registered into that module's namespace by :func:`register_rsl_rl_classes` (called from ``agents/__init__.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


class FixedStdActorCritic(ActorCritic):
    """:class:`rsl_rl.modules.ActorCritic` whose action standard deviation is not learned."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        std_param = self.std if self.noise_std_type == "scalar" else self.log_std
        std_param.requires_grad_(False)


class BoundsLossPPO(PPO):
    """PPO plus rl_games' bounds loss ``sum(clamp(mu - b, 0)^2 + clamp(mu + b, max=0)^2)``.

    :meth:`update` mirrors ``rsl_rl.algorithms.PPO.update`` (rsl_rl 3.0.1) for the feed-forward case, with the
    bounds loss added to the total loss. RND, symmetry and recurrent policies are not supported here.
    """

    def __init__(self, *args, bounds_loss_coef: float = 10.0, bounds_soft_bound: float = 1.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounds_loss_coef = bounds_loss_coef
        self.bounds_soft_bound = bounds_soft_bound
        assert self.rnd is None and self.symmetry is None and not self.policy.is_recurrent, (
            "BoundsLossPPO only supports feed-forward policies without RND or symmetry."
        )

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_bounds_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            bound = self.bounds_soft_bound
            bounds_loss = (
                torch.clamp(mu_batch - bound, min=0.0).square() + torch.clamp(mu_batch + bound, max=0.0).square()
            ).sum(dim=-1).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.bounds_loss_coef * bounds_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_bounds_loss += bounds_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "bounds": mean_bounds_loss / num_updates,
        }


def register_rsl_rl_classes():
    """Make the classes resolvable by name from rsl_rl's runner (which uses ``eval`` on ``class_name``)."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.FixedStdActorCritic = FixedStdActorCritic
    runner_module.BoundsLossPPO = BoundsLossPPO
