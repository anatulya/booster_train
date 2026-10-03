"""Load a checkpoint on a task and report what its first steps look like to the policy.

Prints, at a few steps: the action magnitude and the observation terms with the largest normalized values
(``(obs - mean) / (std + 1e-2)``, the actor's own normalizer). A healthy resume has |action| ~ 0.1-1 and |z| of
at most ~10; a reference the checkpoint has never seen shows up as one or a few terms with |z| in the hundreds.

    python scripts/diag_resume.py --task <task> --checkpoint <abs path to model_*.pt>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import sys

sys.stdout.reconfigure(line_buffering=True)

import gymnasium as gym
import torch

import booster_train.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg)
venv = RslRlVecEnvWrapper(env)
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
runner = OnPolicyRunner(venv, agent.to_dict(), log_dir=None, device="cuda:0")
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device="cuda:0")
norm = runner.alg.policy.actor_obs_normalizer

om = env.unwrapped.observation_manager
names = om.active_terms["policy"]
dims = [d[0] if isinstance(d, tuple) else d for d in om.group_obs_term_dim["policy"]]
starts = [sum(dims[:i]) for i in range(len(dims))]


def term_of(i: int) -> str:
    for name, s, d in zip(names, starts, dims):
        if s <= i < s + d:
            return f"{name}[{i - s}]"
    return str(i)


def policy_obs(obs):
    return obs["policy"] if hasattr(obs, "keys") else obs


obs = venv.get_observations()
with torch.inference_mode():
    for step in range(args.steps):
        o = policy_obs(obs)
        z = (o - norm._mean) / (norm._std + 1e-2)
        act = policy(obs)
        if step in (0, 1, 5, args.steps - 1):
            zmax = z.abs().max(0).values
            top = torch.argsort(zmax, descending=True)[:6]
            print(f"step {step}: |action| mean {act.abs().mean():.3f} max {act.abs().max():.2f} | largest |z|: "
                  + ", ".join(f"{term_of(int(i))}={zmax[i]:.0f}" for i in top))
        obs, _, _, _ = venv.step(act)

command = env.unwrapped.command_manager.get_term("motion")
cp = command.contact_point_local
print("contact_point_local std over all frames (left xyz, right xyz):", cp.std(0).flatten().cpu().numpy().round(4).tolist())
print("done.")
app.close()
