"""Run several checkpoints on one task and report how each behaves from a fresh reset, without training.

One sim launch; for each checkpoint: load its weights (policy + normalizer), reset all envs, run ``--steps`` steps
deterministically, and print mean / max |action| at steps 0 and the last step plus the share of envs that
terminated (not time-out). A checkpoint that suits the task's reference has mean |action| ~ 0.1-1 and few early
terminations; one that does not shows actions in the tens and most envs falling within the first second.

    python scripts/diag_resume_sweep.py --task <task> --checkpoints <abs path> <abs path> ...
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import os
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

results = []
for path in args.checkpoints:
    runner.load(path)
    policy = runner.get_inference_policy(device="cuda:0")
    obs, _ = venv.reset()
    fell = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0")
    first = last = None
    with torch.no_grad():
        for step in range(args.steps):
            act = policy(obs)
            if step == 0:
                first = (act.abs().mean().item(), act.abs().max().item())
            last = (act.abs().mean().item(), act.abs().max().item())
            obs, _, dones, extras = venv.step(act)
            timeouts = extras.get("time_outs", torch.zeros_like(dones))
            fell |= dones.bool() & ~timeouts.bool()
    label = "/".join(path.rstrip("/").split("/")[-2:])
    results.append((label, first, last, fell.float().mean().item()))
    print(f"{label:80s} step0 |a| mean {first[0]:7.3f} max {first[1]:8.2f} | step{args.steps - 1} |a| mean {last[0]:7.3f}"
          f" max {last[1]:8.2f} | terminated within {args.steps} steps: {100 * results[-1][3]:5.1f}%")

print("done.")
app.close()
