"""Like diag_resume_sweep.py, but split by where each env starts: clip frames vs the frames after ``--clip_end``.

For each checkpoint: mean |action| at step 0 and the share of envs terminated within ``--steps``, separately for
envs that started before and after ``--clip_end``, plus which termination terms fired in each group.

    python scripts/diag_resume_split.py --task <task> --clip_end 292 --checkpoints <abs path> ...
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--clip_end", type=int, required=True)
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=50)
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
uw = env.unwrapped
command = uw.command_manager.get_term("motion")
tm = uw.termination_manager

for path in [args.checkpoints[0]] + list(args.checkpoints):  # first pass warms up the env; not reported
    runner.load(path)
    policy = runner.get_inference_policy(device="cuda:0")
    obs, _ = venv.reset()
    start = command.time_steps.clone()
    late = start >= args.clip_end
    done = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0")
    reasons = {name: torch.zeros(args.num_envs, dtype=torch.bool, device="cuda:0") for name in tm.active_terms}
    a0 = None
    norm = runner.alg.policy.actor_obs_normalizer
    om = uw.observation_manager
    tnames = om.active_terms["policy"]
    tdims = [d[0] if isinstance(d, tuple) else d for d in om.group_obs_term_dim["policy"]]
    tstarts = [sum(tdims[:i]) for i in range(len(tdims))]
    with torch.no_grad():
        for step in range(args.steps):
            act = policy(obs)
            if step == 0:
                a0 = act.abs().mean(-1)
                o = obs["policy"] if hasattr(obs, "keys") else obs
                z0 = ((o - norm._mean) / (norm._std + norm.eps)).abs()
            obs, _, dones, extras = venv.step(act)
            new = dones.bool() & ~done
            for name in tm.active_terms:
                reasons[name] |= new & tm.get_term(name).bool()
            done |= dones.bool()
    if path is args.checkpoints[0] and a0 is not None and not hasattr(app, "_warm"):
        app._warm = True
        continue
    label = "/".join(path.rstrip("/").split("/")[-2:])
    print(f"== {label}")
    for name, mask in (("clip starts", ~late), ("settle/hold starts", late)):
        n = int(mask.sum())
        if n == 0:
            continue
        why = ", ".join(f"{k} {100 * (reasons[k] & mask).float().sum() / n:.0f}%" for k in reasons if (reasons[k] & mask).any())
        print(f"   {name:20s} n={n:3d} | step0 |a| mean {a0[mask].mean():7.3f} | terminated {100 * (done & mask).float().sum() / n:5.1f}% | {why}")
        per_term = sorted(
            ((z0[mask][:, s:s + d].max().item(), z0[mask][:, s:s + d].mean().item(), t) for t, s, d in zip(tnames, tstarts, tdims)),
            reverse=True,
        )[:6]
        print("      largest |z| by term (max / mean): " + ", ".join(f"{t} {mx:.0f}/{mn:.1f}" for mx, mn, t in per_term))

print("done.")
app.close()
