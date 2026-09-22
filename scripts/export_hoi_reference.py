"""Bake a hoi_track clip's reference observation terms into a flat npz for booster_deploy.

Every reference term the HoiAsym actor reads is a pure function of the motion file and the frame index --
``_ref_root`` touches only ``motion.body_pos_w``, and ``contact_point_local`` is precomputed once at command
init from an FK pass over the whole clip. The one exception is ``ref_anchor_ori_b``, which needs the robot's
live orientation; this script exports the reference anchor quaternion so the deploy side can form it, exactly
as ``beyond_mimic.py`` already does for ``motion_anchor_ori_b``.

The terms are produced by *calling the environment's own observation functions*, frame by frame, rather than
by reimplementing the frame algebra. That is the point: a second implementation on the deploy side would be
another thing to keep in sync, and the failure mode -- a silently wrong heading frame -- is invisible until
the robot falls over. Here the numbers come from the same code path training used.

The horizon is applied at runtime by index shifting (``ref_block[min(frame + k, last)]``), so only the k=0
block is stored.

Usage::

    python scripts/export_hoi_reference.py \
        --task Booster-K1-Largebox-HoiAsym-sub1_suitcase_029_stand-v0 \
        --output ~/booster_deploy/tasks/hoi_track/motions/sub1_suitcase_029_stand_ref.npz
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True, help="Registered HoiAsym task id to bake the reference from.")
parser.add_argument("--output", required=True, help="Destination .npz.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402

import booster_train.tasks  # noqa: F401, E402
from booster_train.tasks.manager_based.hoi_track.mdp import observations as O  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# name -> (observation function, width). Order matters: it is the within-k ordering the ObservationManager
# produces for ActorObsCfg, i.e. REFERENCE_TERM_FUNCS minus ref_anchor_pos_b. ref_anchor_ori_b is absent
# because it is not a pure lookup; the deploy side forms it from ref_anchor_quat_w and the live robot pose.
REF_TERMS = (
    ("ref_joint_pos", O.ref_joint_pos, 22),
    ("ref_joint_vel", O.ref_joint_vel, 22),
    ("ref_object_pos_refroot", O.ref_object_pos_refroot, 3),
    ("ref_object_ori_refroot", O.ref_object_ori_refroot, 6),
    ("ref_object_lin_vel_refroot", O.ref_object_lin_vel_refroot, 3),
    ("ref_object_ang_vel_refroot", O.ref_object_ang_vel_refroot, 3),
    ("ref_contact_point_objlocal", O.ref_contact_point_objlocal, 6),
    ("ref_contact_point_refroot", O.ref_contact_point_refroot, 6),
    ("ref_contact_flag", O.ref_contact_flag, 2),
)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # add_joint_default_pos randomizes default_joint_pos by +-0.01 rad at startup. That offset is training-time
    # domain randomization, not a property of the robot, and hardware has the nominal pose -- exporting the
    # randomized value would bias joint_pos_rel by up to 0.01 rad on every deployed step.
    env_cfg.events.add_joint_default_pos = None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    command = env.command_manager.get_term("motion")
    motion = command.motion
    if motion.num_clips != 1:
        raise ValueError(f"expected a single-clip task, got {motion.num_clips} clips")
    total = int(motion.time_step_total)
    anchor = command.motion_anchor_body_index

    out: dict[str, np.ndarray] = {name: np.zeros((total, width), np.float32) for name, _, width in REF_TERMS}
    for frame in range(total):
        # Drive the command to this frame and refresh the cached horizon, then read the k=0 terms. This is the
        # same path _update_command takes, so the values are the ones training would have observed here.
        command.time_steps[:] = frame
        command._refresh_reference_frames()
        for name, func, _ in REF_TERMS:
            out[name][frame] = func(env, "motion", 0)[0].cpu().numpy()

    # The action term's scale and offset, so the deploy side reconstructs the residual target with exactly the
    # numbers training used. Recomputing 0.25*effort/stiffness on the deploy side (as beyond_mimic.py does)
    # reads the *deployed* PD gains, which K1BeyondMimicControllerCfg overrides to values far from the Isaac
    # actuator's -- hip pitch 80 against training's 30.2 -- so that route would silently rescale every action.
    action = env.action_manager.get_term("joint_pos")
    scale = action._scale
    out["action_scale"] = (
        (scale if torch.is_tensor(scale) else torch.full((1, 22), float(scale)))[0].cpu().numpy().astype(np.float32)
    )
    out["default_joint_pos"] = command.robot.data.default_joint_pos[0].cpu().numpy().astype(np.float32)

    # Not observation terms: the anchor pose, for ref_anchor_ori_b, the reference ghost and the safety check.
    out["ref_anchor_pos_w"] = motion.body_pos_w[:, anchor].cpu().numpy().astype(np.float32)
    out["ref_anchor_quat_w"] = motion.body_quat_w[:, anchor].cpu().numpy().astype(np.float32)
    # World-frame object pose, so a sim2sim scene can spawn the real object where the clip starts it. Not an
    # observation -- the actor only ever sees the reference object in the reference root's heading frame.
    out["ref_object_pos_w"] = motion.object_pos_w.cpu().numpy().astype(np.float32)
    out["ref_object_quat_w"] = motion.object_quat_w.cpu().numpy().astype(np.float32)

    horizon = (0, *command.cfg.future_steps)
    per_k = sum(width for _, _, width in REF_TERMS) + 6  # +6 for ref_anchor_ori_b, formed at runtime
    meta = {
        "horizon": np.asarray(horizon, np.int64),
        "fps": np.asarray(float(motion.fps) if hasattr(motion, "fps") else 50.0, np.float32),
        "num_frames": np.asarray(total, np.int64),
        # Isaac asset order; booster_deploy's K1_CFG.sim_joint_names matches it, so the deploy side remaps
        # with the existing real2sim/sim2real index tables rather than inventing a new one.
        "joint_names": np.asarray(command.robot.joint_names),
        "term_names": np.asarray([name for name, _, _ in REF_TERMS]),
        "term_widths": np.asarray([width for _, _, width in REF_TERMS], np.int64),
        "obs_dim": np.asarray(70 + len(horizon) * per_k, np.int64),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)) or ".", exist_ok=True)
    np.savez(args_cli.output, **out, **meta)

    print(f"\n[INFO] wrote {args_cli.output}", flush=True)
    print(f"[INFO] {total} frames, horizon {horizon}, per-k block {per_k}, actor obs {int(meta['obs_dim'])}", flush=True)
    for name, _, width in REF_TERMS:
        a = out[name]
        print(f"         {name:30s} {a.shape}  range [{a.min():+.3f}, {a.max():+.3f}]", flush=True)
    env.close()


main()
simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
