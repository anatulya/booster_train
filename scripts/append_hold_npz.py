"""Append a static hold of the final pose to a BeyondMimic motion npz.

The shipped K1 references (``k1_fight_final_deploy.npz``, ``k1_mj2_seg1.npz``) all end with
the last pose of the moving clip repeated forward with every velocity channel set to exactly
zero -- 47 frames (0.94 s) for fight, 200 frames (4.00 s) for the dance. Motions baked
straight out of a retarget have no such tail, so the reference simply stops mid-movement and
the policy gets no training signal for what to do once the clip runs out.

This repeats the final frame's pose ``--frames`` times and zeroes the velocities over that
span, reproducing that pattern exactly. Positions are frozen, not interpolated: the hold is a
literal duplicate of the last pose, which is what the reference motions contain.

.. code-block:: bash

    python append_hold_npz.py --input motions/K1/sub4_smallbox_047.npz \
        --frames 150 --output motions/K1/sub4_smallbox_047_hold.npz
"""

import argparse
import numpy as np

POSE_KEYS = ("joint_pos", "body_pos_w", "body_quat_w", "object_pos_w", "object_quat_w", "contact")
VELOCITY_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w", "object_lin_vel_w", "object_ang_vel_w")
META_KEYS = ("fps", "joint_names", "body_names")
OPTIONAL_META_KEYS = ("contact_names",)


def main():
    parser = argparse.ArgumentParser(description="Append a static hold to a motion npz.")
    parser.add_argument("--input", type=str, required=True, help="Source motion npz.")
    parser.add_argument("--output", type=str, required=True, help="Output motion npz.")
    parser.add_argument("--frames", type=int, default=150, help="Hold length in frames (150 = 3.0 s at 50 Hz).")
    args = parser.parse_args()

    src = np.load(args.input)
    fps = int(np.asarray(src["fps"]).flatten()[0])
    n_src = src["joint_pos"].shape[0]

    out = {}
    for key in POSE_KEYS:
        if key not in src:
            continue
        data = np.asarray(src[key])
        # freeze the final pose across the hold
        hold = np.repeat(data[-1:], args.frames, axis=0)
        out[key] = np.concatenate([data, hold], axis=0)
    for key in VELOCITY_KEYS:
        if key not in src:
            continue
        data = np.asarray(src[key])
        hold = np.zeros((args.frames,) + data.shape[1:], dtype=data.dtype)
        out[key] = np.concatenate([data, hold], axis=0)
    for key in META_KEYS:
        out[key] = src[key]
    for key in OPTIONAL_META_KEYS:
        if key in src:
            out[key] = src[key]

    n_out = out["joint_pos"].shape[0]
    np.savez(args.output, **out)

    # the last frame of the source still carries whatever velocity the clip ended on; the
    # hold starts at exactly zero, so report the step for visibility (the dance reference has
    # the same discontinuity: 3.96 rad/s -> 0.0 in one frame).
    seam_vel = max(np.abs(np.asarray(src[k])[-1]).max() for k in VELOCITY_KEYS if k in src)

    print(f"[INFO]: {args.input}: {n_src} frames ({n_src / fps:.2f} s)")
    print(f"[INFO]: appended {args.frames} held frames ({args.frames / fps:.2f} s), velocities zeroed")
    print(f"[INFO]: seam velocity step: {seam_vel:.4f} -> 0.0 in one frame")
    print(f"[INFO]: wrote {args.output}: {n_out} frames ({n_out / fps:.2f} s), hold begins at frame {n_src}")


if __name__ == "__main__":
    main()
