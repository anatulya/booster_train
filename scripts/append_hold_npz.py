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


QUAT_KEYS = ("body_quat_w", "object_quat_w")


# Playback-rate curves r(u) on u in [0, 1], and their integrals G(u) = int_0^u r (closed form), with s = 3u^2 - 2u^3.
RAMP_PROFILES = {
    "smoothstep": (lambda u: 1 - (3 * u**2 - 2 * u**3), lambda u: u - u**3 + u**4 / 2),
    "longtail": (
        lambda u: (1 - (3 * u**2 - 2 * u**3)) ** 2,
        lambda u: u - 2 * (u**3 - u**4 / 2) + 9 * u**5 / 5 - 2 * u**6 + 4 * u**7 / 7,
    ),
}


def ease_out(src: dict, ramp: int, profile: str = "smoothstep") -> dict:
    """Replace the end of the clip with a smooth stop that lands on the same final pose.

    New frame k = 1..ramp samples the original clip at tau(u) = t0 + ramp * (u - u^3 + u^4 / 2), u = k / ramp, whose
    rate d tau / dk = 1 - smoothstep(u) goes 1 -> 0 with zero slope at both ends. It covers ramp / 2 original frames,
    so it starts at t0 = last - ramp / 2 and finishes exactly on the last frame. Velocities are the original ones
    scaled by that rate, which keeps them consistent with the resampled poses.
    """
    n = src["joint_pos"].shape[0]
    rate_fn, integral_fn = RAMP_PROFILES[profile]
    # Clip frames the ramp consumes, rounded to a whole frame so the ramp starts exactly on a kept frame; the rate
    # is rescaled by the (sub-percent) rounding factor so positions and velocities stay consistent.
    consumed = max(int(round(ramp * integral_fn(1.0))), 1)
    scale = consumed / (ramp * integral_fn(1.0))
    t0 = n - 1 - consumed
    if t0 < 0:
        raise ValueError(f"a {ramp}-frame {profile} ramp consumes {consumed} frames, but the clip has only {n}")
    u = np.arange(1, ramp + 1) / ramp
    tau = t0 + ramp * scale * integral_fn(u)
    rate = scale * rate_fn(u)
    lo = np.clip(np.floor(tau).astype(int), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = (tau - lo).reshape((-1,) + (1,) * 0)

    def interp(data: np.ndarray) -> np.ndarray:
        w = frac.reshape((-1,) + (1,) * (data.ndim - 1))
        return (1 - w) * data[lo] + w * data[hi]

    out = dict(src)
    for key in POSE_KEYS + VELOCITY_KEYS:
        if key not in src:
            continue
        data = src[key].astype(np.float64)
        if key == "contact":
            tail = data[np.rint(tau).astype(int).clip(0, n - 1)]
        elif key in QUAT_KEYS:
            a, b = data[lo], data[hi]
            b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)  # shortest path
            w = frac.reshape((-1,) + (1,) * (data.ndim - 1))
            tail = (1 - w) * a + w * b
            tail /= np.linalg.norm(tail, axis=-1, keepdims=True)
        else:
            tail = interp(data)
            if key in VELOCITY_KEYS:
                tail *= rate.reshape((-1,) + (1,) * (data.ndim - 1))
        out[key] = np.concatenate([data[: t0 + 1], tail], axis=0).astype(src[key].dtype)
    fps = int(np.asarray(src["fps"]).flatten()[0])
    print(
        f"[INFO]: {profile} ease: the last {consumed} frames ({consumed / fps:.2f} s) of motion now play over"
        f" {ramp} frames ({ramp / fps:.2f} s); velocities reach 0 at the seam"
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Append a static hold to a motion npz.")
    parser.add_argument("--input", type=str, required=True, help="Source motion npz.")
    parser.add_argument("--output", type=str, required=True, help="Output motion npz.")
    parser.add_argument("--frames", type=int, default=150, help="Hold length in frames (150 = 3.0 s at 50 Hz).")
    parser.add_argument(
        "--ramp_s",
        type=float,
        default=0.0,
        help=(
            "Ease into the hold instead of stopping in one frame: the tail of the clip is replayed with a playback"
            " rate that falls smoothly (smoothstep) from 1 to 0 over this many seconds, so velocities decay to zero"
            " consistently with the poses and the final pose is unchanged. 0 = the original hard stop."
        ),
    )
    parser.add_argument(
        "--ramp_profile",
        choices=["smoothstep", "longtail"],
        default="smoothstep",
        help=(
            "Playback-rate curve for --ramp_mode ease. 'smoothstep': rate = 1 - smoothstep(u), consumes ramp/2 of the"
            " clip. 'longtail': rate = (1 - smoothstep(u))^2, sheds speed early then creeps into the final pose;"
            " consumes only ~37%% of the ramp length of the clip. Both are smooth (zero slope) at both ends."
        ),
    )
    parser.add_argument(
        "--ramp_mode",
        choices=["ease", "velocity_only"],
        default="ease",
        help=(
            "'ease': retime the end of the clip so it slows into the final pose (starts before the end).\n"
            "'velocity_only': leave the clip untouched and freeze the pose as usual, but fade the reference velocities"
            " from their last-frame values to zero over the first --ramp_s of the hold instead of in one frame."
        ),
    )
    parser.add_argument(
        "--strip_frames",
        type=int,
        default=0,
        help="Drop this many frames from the end of the input first, e.g. 150 to redo the hold of a *_hold.npz.",
    )
    args = parser.parse_args()

    src = {key: np.asarray(value) for key, value in np.load(args.input).items()}
    fps = int(np.asarray(src["fps"]).flatten()[0])
    if args.strip_frames:
        for key in POSE_KEYS + VELOCITY_KEYS:
            if key in src:
                src[key] = src[key][: -args.strip_frames]
    if args.ramp_s > 0 and args.ramp_mode == "ease":
        src = ease_out(src, int(round(args.ramp_s * fps)), args.ramp_profile)
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
        if args.ramp_s > 0 and args.ramp_mode == "velocity_only":
            # smoothstep fade of the last frame's velocity across the start of the hold; the pose stays frozen
            ramp = min(int(round(args.ramp_s * fps)), args.frames)
            u = np.arange(1, ramp + 1) / ramp
            fade = (1 - (3 * u**2 - 2 * u**3)).reshape((-1,) + (1,) * (data.ndim - 1))
            hold[:ramp] = data[-1:] * fade
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
    # step across the seam as written (last clip frame -> first hold frame)
    seam_vel = max(np.abs(out[k][n_src] - out[k][n_src - 1]).max() for k in VELOCITY_KEYS if k in out)

    print(f"[INFO]: {args.input}: {n_src} frames ({n_src / fps:.2f} s)")
    if args.ramp_s > 0 and args.ramp_mode == "velocity_only":
        print(f"[INFO]: appended {args.frames} held frames ({args.frames / fps:.2f} s); velocities fade to 0 over the"
              f" first {min(int(round(args.ramp_s * fps)), args.frames)} frames, pose frozen")  # fmt: skip
    else:
        print(f"[INFO]: appended {args.frames} held frames ({args.frames / fps:.2f} s), velocities zeroed")
    print(f"[INFO]: largest velocity step across the seam: {seam_vel:.4f}")
    print(f"[INFO]: wrote {args.output}: {n_out} frames ({n_out / fps:.2f} s), hold begins at frame {n_src}")


if __name__ == "__main__":
    main()
