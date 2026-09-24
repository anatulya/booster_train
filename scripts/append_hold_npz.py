"""Append a static hold of the final pose to a BeyondMimic motion npz.

The shipped K1 references (``k1_fight_final_deploy.npz``, ``k1_mj2_seg1.npz``) all end with
the last pose of the moving clip repeated forward with every velocity channel set to exactly
zero -- 47 frames (0.94 s) for fight, 200 frames (4.00 s) for the dance. Motions baked
straight out of a retarget have no such tail, so the reference simply stops mid-movement and
the policy gets no training signal for what to do once the clip runs out.

This repeats the final frame's pose ``--frames`` times and zeroes the velocities over that
span, reproducing that pattern exactly. Positions are frozen, not interpolated: the hold is a
literal duplicate of the last pose, which is what the reference motions contain.

``--prepend_frames`` does the same at the other end, holding the first pose before the clip
starts. Retargeted clips often begin mid-movement, so pair it with ``--ease_in_s``, which
brings the opening of the clip up from rest rather than jumping to full speed off the hold.

.. code-block:: bash

    python append_hold_npz.py --input motions/K1/sub4_smallbox_047.npz \
        --frames 150 --output motions/K1/sub4_smallbox_047_hold.npz

    # 3 s start hold + 0.5 s ease-in on a clip that already has its end hold
    python append_hold_npz.py --input clip_hold.npz.bak --output clip_hold.npz \
        --prepend_frames 150 --ease_in_s 0.5 --frames 0
"""

import argparse
import numpy as np

POSE_KEYS = ("joint_pos", "body_pos_w", "body_quat_w", "object_pos_w", "object_quat_w", "contact")
VELOCITY_KEYS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w", "object_lin_vel_w", "object_ang_vel_w")
META_KEYS = ("fps", "joint_names", "body_names")
OPTIONAL_META_KEYS = ("contact_names", "object_name")


QUAT_KEYS = ("body_quat_w", "object_quat_w")


# Playback-rate curves r(u) on u in [0, 1], and their integrals G(u) = int_0^u r (closed form), with s = 3u^2 - 2u^3.
RAMP_PROFILES = {
    "smoothstep": (lambda u: 1 - (3 * u**2 - 2 * u**3), lambda u: u - u**3 + u**4 / 2),
    "longtail": (
        lambda u: (1 - (3 * u**2 - 2 * u**3)) ** 2,
        lambda u: u - 2 * (u**3 - u**4 / 2) + 9 * u**5 / 5 - 2 * u**6 + 4 * u**7 / 7,
    ),
}


def _consumed(ramp: int, profile: str, n: int) -> tuple[int, float]:
    """Clip frames a ramp consumes, rounded to a whole frame so it meets a kept frame exactly, and the (sub-percent)
    factor the rate is rescaled by for that rounding, so positions and velocities stay consistent."""
    integral_1 = RAMP_PROFILES[profile][1](1.0)
    consumed = max(int(round(ramp * integral_1)), 1)
    if consumed > n - 1:
        raise ValueError(f"a {ramp}-frame {profile} ramp consumes {consumed} frames, but the clip has only {n}")
    return consumed, consumed / (ramp * integral_1)


def _resample(src: dict, tau: np.ndarray, rate: np.ndarray) -> dict:
    """Every motion channel sampled at fractional clip frames ``tau``, velocities scaled by the playback ``rate``.

    Positions interpolate linearly, quaternions by normalised lerp along the shortest path, and the contact label
    takes the nearest frame. Returned arrays are float64; callers cast back to the source dtype.
    """
    n = src["joint_pos"].shape[0]
    lo = np.clip(np.floor(tau).astype(int), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = tau - lo

    out = {}
    for key in POSE_KEYS + VELOCITY_KEYS:
        if key not in src:
            continue
        data = src[key].astype(np.float64)
        w = frac.reshape((-1,) + (1,) * (data.ndim - 1))
        if key == "contact":
            out[key] = data[np.rint(tau).astype(int).clip(0, n - 1)]
        elif key in QUAT_KEYS:
            a, b = data[lo], data[hi]
            b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)  # shortest path
            q = (1 - w) * a + w * b
            out[key] = q / np.linalg.norm(q, axis=-1, keepdims=True)
        else:
            out[key] = (1 - w) * data[lo] + w * data[hi]
            if key in VELOCITY_KEYS:
                out[key] *= rate.reshape((-1,) + (1,) * (data.ndim - 1))
    return out


def ease_out(src: dict, ramp: int, profile: str = "smoothstep") -> dict:
    """Replace the end of the clip with a smooth stop that lands on the same final pose.

    New frame k = 1..ramp samples the original clip at tau(u) = t0 + ramp * (u - u^3 + u^4 / 2), u = k / ramp, whose
    rate d tau / dk = 1 - smoothstep(u) goes 1 -> 0 with zero slope at both ends. It covers ramp / 2 original frames,
    so it starts at t0 = last - ramp / 2 and finishes exactly on the last frame. Velocities are the original ones
    scaled by that rate, which keeps them consistent with the resampled poses.
    """
    n = src["joint_pos"].shape[0]
    rate_fn, integral_fn = RAMP_PROFILES[profile]
    consumed, scale = _consumed(ramp, profile, n)
    t0 = n - 1 - consumed
    u = np.arange(1, ramp + 1) / ramp
    tail = _resample(src, t0 + ramp * scale * integral_fn(u), scale * rate_fn(u))

    out = dict(src)
    for key, data in tail.items():
        out[key] = np.concatenate([src[key][: t0 + 1].astype(np.float64), data], axis=0).astype(src[key].dtype)
    fps = int(np.asarray(src["fps"]).flatten()[0])
    print(
        f"[INFO]: {profile} ease-out: the last {consumed} frames ({consumed / fps:.2f} s) of motion now play over"
        f" {ramp} frames ({ramp / fps:.2f} s); velocities reach 0 at the seam"
    )
    return out


def ease_in(src: dict, ramp: int, profile: str = "smoothstep") -> dict:
    """Replace the start of the clip with a smooth start from rest, beginning on the same first pose.

    The mirror of :func:`ease_out`: the playback rate is r(1 - u), rising 0 -> 1 with zero slope at both ends, and
    its integral is G(1) - G(1 - u). New frame k = 0..ramp-1 samples the clip at that position, so frame 0 is the
    original first pose at zero velocity, and frame ``ramp`` lands exactly on original frame ``consumed`` at full
    rate, from where the clip continues untouched. Velocities are the original ones scaled by the rate.
    """
    n = src["joint_pos"].shape[0]
    rate_fn, integral_fn = RAMP_PROFILES[profile]
    consumed, scale = _consumed(ramp, profile, n)
    u = np.arange(0, ramp) / ramp
    head = _resample(src, ramp * scale * (integral_fn(1.0) - integral_fn(1 - u)), scale * rate_fn(1 - u))

    out = dict(src)
    for key, data in head.items():
        out[key] = np.concatenate([data, src[key][consumed:].astype(np.float64)], axis=0).astype(src[key].dtype)
    fps = int(np.asarray(src["fps"]).flatten()[0])
    print(
        f"[INFO]: {profile} ease-in: the first {consumed} frames ({consumed / fps:.2f} s) of motion now play over"
        f" {ramp} frames ({ramp / fps:.2f} s), starting from rest"
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
    parser.add_argument(
        "--prepend_frames",
        type=int,
        default=0,
        help="Also hold the *first* pose for this many frames before the clip (150 = 3.0 s at 50 Hz), velocities zero.",
    )
    parser.add_argument(
        "--ease_in_s",
        type=float,
        default=0.0,
        help=(
            "Start the clip from rest instead of at full speed: its opening is replayed with a playback rate that"
            " rises smoothly from 0 to 1 over this many seconds (--ramp_profile curve), beginning on the unchanged"
            " first pose. Pairs with --prepend_frames, whose held pose it then leaves without a velocity jump."
        ),
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
    if args.ease_in_s > 0:
        src = ease_in(src, int(round(args.ease_in_s * fps)), args.ramp_profile)
    n_src = src["joint_pos"].shape[0]
    pre = args.prepend_frames

    out = {}
    for key in POSE_KEYS:
        if key not in src:
            continue
        data = np.asarray(src[key])
        # freeze the first pose before the clip and the final pose after it
        head = np.repeat(data[:1], pre, axis=0)
        hold = np.repeat(data[-1:], args.frames, axis=0)
        out[key] = np.concatenate([head, data, hold], axis=0)
    for key in VELOCITY_KEYS:
        if key not in src:
            continue
        data = np.asarray(src[key])
        head = np.zeros((pre,) + data.shape[1:], dtype=data.dtype)
        hold = np.zeros((args.frames,) + data.shape[1:], dtype=data.dtype)
        if args.ramp_s > 0 and args.ramp_mode == "velocity_only" and args.frames > 0:
            # smoothstep fade of the last frame's velocity across the start of the hold; the pose stays frozen
            ramp = min(int(round(args.ramp_s * fps)), args.frames)
            u = np.arange(1, ramp + 1) / ramp
            fade = (1 - (3 * u**2 - 2 * u**3)).reshape((-1,) + (1,) * (data.ndim - 1))
            hold[:ramp] = data[-1:] * fade
        out[key] = np.concatenate([head, data, hold], axis=0)
    for key in META_KEYS:
        out[key] = src[key]
    for key in OPTIONAL_META_KEYS:
        if key in src:
            out[key] = src[key]

    n_out = out["joint_pos"].shape[0]
    np.savez(args.output, **out)

    # Velocity steps across each seam as written. The last frame of the source still carries whatever velocity the
    # clip ended on and the hold starts at exactly zero, so the end seam is reported for visibility (the dance
    # reference has the same discontinuity: 3.96 rad/s -> 0.0 in one frame); likewise the start seam, unless
    # --ease_in_s brings the clip up from rest.
    def seam_step(i: int) -> float:
        return max(np.abs(out[k][i] - out[k][i - 1]).max() for k in VELOCITY_KEYS if k in out)

    print(f"[INFO]: {args.input}: {n_src} frames ({n_src / fps:.2f} s) after easing")
    if pre:
        print(f"[INFO]: prepended {pre} held frames ({pre / fps:.2f} s) of the first pose, velocities zeroed")
        print(f"[INFO]: largest velocity step across the start seam: {seam_step(pre):.4f}")
    if args.frames:
        if args.ramp_s > 0 and args.ramp_mode == "velocity_only":
            print(f"[INFO]: appended {args.frames} held frames ({args.frames / fps:.2f} s); velocities fade to 0 over"
                  f" the first {min(int(round(args.ramp_s * fps)), args.frames)} frames, pose frozen")  # fmt: skip
        else:
            print(f"[INFO]: appended {args.frames} held frames ({args.frames / fps:.2f} s), velocities zeroed")
        print(f"[INFO]: largest velocity step across the end seam: {seam_step(pre + n_src):.4f}")
    print(
        f"[INFO]: wrote {args.output}: {n_out} frames ({n_out / fps:.2f} s); clip spans frames {pre}-{pre + n_src - 1}"
    )


if __name__ == "__main__":
    main()
