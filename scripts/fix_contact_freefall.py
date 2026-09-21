"""Repair contact labels across free-fall gaps in a baked HOI motion npz.

The InterMimic suitcase captures drop their contact labels several frames before the object is actually
released: the palms keep descending smoothly at a constant rate while the *object* trajectory switches to free
fall, outruns the hands, and lands early. In ``sub1_suitcase_029`` the box falls 0.181 m in 8 frames at
10 m/s^2 while the palms continue down at a steady 0.65 m/s; ``sub15_suitcase_017`` shows the same 0.183 m
signature. Two different subjects dropping from an identical height is not a coincidence -- it is an artifact
of whatever produced the object track.

The consequence in training is that the reference *asks* for a drop, and the reward pays for it twice over:
``motion_global_object_position_error_exp`` rewards reproducing the fall, and with ``c_ref = 0`` the
interaction reward is ``r_i = 1.0``, i.e. full credit for having let go.

This script fixes the labels only. It finds gaps that look like free fall and marks them as contact, so the
interaction reward keeps gating the hands onto the box all the way down. It deliberately does **not** touch
``object_pos_w``: the object still follows its (wrong) falling trajectory, so the object-position reward still
asks for the drop. What changes is that the interaction term, at weight 20 against that term's 10, now argues
for holding on.

Usage::

    python scripts/fix_contact_freefall.py --input <motion>.npz              # report only
    python scripts/fix_contact_freefall.py --input <motion>.npz --in_place   # rewrite with a .bak
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np

GRAVITY = 9.81


def contact_windows(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame ranges where ``flags`` (T,) is nonzero."""
    edges = np.diff(np.concatenate([[0], (flags > 0).astype(np.int8), [0]]))
    return list(zip(np.where(edges == 1)[0].tolist(), np.where(edges == -1)[0].tolist()))


def find_freefall_gaps(
    object_pos: np.ndarray, contact: np.ndarray, fps: float, accel_tol: float, min_drop: float
) -> list[tuple[int, int]]:
    """Gaps between contact windows where the object falls under (close to) gravity and lands.

    Three conditions together, because any one alone has honest false positives: the object must descend by at
    least ``min_drop``, it must reach its resting height for the clip, and its mean vertical acceleration over
    the *falling* part must sit within ``accel_tol`` of -g. A genuine mid-air release the capture got right
    would still be flagged -- but that is the same repair either way, since the labels are what we trust.

    The acceleration is measured only up to touchdown. Including the landing frame makes the second difference
    swing hard positive as the object stops, which swamps the eight frames of clean -10 m/s^2 before it and
    drags the mean to roughly zero.
    """
    any_contact = contact.max(axis=1)
    windows = contact_windows(any_contact)
    gaps = []
    rest_z = float(object_pos[:, 2].min())
    for (_, prev_end), (next_start, _) in zip(windows[:-1], windows[1:]):
        if next_start - prev_end < 2:
            continue
        z = object_pos[prev_end - 1 : next_start + 1, 2]
        if float(z.min()) - rest_z > 0.01:  # never actually reaches the floor
            continue
        touchdown = int(np.argmax(z <= rest_z + 0.02))
        if touchdown < 3:  # too few frames of fall to estimate an acceleration from
            continue
        drop = float(z[0] - z[touchdown])
        accel = float(np.diff(z[: touchdown + 1], n=2).mean() * fps * fps)
        if drop >= min_drop and abs(accel + GRAVITY) <= accel_tol:
            gaps.append((prev_end, next_start))
    return gaps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, nargs="+", help="Motion npz file(s) to inspect or repair.")
    p.add_argument("--in_place", action="store_true", help="Rewrite the file; the original is kept as <name>.bak.")
    p.add_argument("--output", default=None, help="Write to this path instead of in place (single --input only).")
    p.add_argument(
        "--accel_tol",
        type=float,
        default=4.0,
        help="How far the gap's mean vertical acceleration may sit from -9.81 m/s^2 and still count as free fall.",
    )
    p.add_argument("--min_drop", type=float, default=0.05, help="Minimum descent over the gap, m.")
    p.add_argument(
        "--lookback",
        type=int,
        default=5,
        help="Frames before the gap whose contact pattern is OR-ed to decide which hands to mark as holding.",
    )
    args = p.parse_args()

    if args.output and len(args.input) > 1:
        p.error("--output takes a single --input")

    for path in args.input:
        data = dict(np.load(path))
        contact, object_pos = data["contact"], data["object_pos_w"]
        fps = float(data["fps"]) if "fps" in data else 50.0
        name = os.path.basename(path)

        gaps = find_freefall_gaps(object_pos, contact, fps, args.accel_tol, args.min_drop)
        print(f"\n=== {name}  ({len(contact)} frames, {fps:g} fps)")
        if not gaps:
            print("    no free-fall gaps found; nothing to do")
            continue

        filled = contact.copy()
        for start, end in gaps:
            z = object_pos[start - 1 : end + 1, 2]
            accel = float(np.diff(z, n=2).mean() * fps * fps)
            # Which hands were holding on the way in. OR over a few frames because the labels flicker right
            # before they cut out -- sub15 alternates [1 1], [0 1], [1 0] over the last three.
            pattern = (contact[max(0, start - args.lookback) : start] > 0).any(axis=0)
            if not pattern.any():  # nothing to infer from; fall back to whatever picks it up next
                pattern = (contact[end : end + args.lookback] > 0).any(axis=0)
            filled[start:end] = np.where(pattern, 1, filled[start:end]).astype(contact.dtype)
            print(
                f"    frames {start}-{end} ({(end - start) / fps:.2f}s): drop {float(z[0] - z[-1]):.3f} m at"
                f" {accel:+.1f} m/s^2 -> contact set to {pattern.astype(int).tolist()}"
            )

        changed = int((filled != contact).any(axis=1).sum())
        before = [len(contact_windows(contact[:, i])) for i in range(contact.shape[1])]
        after = [len(contact_windows(filled[:, i])) for i in range(filled.shape[1])]
        print(f"    {changed} frames relabelled; contact windows per hand {before} -> {after}")
        print(f"    contact frames per hand {contact.sum(axis=0).astype(int).tolist()}"
              f" -> {filled.sum(axis=0).astype(int).tolist()}")

        out = args.output or path
        if not (args.in_place or args.output):
            print("    (report only; pass --in_place to write)")
            continue
        if out == path:
            shutil.copy2(path, path + ".bak")
            print(f"    backup -> {os.path.basename(path)}.bak")
        data["contact"] = filled
        np.savez(out, **data)
        print(f"    wrote {out}")


if __name__ == "__main__":
    main()
