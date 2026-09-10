"""Render a BeyondMimic motion npz to an mp4, without Isaac Sim.

The npz already carries every body's world pose for every frame, so playback needs no
simulator, no GPU and no display. Bone connectivity is recovered from the robot URDF, so
the skeleton reflects the real kinematic tree rather than a guessed ordering.

.. code-block:: bash

    # Usage
    python render_npz_video.py --motion ./booster_assets/motions/K1/sub4_smallbox_047.npz \
        --output ./sub4_smallbox_047.mp4
"""

import argparse
import numpy as np
import os
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from booster_assets import BOOSTER_ASSETS_DIR


def build_bones(urdf_path: str, body_names: list[str]) -> list[tuple[int, int]]:
    """Recovers (parent, child) index pairs for the bodies present in the motion.

    Links that the importer merged away are skipped by walking up the URDF tree until an
    ancestor that survived into the motion file is found.
    """
    parent_of = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is not None and child is not None:
            parent_of[child.get("link")] = parent.get("link")

    index_of = {name: i for i, name in enumerate(body_names)}
    bones = []
    for name in body_names:
        ancestor = parent_of.get(name)
        while ancestor is not None and ancestor not in index_of:
            ancestor = parent_of.get(ancestor)
        if ancestor is not None and ancestor != name:
            bones.append((index_of[ancestor], index_of[name]))
    return bones


def bone_color(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("left") or lowered.startswith("aleft"):
        return "#2b7bba"
    if lowered.startswith("right") or lowered.startswith("aright"):
        return "#c4442e"
    return "#555555"


def main():
    parser = argparse.ArgumentParser(description="Render a motion npz to an mp4.")
    parser.add_argument("--motion", type=str, required=True, help="Path to the motion npz file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output mp4 file.")
    parser.add_argument("--urdf", type=str, default=f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.urdf")
    parser.add_argument("--fps", type=int, default=None, help="Override the npz fps.")
    parser.add_argument("--elev", type=float, default=12.0, help="Camera elevation in degrees.")
    parser.add_argument("--azim", type=float, default=-70.0, help="Camera azimuth in degrees.")
    parser.add_argument("--span", type=float, default=0.6, help="Half-width of the tracking window in meters.")
    parser.add_argument("--zmax", type=float, default=1.2, help="Top of the vertical axis in meters.")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame.")
    args = parser.parse_args()

    data = np.load(args.motion)
    body_pos = data["body_pos_w"][:: args.stride]
    body_names = data["body_names"].tolist()
    fps = args.fps if args.fps is not None else int(data["fps"][0]) // args.stride
    num_frames = body_pos.shape[0]

    bones = build_bones(args.urdf, body_names)
    colors = [bone_color(body_names[child]) for _, child in bones]
    print(f"Loaded {args.motion}: {num_frames} frames, {len(body_names)} bodies, {len(bones)} bones, {fps} fps")

    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_box_aspect((1.0, 1.0, 1.1))
    ax.tick_params(labelsize=7, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))

    # seeded with frame 0: matplotlib >=3.10 autoscales on add and rejects an empty collection
    segments = Line3DCollection(
        [[body_pos[0][a], body_pos[0][b]] for a, b in bones], linewidths=2.6, colors=colors
    )
    ax.add_collection3d(segments)
    (joints,) = ax.plot([], [], [], "o", markersize=3.0, color="#222222", zorder=5)
    title = ax.set_title("")

    # a static ground grid, redrawn implicitly by the tracking window
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    writer = FFMpegWriter(fps=fps, bitrate=3200, metadata={"title": os.path.basename(args.motion)})
    with writer.saving(fig, args.output, dpi=args.dpi):
        for frame in range(num_frames):
            pos = body_pos[frame]
            segments.set_segments([[pos[a], pos[b]] for a, b in bones])
            joints.set_data(pos[:, 0], pos[:, 1])
            joints.set_3d_properties(pos[:, 2])

            cx, cy = pos[0, 0], pos[0, 1]
            ax.set_xlim(cx - args.span, cx + args.span)
            ax.set_ylim(cy - args.span, cy + args.span)
            ax.set_zlim(0.0, args.zmax)
            title.set_text(f"frame {frame * args.stride}   t = {frame * args.stride / int(data['fps'][0]):.2f}s")

            writer.grab_frame()
            if (frame + 1) % 50 == 0:
                print(f"  rendered {frame + 1}/{num_frames}")

    print(f"[INFO]: Video saved to {args.output}")


if __name__ == "__main__":
    main()
