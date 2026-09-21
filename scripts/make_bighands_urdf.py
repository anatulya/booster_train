"""Write a K1 URDF variant with a collision pad added to each palm.

The stock hand link's only collider is the whole ``*_Arm_4.STL`` forearm mesh, whose surface sits a few cm
inside where the palm actually meets an object. That is why tracking policies graze a box rather than grip it.
This is the sim equivalent of bolting a rubber pad onto the end effector, which is what you would do on the
real robot: it adds contact area and brings the contact surface out to where the palm really is.

Nothing else in the URDF is touched -- same links, joints, inertias and meshes -- so a policy trained on the
stock robot still loads, and only contact geometry differs.

    python scripts/make_bighands_urdf.py --size 0.09 0.05 0.12
"""

import argparse
import re
from pathlib import Path

# The stub at the end of the hand link, measured off Left_Arm_4.STL: the arm runs along +y and ends at
# y = +0.228, and its last ~5 cm is near-cylindrical, centred on the x = z = 0 axis with radius ~0.025 m.
# The default pad is a sphere sitting on that stub, so it inflates the existing end rather than bolting a
# slab onto the side of it. Right hand mirrors in y.
STUB_CENTRE = (0.0, 0.205, 0.0)
STUB_RADIUS = 0.025

# The pad is rendered as well as collided with, so it is visible in a render instead of being an invisible
# change to contact geometry. Default colour is the robot's own grey, so it reads as part of the machine.
# The pad is rendered as well as collided with, so it is visible in a render instead of being an invisible
# change to contact geometry. Default colour is the robot's own grey, so it reads as part of the machine.
TEMPLATE = """    <visual>
      <origin xyz="{x:.5f} {y:.5f} {z:.5f}" rpy="0 0 0" />
      <geometry>
        {geometry}
      </geometry>
      <material name="palm_pad">
        <color rgba="{r:.5f} {g:.5f} {b:.5f} 1" />
      </material>
    </visual>
    <collision>
      <origin xyz="{x:.5f} {y:.5f} {z:.5f}" rpy="0 0 0" />
      <geometry>
        {geometry}
      </geometry>
    </collision>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="booster_assets/robots/K1/K1_22dof.urdf")
    parser.add_argument("--output", default=None, help="Default: <input stem>_bighands.urdf beside the input.")
    parser.add_argument("--shape", choices=["sphere", "box"], default="sphere",
                        help="'sphere' inflates the existing stub; 'box' bolts a slab on instead.")  # fmt: skip
    parser.add_argument("--radius", type=float, default=0.045,
                        help=f"Sphere pad radius, m. The bare stub is ~{STUB_RADIUS:.3f} m.")  # fmt: skip
    parser.add_argument("--size", type=float, nargs=3, default=(0.09, 0.05, 0.12), metavar=("SX", "SY", "SZ"),
                        help="Box pad size in the hand link frame, m. SY is along the arm. Only with --shape box.")  # fmt: skip
    parser.add_argument("--centre", type=float, nargs=3, default=STUB_CENTRE, metavar=("X", "Y", "Z"),
                        help="Pad centre in the LEFT hand link frame, m; the right hand mirrors in y.")  # fmt: skip
    parser.add_argument("--colour", type=float, nargs=3, default=(0.75294, 0.75294, 0.75294), metavar=("R", "G", "B"),
                        help="Pad colour. Default is the robot's own grey; use e.g. 0.9 0.3 0.1 to make it stand out.")  # fmt: skip
    args = parser.parse_args()

    src = Path(args.input)
    out = Path(args.output) if args.output else src.with_name(f"{src.stem}_bighands.urdf")
    text = src.read_text()

    for link, sign in (("left_hand_link", 1.0), ("right_hand_link", -1.0)):
        pattern = re.compile(rf'(<link\s+name="{link}">.*?)(\s*</link>)', re.S)
        match = pattern.search(text)
        if match is None:
            raise SystemExit(f"{src}: no link named {link!r}")
        if args.shape == "sphere":
            geometry = f'<sphere radius="{args.radius:.5f}" />'
            what = f"sphere r={args.radius * 100:.1f} cm (bare stub r~{STUB_RADIUS * 100:.1f} cm)"
        else:
            geometry = f'<box size="{args.size[0]:.5f} {args.size[1]:.5f} {args.size[2]:.5f}" />'
            what = f"box {args.size[0]*100:.0f} x {args.size[1]*100:.0f} x {args.size[2]*100:.0f} cm"
        pad = TEMPLATE.format(
            x=args.centre[0], y=sign * args.centre[1], z=args.centre[2], geometry=geometry,
            r=args.colour[0], g=args.colour[1], b=args.colour[2],
        )  # fmt: skip
        text = text[: match.start()] + match.group(1) + "\n" + pad + match.group(2) + text[match.end():]
        print(f"[PAD] {link}: {what}"
              f" at ({args.centre[0]:+.3f}, {sign * args.centre[1]:+.3f}, {args.centre[2]:+.3f}) m", flush=True)

    out.write_text(text)
    # Meshes are referenced relatively, so the variant only resolves beside the original.
    assert out.parent == src.parent, "the variant must sit beside the original for its mesh paths to resolve"
    print(f"[PAD] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
