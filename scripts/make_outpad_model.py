"""Outer foot pads (2026-11-12): a half-disc clipped onto the OUTER edge of each 150 x 50 mm board, or the board
moved outward on its ankle. Sideways stability study (measuring.md E1: the robot tips at 6-7 deg sideways).

    .venv/bin/python scripts/make_outpad_model.py --radius 40 [--shift 0] [--mass 12] [--base hw_ank3_b150_legs_pelvis_10dof]

--radius R (mm): a disc of radius R centred on the board's outer edge at mid-length (`<S>_tip_3`, a 4 mm cylinder at
the sole), so R mm of half-disc sticks out sideways; its inner half lies under the board. --shift d (mm): every sole geom
of the board moves d mm outward (more outside, less inside the ankle). --mass g per pad at the disc centre (PLA clip).
Writes models/<base minus _10dof>_op<R>s<d>_10dof{,_train}.xml and .json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "models"
SOLE_Z, HALF_W = -0.03655, 0.025


def edit(xml, R, d, mass, e=0.0):
    for S, sg in (("L", 1), ("R", -1)):
        if e:                                   # outer strip: the board's sole boxes widened outward by e
            def wid(m):
                sx, sy, sz = (float(v) for v in m.group(2).split())
                x, y, z = (float(v) for v in m.group(4).split())
                return f'{m.group(1)}"{sx:g} {sy + e / 2:g} {sz:g}"{m.group(3)}"{x:g} {y + sg * e / 2:g} {z:g}"'
            xml = re.sub(rf'(<geom name="{S}_tip(?:_[12])?" class="foot" type="box" size=)"([^"]+)"(\s+pos=)"([^"]+)"', wid, xml)
        if d:
            def mv(m):
                x, y, z = (float(v) for v in m.group(2).split())
                return f'{m.group(1)}"{x:g} {y + sg * d:g} {z:g}"'
            xml = re.sub(rf'(<geom name="{S}_(?:tip(?:_\d)?|padcol_0)"[^>]*?(?:pos|fromto)=)"([^"]+)"',
                         lambda m: mv(m) if len(m.group(2).split()) == 3 else m.group(0), xml)
            xml = re.sub(rf'(<geom name="{S}_padcol_0"[^>]*?fromto=)"([^"]+)"',
                         lambda m: m.group(1) + '"' + " ".join(
                             f"{float(v) + (sg * d if i % 3 == 1 else 0):g}" for i, v in enumerate(m.group(2).split())) + '"', xml)
        if R or e:
            line = re.search(rf'\n(\s*)<geom name="{S}_tip_2"[^\n]*\n', xml)
            ind = line.group(1)
            add = ""
            if R:
                add += (f'{ind}<geom name="{S}_tip_3" class="foot" type="cylinder" size="{R:g} 0.002" '
                        f'pos="0 {sg * (HALF_W + d):g} {SOLE_Z}" mass="0" friction="1 0.004 0.0004"/>\n')
            ym = sg * (HALF_W + d + (R / 3 if R else e / 2))
            add += (f'{ind}<geom name="{S}_outpad_mass" class="col" type="box" size="0.005 0.005 0.002" '
                    f'pos="0 {ym:g} {SOLE_Z}" mass="{mass:g}"/>\n')
            xml = xml[:line.end()] + add + xml[line.end():]
    if R:
        pl = re.search(r'\n(\s*)<pair geom1="floor" geom2="R_tip_2"([^\n]*)\n', xml)
        add = "".join(f'{pl.group(1)}<pair geom1="floor" geom2="{S}_tip_3"{pl.group(2)}\n' for S in "LR")
        xml = xml[:pl.end()] + add + xml[pl.end():]
    return xml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="hw_ank3_b150_legs_pelvis_10dof")
    ap.add_argument("--radius", type=float, default=40.0)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--mass", type=float, default=12.0)
    ap.add_argument("--strip", type=float, default=0.0, help="mm: widen the board outward (full length) by this")
    a = ap.parse_args()
    R, d, mass = a.radius / 1000, a.shift / 1000, a.mass / 1000
    name = a.base.replace("_10dof", f"_op{a.radius:g}s{a.shift:g}" + (f"w{a.strip:g}" if a.strip else "") + "_10dof")
    for suf in ("", "_train"):
        xml = (M / f"{a.base}{suf}.xml").read_text()
        (M / f"{name}{suf}.xml").write_text(edit(xml, R, d, mass, a.strip / 1000))
    meta = json.loads((M / f"{a.base}.json").read_text())
    meta.update(name=name, total_mass=meta["total_mass"] + 2 * mass * bool(R or a.strip),
                note=meta.get("note", "") + f" | outer pads: half-disc R {a.radius:g} mm, board shifted out {a.shift:g} mm, {a.mass:g} g each")
    (M / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(name)


if __name__ == "__main__":
    main()
