import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'leg_p0' / 'cad'))
import biped, leg
from build123d import export_stl
out = Path(sys.argv[1])
items = [i for i in biped.leg_items() if i.kind in ("part", "servo", "horn")]
pelvis = biped.part_pelvis_biped()
man = []
poses = {"stance": (biped.q(pitch=16, knee=32), biped.q(pitch=16, knee=32))}
for tag, (ql, qr) in poses.items():
    l, r = biped.posed_pair(items, ql, qr)
    all_ = [leg.Item("pelvis_biped", pelvis, "pelvis", color=(0.22,0.23,0.26,1), kind="part")] + l + r
    for k, it in enumerate(all_):
        p = out / f"{tag}_{k:02d}_{it.name}.stl"
        export_stl(it.shape, str(p), tolerance=0.05, angular_tolerance=0.1)
        bb = it.shape.bounding_box()
        man.append(dict(file=p.name, name=it.name, color=list(it.color), kind=it.kind, zmin=bb.min.Z, zmax=bb.max.Z, ymin=bb.min.Y, ymax=bb.max.Y, xmin=bb.min.X, xmax=bb.max.X))
(out / "manifest.json").write_text(json.dumps(man, indent=0))
for m in man:
    print(m["name"], m["kind"], [round(m[k],1) for k in ("xmin","xmax","ymin","ymax","zmin","zmax")], m["color"])
