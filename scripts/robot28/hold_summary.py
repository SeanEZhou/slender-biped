"""Round 28: held-out robot fall states per policy (results/robot28/repro_*_hold_<tag>.json): the share of sim robots
falling, from the states at / after the onset (dt <= 0: the stuck states before the terrible step) and before it."""
import glob, json, sys, collections, numpy as np
R = collections.defaultdict(list)
for f in glob.glob("results/robot28/repro_*_hold_*.json"):
    tag = f.split("_hold_")[1][:-5]
    R[tag] += json.load(open(f))
for tag in sorted(R, key=lambda t: (not t.startswith(("z7_anc2", "z0_y2c")), t)):
    x = R[tag]
    late = [e["fall"] for e in x if e["kind"] == "pre" and e["dt"] <= 0.0]
    near = [e["fall"] for e in x if e["kind"] == "pre" and 0.0 < e["dt"] <= 0.4]
    early = [e["fall"] for e in x if e["kind"] == "pre" and e["dt"] > 0.4]
    tr = len({e["file"] for e in x})
    print(f"{tag:34s} trials {tr:2d} | fall from onset/after {np.mean(late)*100:4.0f}% (n {len(late)})  0.4 s before {np.mean(near)*100:4.0f}% (n {len(near)})  earlier {np.mean(early)*100:4.0f}% (n {len(early)})")
