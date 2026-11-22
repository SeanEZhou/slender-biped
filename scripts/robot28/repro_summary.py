"""Round 28: per fall, did the sim (the logged policy, its gate as it ran) fall the same way, and from how early.
'at onset' = a start at the onset or after (dt <= 0) with > 50 % of the walkers falling; 'early' = the largest dt
(seconds before the onset) with > 50 % falling. Reads results/robot28/repro_<run>_logged.json."""
import glob, json, numpy as np
rows = []
for f in sorted(glob.glob("results/robot28/repro_*_logged.json")):
    X = json.load(open(f))
    for fn in sorted({e["file"] for e in X}):
        P = [e for e in X if e["file"] == fn and e["kind"] == "pre"]
        C = [e for e in X if e["file"] == fn and e["kind"] == "ctl"]
        if not P:
            continue
        side = P[0]["robot_side"]
        hit = [e for e in P if e["fall"] > 0.5]
        early = max([e["dt"] for e in hit], default=None)
        at = any(e["dt"] <= 0 for e in hit)
        rows.append((fn[6:-4], side, at, early, max(e["fall"] for e in P), np.mean([e["fall"] for e in C]) if C else np.nan))
for r in rows:
    print(f"{r[0]:28s} robot {r[1]:8s} | sim falls (>50 %) {'yes' if r[3] is not None else 'no ':3s} from "
          f"{'-' if r[3] is None else f'{r[3]:+.1f} s':7s} before the onset | max {r[4]*100:3.0f} % | controls {r[5]*100:3.0f} %")
n = len(rows); y = sum(r[3] is not None for r in rows)
print(f"{y} of {n} falls reproduced (> 50 % of the sim robots falling from some start); "
      f"from >= 0.4 s before the onset: {sum(r[3] is not None and r[3] >= 0.4 for r in rows)}")
