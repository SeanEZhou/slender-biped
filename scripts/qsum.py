"""One-line summaries of evaluation JSONs: .venv/bin/python scripts/qsum.py <results/eval/*.json ...>"""
import json
import sys

KEYS = ["alive_30s", "nolift_frac", "quiet_frac", "liftoffs_median", "tip_travel_median_m", "steps_per_s_median", "step_length_median_m",
        "mean_vx_median", "walk_1.0m", "walk_2.0m", "startstop_success", "base_speed_rms_median", "drift_median_m",
        "max_tilt_median", "rms_over_rated_median", "overspeed_events_total", "heading_abs_deg"]
for p in sys.argv[1:]:
    d = json.load(open(p))
    s, eps = d["summary"], d["episodes"]
    if "liftoffs_after_settle" in eps[0]:  # recomputed so older files get newer metrics
        s["nolift_frac"] = sum((not e["fell"]) and e["liftoffs_after_settle"] == 0 for e in eps) / len(eps)
    s["heading_abs_deg"] = s.get("heading_change_abs_median_deg", 0.0)
    out = {k: (round(s[k], 3) if isinstance(s[k], float) else s[k]) for k in KEYS if k in s}
    print(p.split("/")[-1].replace(".json", ""), out)
