"""Print the key summary numbers of one or more eval result files (results/eval/<name>.json)."""
import json
import sys
from pathlib import Path

KEYS = ["n", "survival_median_s", "survival_p10_s", "alive_5s", "alive_15s", "alive_30s", "fall_rate", "touchdowns_median", "steps_per_s_median", "bounces_median",
        "stationary_frac", "drift_median_m", "forward_median_m", "mean_vx_median", "walk_0.25m", "walk_0.5m", "walk_1.0m",
        "walk_2.0m", "startstop_success", "v_go1_median", "v_stop_median", "v_go2_median", "sat_frac_median",
        "overspeed_events_total", "max_tilt_median", "heading_change_abs_median_deg"]
for name in sys.argv[1:]:
    p = Path(name) if name.endswith(".json") else Path(__file__).resolve().parents[1] / "results" / "eval" / f"{name}.json"
    s = json.loads(p.read_text())["summary"]
    print(p.stem, {k: (round(s[k], 3) if isinstance(s[k], float) else s[k]) for k in KEYS if k in s})
    print("   peak_tau/stall", s["peak_tau_frac_max"], " rms_tau[N*m]", s["rms_tau_median"],
          " rms/rated", s.get("rms_over_rated_median"), " speed_util", s["speed_util_max"])
