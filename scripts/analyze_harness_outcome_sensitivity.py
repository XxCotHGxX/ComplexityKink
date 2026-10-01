"""Harness-only outcome sensitivity for the camera-ready.

Re-runs the manuscript's sup-Wald threshold searches (using the repository's own
implementation) with (a) the analyzed pass rate and (b) the raw unit-test
harness pass rate, which differs only for earlier-frame rows where an o4-mini
audit verdict replaced the harness value.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(WT / "src"))
from analyze_source_frame_sensitivity import (  # noqa: E402
    prepare_threshold_design,
    threshold_curve,
    wild_bootstrap_threshold,
)

REL = Path(os.environ.get("CK_RELEASE_DIR", WT / "data" / "public_release"))
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 300

p = pd.read_parquet(REL / "prompts")
assert len(p) == 5000, len(p)
base = pd.DataFrame({
    "prompt_id": p["prompt_id"],
    "composite": p["ens_composite"].astype(float),
    "later_candidate": (p["construction_frame"] == "later_candidate").astype(float),
    "task_type": p["task_type"],
})
tt = pd.get_dummies(base["task_type"], prefix="tt", drop_first=True, dtype=float)
tt_cols = list(tt.columns)
base = pd.concat([base, tt], axis=1)

outcomes = {"analyzed": p["mean_pass_rate"], "harness": p["mean_harness_pass_rate"]}
results = {}
for name, y in outcomes.items():
    frame = base.assign(pass_rate=y.astype(float).to_numpy())
    res = {}
    for spec, controls in [("unadjusted", []), ("task_type_fe", tt_cols), ("frame_fe", ["later_candidate"])]:
        res[spec] = wild_bootstrap_threshold(frame, controls, N_BOOT, 42)
    for fr, flag in [("earlier_frame", 0.0), ("later_frame", 1.0)]:
        sub = frame[frame["later_candidate"] == flag].reset_index(drop=True)
        res[fr] = wild_bootstrap_threshold(sub, [], N_BOOT, 42)
        res[fr]["frame_mean_pass"] = float(sub["pass_rate"].mean())
    results[name] = res

# Model-specific unadjusted fits (point estimates only) under both outcomes.
g = pd.read_parquet(REL / "generations", columns=["model_display_name", "prompt_id", "pass_rate", "harness_pass_rate"])
design = prepare_threshold_design(base, [])
x = base["composite"].to_numpy()
per_model = {}
for model, gm in g.groupby("model_display_name"):
    gm = gm.set_index("prompt_id").reindex(base["prompt_id"])
    row = {}
    for name, col in [("analyzed", "pass_rate"), ("harness", "harness_pass_rate")]:
        yv = gm[col].to_numpy(dtype=float)
        gamma, sw = max(threshold_curve(yv, design), key=lambda t: t[1])
        lo, hi = yv[x <= gamma].mean(), yv[x > gamma].mean()
        row[name] = {"gamma": gamma, "sup_wald": round(sw, 2), "low": round(lo, 3), "high": round(hi, 3), "delta_pp": round(100 * (hi - lo), 1)}
    per_model[model] = row
results["per_model"] = per_model

out = WT / "results" / "independent_audit" / "harness_outcome_sensitivity.json"
out.write_text(json.dumps(results, indent=2))

def line(r):
    wb = r["wild_bootstrap"]
    return (f"gamma={r['threshold']:<6} supW={r['sup_wald']:8.2f}  low={r['mean_pass_low']:.3f} "
            f"high={r['mean_pass_high']:.3f}  gap={100*r['raw_regime_gap']:+.1f}pp  boot exc={wb['exceedances']}/{wb['draws']}")

for spec in ["unadjusted", "task_type_fe", "frame_fe", "earlier_frame", "later_frame"]:
    print(f"\n[{spec}]")
    for name in outcomes:
        extra = f"  frame mean={results[name][spec]['frame_mean_pass']:.3f}" if "frame_mean_pass" in results[name][spec] else ""
        print(f"  {name:9s} {line(results[name][spec])}{extra}")

print("\n[per-model unadjusted]  model | analyzed gamma, delta | harness gamma, delta")
down = {"analyzed": [], "harness": []}
for m, r in sorted(per_model.items()):
    a, h = r["analyzed"], r["harness"]
    for k in down:
        if r[k]["delta_pp"] < 0:
            down[k].append(m)
    flag = "  <-- direction flips" if (a["delta_pp"] < 0) != (h["delta_pp"] < 0) else ""
    print(f"  {m:28s} {a['gamma']:6} {a['delta_pp']:+6.1f} | {h['gamma']:6} {h['delta_pp']:+6.1f}{flag}")
for k, v in down.items():
    print(f"downward ({k}): {len(v)}: {v}")
