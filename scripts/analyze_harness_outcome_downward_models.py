import os
import sys
from pathlib import Path

import pandas as pd

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(WT / "src"))
from analyze_source_frame_sensitivity import wild_bootstrap_threshold  # noqa: E402

REL = Path(os.environ.get("CK_RELEASE_DIR", WT / "data" / "public_release"))
p = pd.read_parquet(REL / "prompts", columns=["prompt_id", "ens_composite", "construction_frame"])
g = pd.read_parquet(REL / "generations", columns=["model_display_name", "prompt_id", "pass_rate", "harness_pass_rate"])

for model in ["Kimi K2.5", "Gemini 3.1 Pro Preview", "Grok-3", "GPT-5.4", "Qwen 3.6 Plus"]:
    gm = g[g.model_display_name == model].merge(p, on="prompt_id")
    assert len(gm) == 5000
    for name, col in [("analyzed", "pass_rate"), ("harness", "harness_pass_rate")]:
        frame = pd.DataFrame({
            "composite": gm["ens_composite"].astype(float),
            "later_candidate": (gm["construction_frame"] == "later_candidate").astype(float),
            "pass_rate": gm[col].astype(float),
        }).reset_index(drop=True)
        r = wild_bootstrap_threshold(frame, ["later_candidate"], 300, 42)
        print(f"{model:24s} frame-controlled {name:9s} gamma={r['threshold']:<6} supW={r['sup_wald']:7.2f} "
              f"low={r['mean_pass_low']:.3f} high={r['mean_pass_high']:.3f} "
              f"gap={100 * r['raw_regime_gap']:+.1f}pp exc={r['wild_bootstrap']['exceedances']}/300")
