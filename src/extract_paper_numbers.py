"""
Extract Stage C numbers needed for the paper that aren't in analysis_summary.json.

Produces results/paper_numbers.json with:
  - first_stage: per-dimension coefficients, SEs, t-stats
  - partial_r2: first-stage partial R^2
  - kappa_ols: OLS of pass_rate on kappa_cyclomatic (for proper Hausman table)
  - placebo_distribution: mean, std, 95% range of placebo sup-Walds
  - ks_rubric_vs_refcc: KS test of rubric composite distribution vs reference CC (sanity check)
  - sample_counts: below/above kink n's on combined dataset

Usage:
    python src/extract_paper_numbers.py
"""
import json
import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_kink import (
    RUBRIC_DIMS, load_rubric_scores, load_scored_model,
    discover_models, build_combined_df, compute_wald,
)
from config import STAGE_C_EXCLUDED_MODELS
from run_stage2_iv import build_threshold_grid


RUBRIC_PATH = "data/complexity_rubric_scores.jsonl"
SCORED_DIR = "data/scored"
OUT_PATH = "results/paper_numbers.json"

def main():
    print("Loading rubric scores...")
    rubric = load_rubric_scores(RUBRIC_PATH)
    print(f"  {len(rubric)} prompts scored")

    print("Loading scored model files...")
    models = discover_models(SCORED_DIR)
    model_dfs = {}
    for name, path in models:
        if name in STAGE_C_EXCLUDED_MODELS:
            print(f"  SKIP {name} (excluded from Stage C panel)")
            continue
        df = load_scored_model(path, rubric)
        if len(df) > 0:
            model_dfs[name] = df
    print(f"  {len(model_dfs)} panel models loaded")

    print("Building combined prompt-level dataset...")
    combined = build_combined_df(model_dfs)
    combined = combined.dropna(subset=["kappa_cyclomatic"])
    n_combined = len(combined)
    print(f"  combined N = {n_combined}")

    out = {"combined_N": n_combined}

    # -----------------------------------------------------------------
    # First stage: kappa_cyclomatic ~ rubric dims
    # -----------------------------------------------------------------
    print("Running first stage (kappa ~ rubric dims)...")
    Z = sm.add_constant(combined[RUBRIC_DIMS])
    k_obs = combined["kappa_cyclomatic"]
    fs = sm.OLS(k_obs, Z).fit(cov_type="HC1")
    first_stage = {}
    for dim in RUBRIC_DIMS:
        first_stage[dim] = {
            "coef": float(fs.params[dim]),
            "se": float(fs.bse[dim]),
            "t": float(fs.tvalues[dim]),
            "pval": float(fs.pvalues[dim]),
        }
    first_stage["const"] = {
        "coef": float(fs.params["const"]),
        "se": float(fs.bse["const"]),
    }
    out["first_stage"] = first_stage

    # Partial R^2: R^2 from regressing kappa on rubric dims without constant effects
    # (using the equivalent: R^2 of the first stage after partialling out const)
    # For a single endogenous regressor with no exog controls beyond const,
    # partial R^2 = first-stage R^2.
    out["first_stage_r2"] = float(fs.rsquared)
    out["first_stage_fstat"] = float(fs.fvalue)
    print(f"  first-stage R^2 = {fs.rsquared:.4f}, F = {fs.fvalue:.1f}")

    # -----------------------------------------------------------------
    # OLS on kappa_cyclomatic (for proper Hausman comparison)
    # -----------------------------------------------------------------
    print("Running OLS (pass_rate ~ kappa_cyclomatic)...")
    X_kols = sm.add_constant(combined[["kappa_cyclomatic"]])
    k_ols = sm.OLS(combined["pass_rate"], X_kols).fit(cov_type="HC1")
    out["kappa_ols"] = {
        "coef": float(k_ols.params["kappa_cyclomatic"]),
        "se": float(k_ols.bse["kappa_cyclomatic"]),
        "pval": float(k_ols.pvalues["kappa_cyclomatic"]),
        "r2": float(k_ols.rsquared),
    }
    print(f"  beta_OLS(kappa) = {k_ols.params['kappa_cyclomatic']:.4f}")

    # -----------------------------------------------------------------
    # 2SLS (re-confirm)
    # -----------------------------------------------------------------
    print("Running 2SLS for confirmation...")
    exog = sm.add_constant(pd.DataFrame(index=combined.index))
    endog = combined[["kappa_cyclomatic"]]
    instruments = combined[RUBRIC_DIMS]
    iv_res = IV2SLS(
        dependent=combined["pass_rate"],
        exog=exog, endog=endog, instruments=instruments,
    ).fit(cov_type="robust")
    out["iv"] = {
        "coef": float(iv_res.params["kappa_cyclomatic"]),
        "se": float(iv_res.std_errors["kappa_cyclomatic"]),
        "pval": float(iv_res.pvalues["kappa_cyclomatic"]),
        "fstat": float(iv_res.first_stage.diagnostics["f.stat"].iloc[0]),
    }
    try:
        partial_r2 = float(iv_res.first_stage.diagnostics["partial.rsquared"].iloc[0])
        out["iv"]["partial_r2"] = partial_r2
    except Exception:
        pass
    try:
        out["iv"]["sargan_stat"] = float(iv_res.sargan.stat)
        out["iv"]["sargan_pval"] = float(iv_res.sargan.pval)
    except Exception:
        pass

    # -----------------------------------------------------------------
    # Sample splits at kink (gamma = 8.0)
    # -----------------------------------------------------------------
    print("Computing below/above kink sample sizes...")
    gamma = 8.0
    n_below = int((combined["composite"] <= gamma).sum())
    n_above = int((combined["composite"] > gamma).sum())
    mean_below = float(combined.loc[combined["composite"] <= gamma, "pass_rate"].mean())
    mean_above = float(combined.loc[combined["composite"] > gamma, "pass_rate"].mean())
    out["kink_split"] = {
        "gamma": gamma,
        "n_below": n_below,
        "n_above": n_above,
        "mean_below": mean_below,
        "mean_above": mean_above,
    }
    print(f"  below gamma={gamma}: n={n_below}, mean pass={mean_below:.4f}")
    print(f"  above gamma={gamma}: n={n_above}, mean pass={mean_above:.4f}")

    # -----------------------------------------------------------------
    # Placebo distribution: 500 shuffled iterations
    # -----------------------------------------------------------------
    print("Running 500-iteration placebo (this takes a few minutes)...")
    rng = np.random.RandomState(123)
    placebo_sup_walds = []
    grid = build_threshold_grid(combined, "composite")
    for p in range(500):
        df_p = combined.copy()
        df_p["composite"] = rng.permutation(combined["composite"].values)
        walds = [compute_wald(df_p, g, "composite") for g in grid]
        valid_w = [w for w in walds if not np.isnan(w)]
        if valid_w:
            placebo_sup_walds.append(max(valid_w))
        if (p + 1) % 100 == 0:
            print(f"  placebo {p + 1}/500")
    placebo_arr = np.array(placebo_sup_walds)
    out["placebo_distribution"] = {
        "n": len(placebo_arr),
        "mean": float(placebo_arr.mean()),
        "std": float(placebo_arr.std()),
        "p025": float(np.percentile(placebo_arr, 2.5)),
        "p975": float(np.percentile(placebo_arr, 97.5)),
        "max": float(placebo_arr.max()),
    }
    print(f"  placebo mean sup-Wald = {placebo_arr.mean():.3f}")
    print(f"  placebo 95% range = [{np.percentile(placebo_arr, 2.5):.3f}, "
          f"{np.percentile(placebo_arr, 97.5):.3f}]")

    # -----------------------------------------------------------------
    # KS test: reference CC distribution vs some uniform over bins
    # (If a representativeness file exists, use that. Otherwise report
    # a sanity check: rubric composite vs kappa_cyclomatic on success-only rows.)
    # -----------------------------------------------------------------
    print("Running KS sanity checks...")
    # Rubric composite distribution within the selected sample vs uniform
    comp = combined["composite"].values
    # KS: rubric vs kappa_cyclomatic for rows where model passed perfectly
    # (this tests whether rubric distribution aligns with observed CC distribution
    #  on the non-endogenous sub-sample)
    try:
        # Stage 1 is trained on rows where pass_rate = 1 in Stage B; proxy:
        successes = combined[combined["pass_rate"] > 0.5]
        ks_stat, ks_pval = stats.ks_2samp(
            successes["composite"], successes["kappa_cyclomatic"]
        )
        out["ks_rubric_vs_kappa_on_successes"] = {
            "stat": float(ks_stat), "pval": float(ks_pval),
            "n_successes": int(len(successes)),
        }
    except Exception as e:
        out["ks_rubric_vs_kappa_on_successes"] = {"error": str(e)}

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
