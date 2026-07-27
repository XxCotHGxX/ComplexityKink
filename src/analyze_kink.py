"""
Rubric-based analysis of the Complexity Kink.

Runs per-model and combined statistical workups using LLM rubric complexity
scores as instruments. Produces interactive Plotly visualizations in light mode.

Usage:
    python src/analyze_kink.py [--fast] [--scored-dir data/scored]

The --fast flag reduces bootstrap iterations from 500 to 100 for quick iteration.
"""
import argparse
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUBRIC_DIMS = [
    "branching", "iteration", "state",
    "data_structures", "edge_cases", "composition",
]

# Threshold grid and regime size are inherited from the Stage 2 estimator so
# the rubric-based analysis uses the same Hansen-search philosophy as
# run_stage2_iv.py: percentile-spaced candidates that guarantee valid splits.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    HANSEN_BOOTSTRAP_ITERATIONS,
    PLACEBO_ITERATIONS,
    STAGE_C_EXCLUDED_MODELS,
    THRESHOLD_CI_BOOTSTRAP,
    MIN_REGIME_SIZE as MIN_REGIME,
)
from run_stage2_iv import build_threshold_grid

COLORS = {
    "bg": "#ffffff",
    "card": "#f8f9fa",
    "text": "#1a1a2e",
    "muted": "#6c757d",
    "accent": "#2563eb",
    "green": "#059669",
    "red": "#dc2626",
    "orange": "#d97706",
    "purple": "#7c3aed",
    "grid": "#e5e7eb",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["card"],
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"], size=13),
    margin=dict(l=60, r=30, t=80, b=60),
)

MODEL_PALETTE = [
    "#2563eb", "#059669", "#dc2626", "#d97706", "#7c3aed",
    "#0891b2", "#be185d", "#4f46e5", "#ca8a04", "#16a34a",
    "#9333ea", "#e11d48", "#0d9488", "#c026d3", "#ea580c",
    "#6366f1", "#65a30d", "#db2777",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rubric_scores(path):
    """Load rubric scores into a dict keyed by prompt_id."""
    rubric = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get("prompt_id")
            composite = rec.get("composite", rec.get("composite_mean"))
            scores = rec.get("scores") or rec.get("scores_mean") or {}
            if pid and composite is not None:
                rubric[pid] = {
                    "composite": composite,
                    **scores,
                }
    return rubric


def load_reference_cc(path):
    """Load reference cyclomatic complexity from experiment prompts."""
    ref = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get("prompt_id")
            cc = rec.get("reference_cc")
            if pid and cc is not None:
                ref[pid] = cc
    return ref


def load_scored_model(path, rubric):
    """Load a scored model JSONL and join with rubric scores."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id")
            if pid not in rubric:
                continue
            pr = rec.get("pass_rate")
            if pr is None:
                pr = 0.0
            # Compute LOC from code if available
            code = rec.get("code_cleaned") or ""
            loc = len([l for l in code.splitlines() if l.strip()]) if code else None
            row = {
                "prompt_id": pid,
                "pass_rate": float(pr),
                "kappa_cyclomatic": rec.get("kappa_cyclomatic"),
                "loc": loc,
                "composite": rubric[pid]["composite"],
            }
            for dim in RUBRIC_DIMS:
                row[dim] = rubric[pid].get(dim)
            rows.append(row)
    df = pd.DataFrame(rows)
    # Drop rows missing rubric dimensions
    df.dropna(subset=["composite"] + RUBRIC_DIMS, inplace=True)

    # Guard against silently analysing a subsample. If the scored panel and the
    # rubric file disagree on prompt ids (e.g. a stale pre-replacement ensemble
    # file paired with the current scored panel), the join drops rows without
    # error and every downstream number is computed on a fraction of the data.
    # A join that keeps <90% of scored rows is almost always a wrong-file bug,
    # not legitimate missingness.
    n_scored = sum(1 for _ in open(path, "r", encoding="utf-8"))
    kept_frac = len(df) / n_scored if n_scored else 1.0
    if kept_frac < 0.90:
        print(
            f"    WARNING: rubric join kept {len(df)}/{n_scored} "
            f"({kept_frac:.0%}) of scored rows for {os.path.basename(path)}. "
            f"This usually means the --rubric file does not match the scored "
            f"panel (e.g. a stale ensemble file). Check the rubric path.",
            file=sys.stderr,
        )
    return df


def discover_models(scored_dir):
    """Auto-discover scored model files, excluding backups and summaries."""
    files = glob.glob(os.path.join(scored_dir, "*.jsonl"))
    result = []
    for f in files:
        base = os.path.basename(f)
        if "_original" in base or "_summary" in base or "_cloud" in base:
            continue
        name = base.replace(".jsonl", "")
        result.append((name, f))
    return sorted(result)


def build_combined_df(model_dfs):
    """Average pass_rate across models per prompt, join rubric."""
    frames = []
    for name, df in model_dfs.items():
        frames.append(df[["prompt_id", "pass_rate", "composite", "kappa_cyclomatic"]
                         + RUBRIC_DIMS].copy())
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames)
    # Average pass_rate across models for each prompt
    agg = combined.groupby("prompt_id").agg(
        pass_rate=("pass_rate", "mean"),
        kappa_cyclomatic=("kappa_cyclomatic", "mean"),
        composite=("composite", "first"),
        **{dim: (dim, "first") for dim in RUBRIC_DIMS},
    ).reset_index()
    agg.dropna(subset=["composite"] + RUBRIC_DIMS, inplace=True)
    return agg


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def run_ols_rubric(df):
    """OLS: pass_rate ~ rubric_composite. Rubric is exogenous."""
    X = sm.add_constant(df[["composite"]])
    y = df["pass_rate"]
    res = sm.OLS(y, X).fit(cov_type="HC1")
    return {
        "ols_coef": res.params["composite"],
        "ols_se": res.bse["composite"],
        "ols_pval": res.pvalues["composite"],
        "ols_r2": res.rsquared,
    }


def run_2sls(df):
    """2SLS: instrument kappa_cyclomatic with 6 rubric dimensions."""
    valid = df.dropna(subset=["kappa_cyclomatic"])
    if len(valid) < 100:
        return {
            "iv_coef": np.nan, "iv_se": np.nan, "iv_pval": np.nan,
            "iv_fstat": np.nan, "iv_j_pval": np.nan,
        }
    dep = valid["pass_rate"]
    exog = sm.add_constant(pd.DataFrame(index=valid.index))
    endog = valid[["kappa_cyclomatic"]]
    instruments = valid[RUBRIC_DIMS]

    try:
        model = IV2SLS(dependent=dep, exog=exog, endog=endog, instruments=instruments)
        res = model.fit(cov_type="robust")
        fstat = res.first_stage.diagnostics["f.stat"].iloc[0]
        try:
            j_pval = res.sargan.pval
        except Exception:
            j_pval = np.nan
        return {
            "iv_coef": res.params["kappa_cyclomatic"],
            "iv_se": res.std_errors["kappa_cyclomatic"],
            "iv_pval": res.pvalues["kappa_cyclomatic"],
            "iv_fstat": fstat,
            "iv_j_pval": j_pval,
        }
    except Exception as e:
        print(f"    2SLS failed: {e}")
        return {
            "iv_coef": np.nan, "iv_se": np.nan, "iv_pval": np.nan,
            "iv_fstat": np.nan, "iv_j_pval": np.nan,
        }


def run_hausman(df):
    """Hausman test comparing OLS-on-output-CC vs 2SLS."""
    valid = df.dropna(subset=["kappa_cyclomatic"])
    if len(valid) < 100:
        return {"hausman_stat": np.nan, "hausman_pval": np.nan}

    y = valid["pass_rate"]
    X_ols = sm.add_constant(valid[["kappa_cyclomatic"]])
    ols = sm.OLS(y, X_ols).fit()
    beta_ols = ols.params["kappa_cyclomatic"]
    var_ols = ols.cov_params().loc["kappa_cyclomatic", "kappa_cyclomatic"]

    # Stage 1: regress kappa_cyclomatic on rubric dims
    Z = sm.add_constant(valid[RUBRIC_DIMS])
    s1 = sm.OLS(valid["kappa_cyclomatic"], Z).fit()
    kappa_hat = s1.fittedvalues

    # Stage 2: regress pass_rate on kappa_hat
    X_2sls = sm.add_constant(pd.DataFrame({"kappa_hat": kappa_hat}))
    s2 = sm.OLS(y, X_2sls).fit()
    beta_2sls = s2.params["kappa_hat"]
    var_2sls = s2.cov_params().loc["kappa_hat", "kappa_hat"]

    var_diff = var_2sls - var_ols
    if var_diff > 0:
        H = (beta_2sls - beta_ols) ** 2 / var_diff
        p = 1 - stats.chi2.cdf(H, df=1)
        return {"hausman_stat": float(H), "hausman_pval": float(p)}
    return {"hausman_stat": np.nan, "hausman_pval": np.nan}


def compute_wald(df, gamma, kappa_col="composite"):
    """Wald statistic for a candidate threshold gamma."""
    low = df[df[kappa_col] <= gamma]
    high = df[df[kappa_col] > gamma]
    if len(low) < MIN_REGIME or len(high) < MIN_REGIME:
        return np.nan

    X_low = sm.add_constant(low[[kappa_col]])
    X_high = sm.add_constant(high[[kappa_col]])
    X_pool = sm.add_constant(df[[kappa_col]])

    try:
        r_low = sm.OLS(low["pass_rate"], X_low).fit()
        r_high = sm.OLS(high["pass_rate"], X_high).fit()
        r_pool = sm.OLS(df["pass_rate"], X_pool).fit()
    except Exception:
        return np.nan

    rss_pool = r_pool.ssr
    rss_split = r_low.ssr + r_high.ssr
    if rss_split == 0:
        return np.nan

    k = len(r_low.params)
    n = len(df)
    return ((rss_pool - rss_split) / k) / (rss_split / (n - 2 * k))


def hansen_threshold_test(df, n_boot, n_ci_boot, kappa_col="composite"):
    """Bootstrap Hansen threshold test on rubric composite."""
    # Observed Wald curve
    wald_curve = []
    for gamma in build_threshold_grid(df, kappa_col):
        w = compute_wald(df, gamma, kappa_col)
        wald_curve.append((gamma, w))

    valid = [(g, w) for g, w in wald_curve if not np.isnan(w)]
    if not valid:
        return {
            "kink_threshold": np.nan, "kink_sup_wald": np.nan,
            "kink_pval": np.nan, "kink_ci_lower": np.nan,
            "kink_ci_upper": np.nan, "wald_curve": wald_curve,
            "mean_pass_low": np.nan, "mean_pass_high": np.nan,
        }

    best_gamma, sup_wald = max(valid, key=lambda x: x[1])

    # Regime stats
    low = df[df[kappa_col] <= best_gamma]
    high = df[df[kappa_col] > best_gamma]
    mean_low = low["pass_rate"].mean() if len(low) > 0 else np.nan
    mean_high = high["pass_rate"].mean() if len(high) > 0 else np.nan

    # Bootstrap under H0 (wild bootstrap)
    X_pool = sm.add_constant(df[[kappa_col]])
    pooled = sm.OLS(df["pass_rate"], X_pool).fit()
    fitted = pooled.fittedvalues.values
    resid = pooled.resid.values
    rng = np.random.RandomState(42)

    boot_sup_walds = []
    for b in range(n_boot):
        weights = rng.choice([-1, 1], size=len(df))
        y_boot = fitted + resid * weights
        df_boot = df.copy()
        df_boot["pass_rate"] = y_boot
        bw = [compute_wald(df_boot, g, kappa_col) for g in build_threshold_grid(df, kappa_col)]
        bw_valid = [w for w in bw if not np.isnan(w)]
        if bw_valid:
            boot_sup_walds.append(max(bw_valid))
        if (b + 1) % 100 == 0:
            print(f"      Bootstrap {b + 1}/{n_boot}")

    p_val = np.nan
    if boot_sup_walds:
        p_val = float(np.mean(np.array(boot_sup_walds) >= sup_wald))

    # CI under H1 (pairs bootstrap)
    boot_gammas = []
    for b in range(n_ci_boot):
        df_b = df.sample(frac=1.0, replace=True, random_state=42 + b)
        bw = [(g, compute_wald(df_b, g, kappa_col)) for g in build_threshold_grid(df, kappa_col)]
        bw_valid = [(g, w) for g, w in bw if not np.isnan(w)]
        if bw_valid:
            boot_gammas.append(max(bw_valid, key=lambda x: x[1])[0])

    ci_lo = np.percentile(boot_gammas, 2.5) if boot_gammas else np.nan
    ci_hi = np.percentile(boot_gammas, 97.5) if boot_gammas else np.nan

    return {
        "kink_threshold": float(best_gamma),
        "kink_sup_wald": float(sup_wald),
        "kink_pval": p_val,
        "kink_ci_lower": float(ci_lo),
        "kink_ci_upper": float(ci_hi),
        "wald_curve": wald_curve,
        "mean_pass_low": float(mean_low),
        "mean_pass_high": float(mean_high),
    }


def placebo_test(df, n_iter, kappa_col="composite"):
    """Shuffle rubric composite and re-run threshold detection."""
    rng = np.random.RandomState(123)
    placebo_sup_walds = []
    for p in range(n_iter):
        df_p = df.copy()
        df_p[kappa_col] = rng.permutation(df[kappa_col].values)
        walds = [compute_wald(df_p, g, kappa_col) for g in build_threshold_grid(df, kappa_col)]
        valid_w = [w for w in walds if not np.isnan(w)]
        if valid_w:
            placebo_sup_walds.append(max(valid_w))
        if (p + 1) % 100 == 0:
            print(f"      Placebo {p + 1}/{n_iter}")
    return placebo_sup_walds


def run_polynomial_comparison(df, kappa_col="composite"):
    """Compare cubic polynomial vs piecewise threshold model via AIC/BIC."""
    y = df["pass_rate"].values
    x = df[kappa_col].values

    # Linear model (null)
    X_lin = sm.add_constant(x)
    res_lin = sm.OLS(y, X_lin).fit()

    # Cubic polynomial
    X_cubic = sm.add_constant(np.column_stack([x, x**2, x**3]))
    res_cubic = sm.OLS(y, X_cubic).fit()

    # Piecewise (threshold at best Wald point)
    # Find best split
    best_gamma = None
    best_rss = np.inf
    for gamma in build_threshold_grid(df, kappa_col):
        low_mask = x <= gamma
        high_mask = x > gamma
        if low_mask.sum() < MIN_REGIME or high_mask.sum() < MIN_REGIME:
            continue
        X_pw = sm.add_constant(np.column_stack([
            x,
            (x - gamma) * high_mask,  # slope change above threshold
            high_mask.astype(float),   # intercept shift
        ]))
        try:
            res_pw = sm.OLS(y, X_pw).fit()
            if res_pw.ssr < best_rss:
                best_rss = res_pw.ssr
                best_gamma = gamma
        except Exception:
            continue

    pw_result = {}
    if best_gamma is not None:
        high_mask = x > best_gamma
        X_pw = sm.add_constant(np.column_stack([
            x,
            (x - best_gamma) * high_mask,
            high_mask.astype(float),
        ]))
        res_pw = sm.OLS(y, X_pw).fit()
        pw_result = {
            "piecewise_aic": res_pw.aic,
            "piecewise_bic": res_pw.bic,
            "piecewise_r2": res_pw.rsquared,
        }

    return {
        "linear_aic": res_lin.aic,
        "linear_bic": res_lin.bic,
        "linear_r2": res_lin.rsquared,
        "cubic_aic": res_cubic.aic,
        "cubic_bic": res_cubic.bic,
        "cubic_r2": res_cubic.rsquared,
        **pw_result,
    }


def run_fractional_probit(df, kappa_col="composite"):
    """Fractional probit robustness check (Papke & Wooldridge 1996).

    GLM with probit link and binomial family on the fractional pass_rate.
    """
    y = df["pass_rate"].values
    X = sm.add_constant(df[[kappa_col]])
    try:
        model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Probit()))
        res = model.fit()
        coef = res.params[kappa_col]
        se = res.bse[kappa_col]
        pval = res.pvalues[kappa_col]
        # Marginal effect at the mean
        xbar = df[kappa_col].mean()
        phi = stats.norm.pdf(res.params["const"] + coef * xbar)
        marginal = coef * phi
        return {
            "fprobit_coef": float(coef),
            "fprobit_se": float(se),
            "fprobit_pval": float(pval),
            "fprobit_marginal_at_mean": float(marginal),
        }
    except Exception as e:
        print(f"    Fractional probit failed: {e}")
        return {
            "fprobit_coef": np.nan, "fprobit_se": np.nan,
            "fprobit_pval": np.nan, "fprobit_marginal_at_mean": np.nan,
        }


def compute_cc_loc_correlation(df):
    """Report correlation between output CC and lines of code."""
    valid = df.dropna(subset=["kappa_cyclomatic", "loc"])
    if len(valid) < 50:
        return {"cc_loc_pearson": np.nan, "cc_loc_n": 0}
    r = np.corrcoef(valid["kappa_cyclomatic"], valid["loc"])[0, 1]
    return {"cc_loc_pearson": float(r), "cc_loc_n": len(valid)}


def analyze_model(name, df, n_boot, n_ci_boot, n_placebo):
    """Full statistical battery for one model (or combined)."""
    print(f"  [{name}] N={len(df)}")
    result = {"model": name, "N": len(df)}

    # OLS on rubric composite
    result.update(run_ols_rubric(df))
    print(f"    OLS: coef={result['ols_coef']:.6f}, R2={result['ols_r2']:.4f}")

    # 2SLS: instrument output CC with rubric dims
    result.update(run_2sls(df))
    if not np.isnan(result["iv_fstat"]):
        print(f"    2SLS: coef={result['iv_coef']:.6f}, F={result['iv_fstat']:.1f}")

    # Hausman
    result.update(run_hausman(df))
    if not np.isnan(result["hausman_pval"]):
        print(f"    Hausman: stat={result['hausman_stat']:.2f}, p={result['hausman_pval']:.6f}")

    # Hansen threshold
    print(f"    Running Hansen threshold test...")
    hansen = hansen_threshold_test(df, n_boot, n_ci_boot)
    wald_curve = hansen.pop("wald_curve")
    result.update(hansen)
    result["_wald_curve"] = wald_curve
    if not np.isnan(result["kink_threshold"]):
        ci_str = ""
        if not np.isnan(result["kink_ci_lower"]):
            ci_str = f" [{result['kink_ci_lower']:.1f}, {result['kink_ci_upper']:.1f}]"
        print(f"    Kink: threshold={result['kink_threshold']:.1f}{ci_str}, "
              f"p={result['kink_pval']:.4f}")
        print(f"    Pass rate: {result['mean_pass_low']:.3f} (low) -> "
              f"{result['mean_pass_high']:.3f} (high)")

    # Placebo
    print(f"    Running placebo test...")
    placebo_walds = placebo_test(df, n_placebo)
    result["_placebo_walds"] = placebo_walds
    if placebo_walds and not np.isnan(result.get("kink_sup_wald", np.nan)):
        p_placebo = float(np.mean(np.array(placebo_walds) >= result["kink_sup_wald"]))
        result["placebo_pval"] = p_placebo
        print(f"    Placebo p-value: {p_placebo:.4f}")

    # Polynomial comparison (cubic vs piecewise threshold)
    poly = run_polynomial_comparison(df)
    result.update(poly)
    best_model = "piecewise"
    if "piecewise_bic" in poly:
        models_bic = {
            "linear": poly["linear_bic"],
            "cubic": poly["cubic_bic"],
            "piecewise": poly["piecewise_bic"],
        }
        best_model = min(models_bic, key=models_bic.get)
        print(f"    Model comparison (BIC): linear={poly['linear_bic']:.0f}, "
              f"cubic={poly['cubic_bic']:.0f}, piecewise={poly['piecewise_bic']:.0f} "
              f"-> {best_model} wins")
    result["best_functional_form"] = best_model

    # Sensitivity of the detected kink to the two hyperparameters that
    # define the grid search (min-regime size) and the Stage 2 estimator
    # (percentile grid density). A stable kink should shift by no more than
    # a grid-spacing unit under plausible alternatives; large shifts are a
    # signal that the reported threshold is partly a grid artifact.
    import config as _cfg
    sens = []
    orig_mrs = _cfg.MIN_REGIME_SIZE
    orig_npts = _cfg.THRESHOLD_GRID_N_POINTS
    try:
        for mrs in (max(30, orig_mrs // 2), orig_mrs, orig_mrs * 2):
            for npts in (20, 40, 80):
                _cfg.MIN_REGIME_SIZE = mrs
                _cfg.THRESHOLD_GRID_N_POINTS = npts
                try:
                    h = hansen_threshold_test(df, n_boot=0, n_ci_boot=0)
                    sens.append({
                        "min_regime": mrs,
                        "grid_points": npts,
                        "threshold": h["kink_threshold"],
                        "sup_wald": h["kink_sup_wald"],
                    })
                except Exception:
                    sens.append({"min_regime": mrs, "grid_points": npts,
                                 "threshold": np.nan, "sup_wald": np.nan})
    finally:
        _cfg.MIN_REGIME_SIZE = orig_mrs
        _cfg.THRESHOLD_GRID_N_POINTS = orig_npts
    result["_sensitivity_table"] = sens
    thrs = [s["threshold"] for s in sens if not np.isnan(s["threshold"])]
    if thrs:
        result["sensitivity_threshold_range"] = (float(min(thrs)), float(max(thrs)))
        print(f"    Sensitivity: threshold in [{min(thrs):.2f}, {max(thrs):.2f}] "
              f"across {len(sens)} (min_regime x grid_points) combos")

    # Fractional probit robustness
    fprobit = run_fractional_probit(df)
    result.update(fprobit)
    if not np.isnan(fprobit["fprobit_coef"]):
        print(f"    Fractional probit: coef={fprobit['fprobit_coef']:.5f}, "
              f"marginal={fprobit['fprobit_marginal_at_mean']:.5f}, "
              f"p={fprobit['fprobit_pval']:.6f}")

    # CC-LOC correlation (if LOC data available)
    if "loc" in df.columns:
        cc_loc = compute_cc_loc_correlation(df)
        result.update(cc_loc)
        if not np.isnan(cc_loc["cc_loc_pearson"]):
            print(f"    CC-LOC correlation: r={cc_loc['cc_loc_pearson']:.3f} "
                  f"(N={cc_loc['cc_loc_n']})")

    return result


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def style_axes(fig):
    fig.update_xaxes(
        gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
        linecolor=COLORS["grid"],
    )
    fig.update_yaxes(
        gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
        linecolor=COLORS["grid"],
    )
    return fig


def viz_per_model_curves(all_results, model_dfs, outdir):
    """Small-multiples: pass rate vs rubric composite per model."""
    models = [k for k in all_results if k != "_combined"]
    n = len(models)
    if n == 0:
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[m.replace("_", " ") for m in models],
        horizontal_spacing=0.08, vertical_spacing=0.10,
    )

    for i, model in enumerate(models):
        r, c = divmod(i, cols)
        df = model_dfs[model]
        res = all_results[model]
        color = MODEL_PALETTE[i % len(MODEL_PALETTE)]

        # Bin and compute mean + CI
        df_copy = df.copy()
        df_copy["bin"] = pd.cut(df_copy["composite"], bins=20)
        binned = df_copy.groupby("bin", observed=True).agg(
            x=("composite", "mean"),
            y=("pass_rate", "mean"),
            se=("pass_rate", "sem"),
        ).dropna().reset_index()

        # CI band
        fig.add_trace(go.Scatter(
            x=list(binned["x"]) + list(binned["x"][::-1]),
            y=list(binned["y"] + 1.96 * binned["se"])
              + list((binned["y"] - 1.96 * binned["se"])[::-1]),
            fill="toself", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ), row=r + 1, col=c + 1)

        # Mean line
        fig.add_trace(go.Scatter(
            x=binned["x"], y=binned["y"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            showlegend=False,
            hovertemplate="Composite=%{x:.1f}<br>Pass Rate=%{y:.1%}<extra></extra>",
        ), row=r + 1, col=c + 1)

        # Kink line
        kink = res.get("kink_threshold")
        if kink and not np.isnan(kink):
            fig.add_vline(
                x=kink, line_dash="dash", line_color=COLORS["red"],
                line_width=1.5, row=r + 1, col=c + 1,
            )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text="Per-Model Pass Rate vs Rubric Complexity",
            font_size=18,
        ),
        height=280 * rows, width=350 * cols,
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".0%")
    style_axes(fig)
    fig.write_html(os.path.join(outdir, "per_model_kink_curves.html"))
    print(f"  Saved per_model_kink_curves.html")


def viz_forest_plot(all_results, outdir):
    """Forest plot of kink thresholds across models."""
    models = []
    for name, res in sorted(all_results.items()):
        kink = res.get("kink_threshold")
        if kink and not np.isnan(kink):
            models.append((name, res))

    if not models:
        return

    fig = go.Figure()
    names = []
    for i, (name, res) in enumerate(models):
        display = name.replace("_", " ")
        if name == "_combined":
            display = "COMBINED"
        names.append(display)
        kink = res["kink_threshold"]
        ci_lo = res.get("kink_ci_lower", kink)
        ci_hi = res.get("kink_ci_upper", kink)
        if np.isnan(ci_lo):
            ci_lo = kink
        if np.isnan(ci_hi):
            ci_hi = kink

        color = COLORS["accent"] if name != "_combined" else COLORS["red"]
        size = 8 if name != "_combined" else 12

        fig.add_trace(go.Scatter(
            x=[kink], y=[i],
            mode="markers",
            marker=dict(size=size, color=color, symbol="diamond"),
            showlegend=False,
            hovertemplate=f"{display}<br>Threshold={kink:.1f} [{ci_lo:.1f}, {ci_hi:.1f}]<extra></extra>",
        ))
        # CI whisker
        fig.add_shape(
            type="line", x0=ci_lo, x1=ci_hi, y0=i, y1=i,
            line=dict(color=color, width=2),
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Kink Threshold Across Models (95% CI)", font_size=18),
        xaxis_title="Rubric Composite Threshold",
        yaxis=dict(
            tickvals=list(range(len(names))),
            ticktext=names,
        ),
        height=max(400, 40 * len(names) + 150),
        width=800,
    )
    style_axes(fig)
    fig.write_html(os.path.join(outdir, "kink_forest_plot.html"))
    print(f"  Saved kink_forest_plot.html")


def viz_combined_kink(combined_df, combined_result, outdir):
    """Headline figure: combined pass rate vs rubric composite."""
    if combined_df.empty:
        return

    df = combined_df.copy()
    kink = combined_result.get("kink_threshold")

    # Bin
    df["bin"] = pd.cut(df["composite"], bins=30)
    binned = df.groupby("bin", observed=True).agg(
        x=("composite", "mean"),
        y=("pass_rate", "mean"),
        se=("pass_rate", "sem"),
        n=("pass_rate", "count"),
    ).dropna().reset_index()

    fig = go.Figure()

    # CI band
    fig.add_trace(go.Scatter(
        x=list(binned["x"]) + list(binned["x"][::-1]),
        y=list(binned["y"] + 1.96 * binned["se"])
          + list((binned["y"] - 1.96 * binned["se"])[::-1]),
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", showlegend=True, hoverinfo="skip",
    ))

    # Mean trend
    fig.add_trace(go.Scatter(
        x=binned["x"], y=binned["y"],
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=6, color=COLORS["accent"]),
        name="Mean Pass Rate (all models)",
        hovertemplate="Composite=%{x:.1f}<br>Pass Rate=%{y:.1%}<extra></extra>",
    ))

    # Kink line + annotation
    if kink and not np.isnan(kink):
        fig.add_vline(x=kink, line_dash="dash", line_color=COLORS["red"], line_width=2)
        fig.add_annotation(
            x=kink, y=0.5,
            text=f"Complexity Kink<br>composite = {kink:.1f}",
            showarrow=True, arrowhead=2, arrowcolor=COLORS["red"],
            font=dict(color=COLORS["red"], size=14),
            ax=70, ay=-40,
        )

        # Regime labels
        mean_lo = combined_result.get("mean_pass_low", 0)
        mean_hi = combined_result.get("mean_pass_high", 0)
        fig.add_annotation(
            x=kink / 2, y=mean_lo + 0.03,
            text=f"<b>Below kink</b><br>Pass Rate = {mean_lo:.1%}",
            showarrow=False, font=dict(color=COLORS["green"], size=12),
        )
        fig.add_annotation(
            x=kink + (16 - kink) / 2, y=mean_hi + 0.03,
            text=f"<b>Above kink</b><br>Pass Rate = {mean_hi:.1%}",
            showarrow=False, font=dict(color=COLORS["red"], size=12),
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=(
                "The Complexity Kink<br>"
                '<sup style="color:#6c757d">'
                "Average pass rate across all models vs rubric-scored complexity"
                "</sup>"
            ),
            font_size=18,
        ),
        xaxis_title="Rubric Composite Score (0-24)",
        yaxis_title="Pass Rate",
        yaxis_tickformat=".0%",
        height=550, width=950,
        legend=dict(x=0.70, y=0.95, bgcolor="rgba(0,0,0,0)"),
    )
    style_axes(fig)
    fig.write_html(os.path.join(outdir, "combined_kink.html"))
    print(f"  Saved combined_kink.html")


def viz_three_measures(combined_df, ref_cc, outdir):
    """Side-by-side: naive output CC vs reference CC vs rubric composite."""
    if combined_df.empty:
        return

    df = combined_df.copy()
    df["reference_cc"] = df["prompt_id"].map(ref_cc)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            '<span style="color:#dc2626">Output CC (Biased)</span>',
            '<span style="color:#d97706">Reference CC (Canonical)</span>',
            '<span style="color:#059669">Rubric Composite (Ours)</span>',
        ],
        horizontal_spacing=0.08,
    )

    def add_binned(col, color, row, col_idx, nbins=25):
        valid = df.dropna(subset=[col])
        valid_copy = valid.copy()
        valid_copy["_bin"] = pd.cut(valid_copy[col], bins=nbins)
        binned = valid_copy.groupby("_bin", observed=True).agg(
            x=(col, "mean"), y=("pass_rate", "mean"),
        ).dropna().reset_index()
        fig.add_trace(go.Scatter(
            x=binned["x"], y=binned["y"],
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color),
            showlegend=False,
        ), row=row, col=col_idx)

    add_binned("kappa_cyclomatic", COLORS["red"], 1, 1)
    add_binned("reference_cc", COLORS["orange"], 1, 2)
    add_binned("composite", COLORS["green"], 1, 3)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=(
                "Three Complexity Measures Compared<br>"
                '<sup style="color:#6c757d">'
                "Left: endogenous output CC | Center: canonical solution CC | "
                "Right: rubric-scored prompt complexity"
                "</sup>"
            ),
            font_size=18,
        ),
        height=450, width=1100,
    )
    for i in range(1, 4):
        fig.update_yaxes(tickformat=".0%", row=1, col=i)
    fig.update_yaxes(title_text="Pass Rate", row=1, col=1)
    fig.update_xaxes(title_text="Output CC", row=1, col=1)
    fig.update_xaxes(title_text="Reference CC", row=1, col=2)
    fig.update_xaxes(title_text="Rubric Composite", row=1, col=3)
    style_axes(fig)
    fig.write_html(os.path.join(outdir, "three_measures.html"))
    print(f"  Saved three_measures.html")


def viz_summary_table(all_results, outdir):
    """Summary table of all model results."""
    rows = []
    for name, res in sorted(all_results.items()):
        display = name.replace("_", " ")
        if name == "_combined":
            display = "COMBINED"
        kink = res.get("kink_threshold", np.nan)
        ci = ""
        if not np.isnan(res.get("kink_ci_lower", np.nan)):
            ci = f"[{res['kink_ci_lower']:.1f}, {res['kink_ci_upper']:.1f}]"
        rows.append([
            display,
            res.get("N", 0),
            f"{res.get('ols_coef', 0):.5f}",
            f"{res.get('ols_r2', 0):.4f}",
            f"{res.get('iv_fstat', 0):.1f}" if not np.isnan(res.get("iv_fstat", np.nan)) else "-",
            f"{res.get('hausman_pval', 0):.4f}" if not np.isnan(res.get("hausman_pval", np.nan)) else "-",
            f"{kink:.1f}" if not np.isnan(kink) else "-",
            ci,
            f"{res.get('kink_pval', 0):.4f}" if not np.isnan(res.get("kink_pval", np.nan)) else "-",
            f"{res.get('mean_pass_low', 0):.1%}" if not np.isnan(res.get("mean_pass_low", np.nan)) else "-",
            f"{res.get('mean_pass_high', 0):.1%}" if not np.isnan(res.get("mean_pass_high", np.nan)) else "-",
            res.get("best_functional_form", "-"),
            f"{res.get('fprobit_coef', 0):.5f}" if not np.isnan(res.get("fprobit_coef", np.nan)) else "-",
            f"{res.get('cc_loc_pearson', 0):.3f}" if not np.isnan(res.get("cc_loc_pearson", np.nan)) else "-",
        ])

    headers = [
        "Model", "N", "OLS Coef", "R2", "1st-Stage F",
        "Hausman p", "Kink", "95% CI", "Hansen p",
        "Pass (Low)", "Pass (High)", "Best Form", "F-Probit", "CC-LOC r",
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color=COLORS["accent"],
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=list(zip(*rows)) if rows else [[] for _ in headers],
            fill_color=[
                [COLORS["card"] if i % 2 == 0 else COLORS["bg"] for i in range(len(rows))]
            ] * len(headers),
            font=dict(color=COLORS["text"], size=11),
            align="center",
        ),
    )])

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Model Summary Statistics", font_size=18),
        height=max(400, 40 * len(rows) + 150), width=1200,
    )
    fig.write_html(os.path.join(outdir, "model_summary_table.html"))
    print(f"  Saved model_summary_table.html")


def viz_wald_curve(combined_result, outdir):
    """Wald statistic vs threshold for combined analysis."""
    wald_curve = combined_result.get("_wald_curve", [])
    if not wald_curve:
        return

    gammas = [g for g, w in wald_curve if not np.isnan(w)]
    walds = [w for g, w in wald_curve if not np.isnan(w)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gammas, y=walds,
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=5),
        name="Wald Statistic",
    ))

    # Mark the peak
    best_idx = np.argmax(walds)
    fig.add_trace(go.Scatter(
        x=[gammas[best_idx]], y=[walds[best_idx]],
        mode="markers",
        marker=dict(size=12, color=COLORS["red"], symbol="star"),
        name=f"Threshold = {gammas[best_idx]:.1f}",
    ))

    # Placebo critical value
    placebo = combined_result.get("_placebo_walds", [])
    if placebo:
        cv_95 = np.percentile(placebo, 95)
        fig.add_hline(
            y=cv_95, line_dash="dot", line_color=COLORS["muted"],
            annotation_text=f"95th percentile placebo ({cv_95:.1f})",
            annotation_font_color=COLORS["muted"],
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Hansen Threshold Detection (Wald Curve)", font_size=18),
        xaxis_title="Candidate Threshold (Rubric Composite)",
        yaxis_title="Wald Statistic",
        height=450, width=800,
        legend=dict(x=0.65, y=0.95, bgcolor="rgba(0,0,0,0)"),
    )
    style_axes(fig)
    fig.write_html(os.path.join(outdir, "wald_curve.html"))
    print(f"  Saved wald_curve.html")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rubric-based Complexity Kink analysis")
    parser.add_argument("--fast", action="store_true",
                        help="Reduce bootstrap to 100 iterations")
    parser.add_argument("--scored-dir", default=None,
                        help="Directory containing scored JSONL files")
    parser.add_argument("--rubric", default=None,
                        help="Path to rubric scores JSONL")
    parser.add_argument("--prompts", default=None,
                        help="Path to experiment prompts JSONL")
    parser.add_argument("--outdir", default=None,
                        help="Directory for analysis outputs")
    parser.add_argument("--exclude-model", nargs="*", default=[],
                        help="Model IDs to exclude from this analysis run")
    parser.add_argument("--include-model", nargs="*", default=[],
                        help="If set, analyze only these model IDs")
    parser.add_argument("--min-rows", type=int, default=200,
                        help="Minimum valid scored rows required per model")
    parser.add_argument("--restrict-prompts", default=None,
                        help="Optional file of prompt_ids (one per line). When set, "
                             "the analysis is restricted to those prompts. Used for "
                             "subset robustness checks (e.g. organic-only prompts). "
                             "Defaults to off so the primary run is unchanged.")
    parser.add_argument("--combined-only", action="store_true",
                        help="Run the full statistical battery only on the combined prompt-level panel")
    parser.add_argument("--skip-combined", action="store_true",
                        help="Skip the combined prompt-level analysis")
    parser.add_argument("--skip-visualizations", action="store_true",
                        help="Write analysis_summary.json without Plotly HTML outputs")
    parser.add_argument("--n-boot", type=int, default=None,
                        help="Override Hansen wild bootstrap iterations")
    parser.add_argument("--n-ci-boot", type=int, default=None,
                        help="Override threshold CI pairs bootstrap iterations")
    parser.add_argument("--n-placebo", type=int, default=None,
                        help="Override placebo threshold iterations")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scored_dir = args.scored_dir or os.path.join(base, "data", "scored")
    rubric_path = args.rubric or os.path.join(base, "data", "complexity_rubric_scores.jsonl")
    prompts_path = args.prompts or os.path.join(base, "data", "experiment_prompts.jsonl")
    outdir = args.outdir or os.path.join(base, "results")
    os.makedirs(outdir, exist_ok=True)
    excluded_models = set(args.exclude_model)
    included_models = set(args.include_model)

    n_boot = args.n_boot if args.n_boot is not None else (
        100 if args.fast else HANSEN_BOOTSTRAP_ITERATIONS
    )
    n_ci_boot = args.n_ci_boot if args.n_ci_boot is not None else (
        50 if args.fast else THRESHOLD_CI_BOOTSTRAP
    )
    n_placebo = args.n_placebo if args.n_placebo is not None else (
        100 if args.fast else PLACEBO_ITERATIONS
    )

    print(
        f"Bootstrap iterations: Hansen={n_boot}, CI={n_ci_boot}, "
        f"placebo={n_placebo} (--fast={'yes' if args.fast else 'no'})"
    )

    # Load data
    print("Loading rubric scores...")
    rubric = load_rubric_scores(rubric_path)
    print(f"  {len(rubric)} prompts with rubric scores")

    if args.restrict_prompts:
        with open(args.restrict_prompts, "r", encoding="utf-8") as f:
            keep = {line.strip() for line in f if line.strip()}
        before = len(rubric)
        rubric = {pid: v for pid, v in rubric.items() if pid in keep}
        print(f"  Restricted to {len(rubric)} prompts "
              f"(from {before}) via {args.restrict_prompts}")

    print("Loading reference CC...")
    ref_cc = load_reference_cc(prompts_path)
    print(f"  {len(ref_cc)} prompts with reference CC")

    print("Discovering scored models...")
    models = discover_models(scored_dir)
    print(f"  Found {len(models)} model files")
    if STAGE_C_EXCLUDED_MODELS:
        excluded = ", ".join(sorted(STAGE_C_EXCLUDED_MODELS))
        print(f"  Excluding from Stage C panel: {excluded}")

    # Load each model
    model_dfs = {}
    for name, path in models:
        if name in STAGE_C_EXCLUDED_MODELS:
            print(f"  Skipping {name}: excluded from Stage C panel")
            continue
        if name in excluded_models:
            print(f"  Skipping {name}: excluded by --exclude-model")
            continue
        if included_models and name not in included_models:
            print(f"  Skipping {name}: not in --include-model")
            continue
        df = load_scored_model(path, rubric)
        if len(df) < args.min_rows:
            print(
                f"  Skipping {name}: only {len(df)} valid rows "
                f"(need {args.min_rows}+)"
            )
            continue
        model_dfs[name] = df
        print(f"  {name}: {len(df)} rows")

    if not model_dfs:
        print("No models with sufficient data. Exiting.")
        return

    # Run analysis
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    all_results = {}
    if args.combined_only:
        print("Skipping per-model resampling because --combined-only was set.")
    else:
        for name, df in model_dfs.items():
            all_results[name] = analyze_model(name, df, n_boot, n_ci_boot, n_placebo)

    # Combined analysis
    combined_df = pd.DataFrame()
    if args.skip_combined:
        print("\nSkipping combined analysis because --skip-combined was set.")
    else:
        print("\n--- Combined Analysis (Option D: average across models) ---")
        combined_df = build_combined_df(model_dfs)
        if not combined_df.empty:
            all_results["_combined"] = analyze_model(
                "Combined", combined_df, n_boot, n_ci_boot, n_placebo,
            )

    # Generate visualizations
    if args.skip_visualizations:
        print("\nSkipping visualizations because --skip-visualizations was set.")
    else:
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        viz_per_model_curves(all_results, model_dfs, outdir)
        viz_forest_plot(all_results, outdir)
        if "_combined" in all_results:
            viz_combined_kink(combined_df, all_results["_combined"], outdir)
            viz_wald_curve(all_results["_combined"], outdir)
        if not combined_df.empty:
            viz_three_measures(combined_df, ref_cc, outdir)
        viz_summary_table(all_results, outdir)

    # Save JSON summary (strip internal arrays)
    summary = {}
    for name, res in all_results.items():
        clean = {k: v for k, v in res.items() if not k.startswith("_")}
        summary[name] = clean

    summary_path = os.path.join(outdir, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {summary_path}")
    print(f"Visualizations saved to {outdir}/")


if __name__ == "__main__":
    main()
