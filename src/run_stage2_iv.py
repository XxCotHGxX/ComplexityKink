"""
Stage 2: The corrected analysis pipeline.

Implements:
1. Proper 2SLS via linearmodels.iv.IV2SLS (correct standard errors)
2. Sargan-Hansen J-test for instrument validity
3. Hausman endogeneity test (OLS vs 2SLS)
4. Bootstrap Hansen threshold test (sup-Wald with null distribution)
5. Placebo test (shuffled instruments)
6. Continuous pass rate as dependent variable

DATA PROVENANCE: Imports data loading from ``data_loader`` (single source
of truth). `CONTROL_COLS` is intentionally empty ,  see the rationale block
in config.py. Earlier versions of this pipeline included `e_norm` and
`m_mem_jaccard` as controls; both are post-treatment (derived from the
generated code) and so reintroduce the very endogeneity the IV strategy
removes. Legitimate controls must be derivable pre-generation from the
instruction or from fixed task metadata.
"""
import json
import argparse
import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEFAULT_ENRICHED_FILE, DEFAULT_OOF_FILE, DEFAULT_MODEL_PATH, OUTPUT_DIR,
    IV_FEATURE_COLS, REDUCED_IV_COLS, CONSERVATIVE_IV_COLS, CONTROL_COLS,
    THRESHOLD_GRID_MODE, THRESHOLD_GRID_N_POINTS,
    THRESHOLD_GRID_START, THRESHOLD_GRID_END, THRESHOLD_GRID_STEP,
    HANSEN_BOOTSTRAP_ITERATIONS, PLACEBO_ITERATIONS, THRESHOLD_CI_BOOTSTRAP,
    MIN_REGIME_SIZE, CLUSTER_COL,
)
from data_loader import load_enriched_dataset


def _fit_kwargs(df):
    """Cluster-robust covariance if the cluster column is present; else HC1.

    Clustering on ``CLUSTER_COL`` (prompt id) accounts for within-prompt
    correlation across the ~18 models that answer each prompt. Without it
    the 2SLS standard errors treat (prompt, model) pairs as independent
    draws, which they are not.
    """
    if CLUSTER_COL in df.columns and df[CLUSTER_COL].notna().any():
        return {"cov_type": "clustered", "clusters": df[CLUSTER_COL]}
    return {"cov_type": "robust"}


def build_threshold_grid(df, kappa_col="kappa_predicted"):
    """Return a threshold grid that guarantees MIN_REGIME_SIZE on both sides.

    The percentile mode converts MIN_REGIME_SIZE into quantile bounds of
    ``kappa_col`` so every candidate threshold yields a valid split by
    construction. A fixed absolute grid would silently exclude candidates
    near the tails of a right-skewed kappa distribution and bias the
    sup-Wald search toward the centre (which is precisely the regime where
    noise-driven local maxima are most likely).
    """
    if THRESHOLD_GRID_MODE == "percentile":
        kappa = df[kappa_col].dropna().values
        n = len(kappa)
        if n < 2 * MIN_REGIME_SIZE:
            # Too few observations for a valid split at any threshold.
            return np.array([])
        q_low = MIN_REGIME_SIZE / n
        q_high = 1.0 - q_low
        qs = np.linspace(q_low, q_high, THRESHOLD_GRID_N_POINTS)
        grid = np.quantile(kappa, qs)
        # Deduplicate after quantile rounding on discrete kappa values.
        return np.unique(np.round(grid, 4))
    return np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END, THRESHOLD_GRID_STEP)


def load_data_for_stage2(enriched_file, model_path):
    """Load data and generate predicted kappa from Stage 1 model if needed."""
    df, feature_cols = load_enriched_dataset(enriched_file)
    model = joblib.load(model_path)
    
    # Generate predicted kappa (instrumented variable) if not pre-provided
    if df['kappa_predicted'].isnull().any():
        missing = df['kappa_predicted'].isnull()
        X_features = df.loc[missing, feature_cols].values
        df.loc[missing, 'kappa_predicted'] = model.predict(X_features)
        print(f"Filled {missing.sum()} missing kappa_predicted from model")
    
    print(f"Mean kappa_actual: {df['kappa_actual'].mean():.2f}")
    print(f"Mean kappa_predicted: {df['kappa_predicted'].mean():.2f}")
    
    return df, feature_cols


# ============================================================
# 1. PROPER 2SLS VIA LINEARMODELS
# ============================================================

def run_proper_2sls(df, feature_cols):
    """
    Run proper Two-Stage Least Squares using linearmodels.IV2SLS.
    This produces correct standard errors that account for the
    generated regressor problem (which the old manual OLS approach did not).
    """
    print("\n" + "="*70)
    print("PROPER 2SLS ESTIMATION (linearmodels.IV2SLS)")
    print("="*70)
    
    # Prepare variables
    dep_var = df['pass_rate']
    
    # Exogenous controls (included in both stages)
    exog_cols = [c for c in CONTROL_COLS if c in df.columns]
    exog = sm.add_constant(df[exog_cols])
    
    # Endogenous variable (kappa measured from output - biased)
    endog = df[['kappa_actual']]
    
    # Instruments (instruction features - exogenous)
    instruments = df[feature_cols]
    
    # --- Model A: Naive OLS (biased baseline) ---
    # Cluster-robust SEs when the prompt id is present; HC1 is kept as a
    # fallback so partial datasets (e.g. the verification sample) still run.
    print("\n--- Model A: Naive OLS (uses output kappa - BIASED) ---")
    X_naive = sm.add_constant(df[['kappa_actual'] + exog_cols])
    if CLUSTER_COL in df.columns and df[CLUSTER_COL].notna().any():
        ols_naive = sm.OLS(dep_var, X_naive).fit(
            cov_type='cluster', cov_kwds={'groups': df[CLUSTER_COL]}
        )
    else:
        ols_naive = sm.OLS(dep_var, X_naive).fit(cov_type='HC1')
    
    print(f"  kappa_actual coef: {ols_naive.params['kappa_actual']:.6f}")
    print(f"  kappa_actual SE:   {ols_naive.bse['kappa_actual']:.6f}")
    print(f"  kappa_actual p:    {ols_naive.pvalues['kappa_actual']:.6f}")
    for ctrl in exog_cols:
        print(f"  {ctrl} coef: {ols_naive.params[ctrl]:.6f} (p={ols_naive.pvalues[ctrl]:.4f})")
    print(f"  R-squared: {ols_naive.rsquared:.4f}")
    
    # --- Model B: Proper 2SLS (unbiased) ---
    print("\n--- Model B: 2SLS IV Estimation (uses predicted kappa - UNBIASED) ---")
    try:
        iv_model = IV2SLS(
            dependent=dep_var,
            exog=exog,
            endog=endog,
            instruments=instruments
        )
        iv_results = iv_model.fit(**_fit_kwargs(df))
        
        print(f"  kappa_actual coef (instrumented): {iv_results.params['kappa_actual']:.6f}")
        print(f"  kappa_actual SE:   {iv_results.std_errors['kappa_actual']:.6f}")
        print(f"  kappa_actual p:    {iv_results.pvalues['kappa_actual']:.6f}")
        for ctrl in exog_cols:
            if ctrl in iv_results.params.index:
                print(f"  {ctrl} coef: {iv_results.params[ctrl]:.6f} (p={iv_results.pvalues[ctrl]:.4f})")
        
        # First-stage diagnostics
        print("\n--- First-Stage Diagnostics ---")
        fs = iv_results.first_stage
        print(f"  First-stage F-statistic: {fs.diagnostics['f.stat'].iloc[0]:.2f}")
        print(f"  First-stage F p-value:   {fs.diagnostics['f.pval'].iloc[0]:.6f}")
        fstat = fs.diagnostics['f.stat'].iloc[0]
        if fstat > 10:
            print(f"  PASS: F > 10 (strong instruments)")
        else:
            print(f"  WARNING: F < 10 (weak instruments - 2SLS may be unreliable)")
        
        # Partial R-squared
        print(f"  Partial R-squared: {fs.diagnostics['partial.rsquared'].iloc[0]:.4f}")
        
        # --- Sargan-Hansen J-test for instrument validity ---
        print("\n--- Sargan-Hansen J-Test (Over-Identifying Restrictions) ---")
        print("  NOTE: This tests whether instruments are validly excluded")
        print("  from the structural equation.  Distinct from the Hansen")
        print("  threshold test (which tests for structural breaks).")
        try:
            j_stat = iv_results.sargan
            print(f"  J-statistic:  {j_stat.stat:.4f}")
            print(f"  J p-value:    {j_stat.pval:.6f}")
            print(f"  J df:         {j_stat.df}")
            if j_stat.pval > 0.05:
                print(f"  PASS: Cannot reject instrument validity (p > 0.05)")
            else:
                print(f"  WARNING: Instruments may be invalid (p < 0.05).")
                print(f"  Consider dropping suspect instruments (inst_tokens, inst_avg_word_len).")
            # Bind properties so we can extract them later
            iv_results.j_stat_val = j_stat.stat
            iv_results.j_pval = j_stat.pval
        except Exception as e:
            print(f"  J-test failed: {e}")
            iv_results.j_stat_val = np.nan
            iv_results.j_pval = np.nan
        
    except Exception as e:
        print(f"  2SLS failed: {e}")
        print("  Falling back to manual reduced-form approach...")
        iv_results = None
    
    return ols_naive, iv_results


# ============================================================
# 2. HAUSMAN ENDOGENEITY TEST
# ============================================================

def hausman_test(df, feature_cols, iv_results=None):
    """
    Hausman test: formally tests whether OLS and 2SLS produce
    significantly different estimates. If p < 0.05, endogeneity is confirmed
    and 2SLS should be preferred over OLS.
    """
    print("\n" + "="*70)
    print("HAUSMAN ENDOGENEITY TEST")
    print("="*70)
    
    if iv_results is not None and hasattr(iv_results, 'wu_hausman'):
        print("  Using built-in Wu-Hausman test from linearmodels IV2SLS...")
        try:
            H = iv_results.wu_hausman.stat
            p_hausman = iv_results.wu_hausman.pval
            print(f"  Hausman statistic: {H:.4f}")
            print(f"  Hausman p-value:   {p_hausman:.6f}")
            if p_hausman < 0.05:
                print(f"  RESULT: Endogeneity CONFIRMED (p < 0.05). 2SLS preferred.")
            else:
                print(f"  RESULT: No significant endogeneity detected. OLS may be sufficient.")
            return H, p_hausman
        except Exception as e:
            print(f"  Built-in Wu-Hausman failed: {e}")
            pass
            
    exog_cols = [c for c in CONTROL_COLS if c in df.columns]
    
    # OLS estimate
    X_ols = sm.add_constant(df[['kappa_actual'] + exog_cols])
    ols = sm.OLS(df['pass_rate'], X_ols).fit()
    beta_ols = ols.params['kappa_actual']
    var_ols = ols.cov_params().loc['kappa_actual', 'kappa_actual']
    
    # 2SLS estimate (manual reduced form for Hausman comparison)
    # Stage 1: Regress kappa_actual on instruments + controls
    Z = sm.add_constant(df[feature_cols + exog_cols])
    stage1 = sm.OLS(df['kappa_actual'], Z).fit()
    kappa_hat = stage1.fittedvalues
    
    # Stage 2: Regress pass_rate on kappa_hat + controls
    X_2sls = sm.add_constant(pd.DataFrame({
        'kappa_hat': kappa_hat,
        **{c: df[c] for c in exog_cols}
    }))
    stage2 = sm.OLS(df['pass_rate'], X_2sls).fit()
    beta_2sls = stage2.params['kappa_hat']
    var_2sls = stage2.cov_params().loc['kappa_hat', 'kappa_hat']
    
    # Hausman statistic: H = (beta_2sls - beta_ols)^2 / (var_2sls - var_ols)
    var_diff = var_2sls - var_ols
    if var_diff > 0:
        H = (beta_2sls - beta_ols)**2 / var_diff
        p_hausman = 1 - stats.chi2.cdf(H, df=1)
        
        print(f"  OLS coefficient (kappa):  {beta_ols:.6f}")
        print(f"  2SLS coefficient (kappa): {beta_2sls:.6f}")
        print(f"  Difference:               {beta_2sls - beta_ols:.6f}")
        print(f"  Hausman statistic (chi2): {H:.4f}")
        print(f"  Hausman p-value:          {p_hausman:.6f}")
        
        if p_hausman < 0.05:
            print(f"  RESULT: Endogeneity CONFIRMED (p < 0.05). 2SLS preferred.")
        else:
            print(f"  RESULT: No significant endogeneity detected. OLS may be sufficient.")
    else:
        print(f"  WARNING: Negative variance difference ({var_diff:.6f}).")
        print(f"  This can occur with weak instruments. Hausman test inconclusive.")
        H = np.nan
        p_hausman = np.nan
    
    return H, p_hausman


# ============================================================
# 3. BOOTSTRAP HANSEN THRESHOLD TEST
# ============================================================

def compute_threshold_wald(df, gamma, dep_col='pass_rate', kappa_col='kappa_predicted'):
    """
    Compute Wald statistic for a specific threshold gamma.
    Tests H0: beta_low = beta_high (no structural break at gamma).
    """
    exog_cols = [c for c in CONTROL_COLS if c in df.columns]
    
    low = df[df[kappa_col] <= gamma]
    high = df[df[kappa_col] > gamma]
    
    if len(low) < MIN_REGIME_SIZE or len(high) < MIN_REGIME_SIZE:
        return np.nan
    
    # Fit separate regressions
    X_low = sm.add_constant(low[[kappa_col] + exog_cols])
    X_high = sm.add_constant(high[[kappa_col] + exog_cols])
    
    try:
        res_low = sm.OLS(low[dep_col], X_low).fit()
        res_high = sm.OLS(high[dep_col], X_high).fit()
    except Exception:
        return np.nan
    
    # Pooled regression (no break)
    X_pooled = sm.add_constant(df[[kappa_col] + exog_cols])
    res_pooled = sm.OLS(df[dep_col], X_pooled).fit()
    
    # Wald-type statistic: reduction in RSS
    rss_pooled = res_pooled.ssr
    rss_split = res_low.ssr + res_high.ssr
    
    if rss_split == 0:
        return np.nan
    
    # F-type statistic
    k = len(res_low.params)
    n = len(df)
    f_stat = ((rss_pooled - rss_split) / k) / (rss_split / (n - 2 * k))
    
    return f_stat


def hansen_threshold_test(df, dep_col='pass_rate', kappa_col='kappa_predicted'):
    """
    Proper bootstrap Hansen threshold test.
    
    1. Grid search over candidate thresholds to find sup-Wald statistic
    2. Bootstrap under the null (no threshold) to get critical values
    3. Compare observed sup-Wald to bootstrap distribution for p-value
    """
    print("\n" + "="*70)
    print("HANSEN BOOTSTRAP THRESHOLD TEST")
    print("="*70)
    
    thresholds = build_threshold_grid(df, kappa_col)

    # Step 1: Compute observed Wald statistics across the grid
    if len(thresholds) == 0:
        print("  ERROR: Grid is empty ,  sample too small for MIN_REGIME_SIZE.")
        return None, None, None, None, None, []
    print(f"  Searching {len(thresholds)} candidate thresholds "
          f"[{thresholds.min():.2f}, {thresholds.max():.2f}] "
          f"(mode={THRESHOLD_GRID_MODE})...")
    wald_stats = []
    for gamma in thresholds:
        w = compute_threshold_wald(df, gamma, dep_col, kappa_col)
        wald_stats.append((gamma, w))

    valid_stats = [(g, w) for g, w in wald_stats if not np.isnan(w)]
    if not valid_stats:
        print("  ERROR: No valid threshold found. All splits too small.")
        return None, None, None, None, None, wald_stats
    
    # Observed sup-Wald
    best_gamma, sup_wald = max(valid_stats, key=lambda x: x[1])
    print(f"\n  Observed threshold (gamma*): {best_gamma:.1f}")
    print(f"  Observed sup-Wald statistic: {sup_wald:.4f}")
    
    # Report regime statistics
    low_regime = df[df[kappa_col] <= best_gamma]
    high_regime = df[df[kappa_col] > best_gamma]
    print(f"\n  Low regime  (kappa <= {best_gamma}):  N={len(low_regime)}, mean pass_rate={low_regime[dep_col].mean():.4f}")
    print(f"  High regime (kappa >  {best_gamma}):  N={len(high_regime)}, mean pass_rate={high_regime[dep_col].mean():.4f}")
    print(f"  Difference in pass rates: {low_regime[dep_col].mean() - high_regime[dep_col].mean():.4f}")
    
    # Step 2: Bootstrap under the null (no threshold)
    print(f"\n  Bootstrapping null distribution ({HANSEN_BOOTSTRAP_ITERATIONS} iterations)...")
    exog_cols = [c for c in CONTROL_COLS if c in df.columns]
    
    # Fit pooled model under H0
    X_pooled = sm.add_constant(df[[kappa_col] + exog_cols])
    pooled_model = sm.OLS(df[dep_col], X_pooled).fit()
    fitted_values = pooled_model.fittedvalues
    residuals = pooled_model.resid
    
    boot_sup_walds = []
    rng = np.random.RandomState(42)
    
    for b in range(HANSEN_BOOTSTRAP_ITERATIONS):
        # Wild bootstrap: multiply residuals by Rademacher weights
        weights = rng.choice([-1, 1], size=len(df))
        y_boot = fitted_values + residuals * weights
        
        df_boot = df.copy()
        df_boot[dep_col] = y_boot
        
        # Find sup-Wald for this bootstrap sample
        boot_walds = []
        for gamma in thresholds:
            w = compute_threshold_wald(df_boot, gamma, dep_col, kappa_col)
            if not np.isnan(w):
                boot_walds.append(w)
        
        if boot_walds:
            boot_sup_walds.append(max(boot_walds))
        
        if (b + 1) % 100 == 0:
            print(f"    Bootstrap iteration {b + 1}/{HANSEN_BOOTSTRAP_ITERATIONS}")
            
    # Step 3: Threshold Confidence Interval (cluster bootstrap under the alternative).
    # We resample whole clusters (prompt ids), not individual rows. Row-level
    # resampling would treat the ~18 model observations per prompt as
    # independent draws and produce an artificially tight CI.
    print(f"\n  Building {THRESHOLD_CI_BOOTSTRAP} cluster-bootstrap threshold CIs...")
    boot_gammas = []
    if CLUSTER_COL in df.columns:
        clusters = df.groupby(CLUSTER_COL, sort=False).indices
        cluster_ids = np.array(list(clusters.keys()))
        cluster_rows = [clusters[cid] for cid in cluster_ids]
    else:
        cluster_ids = np.arange(len(df))
        cluster_rows = [np.array([i]) for i in cluster_ids]

    for b in range(THRESHOLD_CI_BOOTSTRAP):
        rng_b = np.random.RandomState(10_000 + b)
        sampled = rng_b.choice(len(cluster_ids), size=len(cluster_ids), replace=True)
        row_idx = np.concatenate([cluster_rows[i] for i in sampled])
        df_boot = df.iloc[row_idx]

        boot_walds = []
        for gamma in thresholds:
            w = compute_threshold_wald(df_boot, gamma, dep_col, kappa_col)
            if not np.isnan(w):
                boot_walds.append((gamma, w))

        if boot_walds:
            boot_best_gamma, _ = max(boot_walds, key=lambda x: x[1])
            boot_gammas.append(boot_best_gamma)
            
    ci_lower = None
    ci_upper = None
    if boot_gammas:
        ci_lower = np.percentile(boot_gammas, 2.5)
        ci_upper = np.percentile(boot_gammas, 97.5)
        print(f"  Threshold 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(f"  Bootstrap threshold std dev: {np.std(boot_gammas):.2f}")
    
    # Step 4: p-value = proportion of bootstrap sup-Walds exceeding observed
    if boot_sup_walds:
        p_hansen = np.mean(np.array(boot_sup_walds) >= sup_wald)
        
        print(f"\n  Bootstrap p-value: {p_hansen:.4f}")
        if p_hansen < 0.01:
            print(f"  RESULT: Threshold is HIGHLY SIGNIFICANT (p < 0.01)")
        elif p_hansen < 0.05:
            print(f"  RESULT: Threshold is SIGNIFICANT (p < 0.05)")
        elif p_hansen < 0.10:
            print(f"  RESULT: Threshold is MARGINALLY SIGNIFICANT (p < 0.10)")
        else:
            print(f"  RESULT: No significant threshold detected (p >= 0.10)")
    else:
        p_hansen = np.nan
        print(f"  WARNING: Bootstrap produced no valid statistics.")
    
    return best_gamma, sup_wald, p_hansen, ci_lower, ci_upper, wald_stats


# ============================================================
# 4. PLACEBO TEST
# ============================================================

def placebo_test(df, dep_col='pass_rate', kappa_col='kappa_predicted'):
    """
    Placebo test: shuffle kappa_predicted and re-run threshold detection.
    If the kink is real, the true threshold should be far from the placebo
    distribution of thresholds.
    """
    print("\n" + "="*70)
    print("PLACEBO TEST (Shuffled Instruments)")
    print("="*70)
    
    # Placebo uses the same grid as the observed test so the null distribution
    # it generates is directly comparable. Seed 123 is kept distinct from the
    # Hansen bootstrap seed (42) so the two procedures draw independent
    # random sequences and cannot accidentally share state.
    thresholds = build_threshold_grid(df, kappa_col)
    if len(thresholds) == 0:
        print("  ERROR: Grid is empty ,  sample too small for MIN_REGIME_SIZE.")
        return [], []
    rng = np.random.RandomState(123)
    
    placebo_gammas = []
    placebo_sup_walds = []
    
    print(f"  Running {PLACEBO_ITERATIONS} placebo iterations...")
    
    for p in range(PLACEBO_ITERATIONS):
        df_placebo = df.copy()
        df_placebo[kappa_col] = rng.permutation(df[kappa_col].values)
        
        walds = []
        for gamma in thresholds:
            w = compute_threshold_wald(df_placebo, gamma, dep_col, kappa_col)
            if not np.isnan(w):
                walds.append((gamma, w))
        
        if walds:
            best_g, best_w = max(walds, key=lambda x: x[1])
            placebo_gammas.append(best_g)
            placebo_sup_walds.append(best_w)
        
        if (p + 1) % 100 == 0:
            print(f"    Placebo iteration {p + 1}/{PLACEBO_ITERATIONS}")
    
    if placebo_gammas:
        print(f"\n  Placebo threshold distribution:")
        print(f"    Mean:   {np.mean(placebo_gammas):.2f}")
        print(f"    Median: {np.median(placebo_gammas):.2f}")
        print(f"    Std:    {np.std(placebo_gammas):.2f}")
        print(f"    [5%, 95%]: [{np.percentile(placebo_gammas, 5):.2f}, {np.percentile(placebo_gammas, 95):.2f}]")
        
        print(f"\n  Placebo sup-Wald distribution:")
        print(f"    Mean:   {np.mean(placebo_sup_walds):.2f}")
        print(f"    [5%, 95%]: [{np.percentile(placebo_sup_walds, 5):.2f}, {np.percentile(placebo_sup_walds, 95):.2f}]")
    
    return placebo_gammas, placebo_sup_walds


# ============================================================
# 5. REGIME-SPLIT 2SLS (The "Complexity Kink")
# ============================================================

def run_regime_split_2sls(df, feature_cols, gamma, dep_var='pass_rate', kappa_col='kappa_actual'):
    """
    Runs 2SLS separately on the low-complexity and high-complexity regimes
    defined by the threshold gamma. This cleanly handles the endogeneity
    in BOTH regimes, avoiding the bias of just comparing OLS splits.
    """
    print("\n" + "="*70)
    print("REGIME-SPLIT 2SLS (Estimating the Kink)")
    print("="*70)
    
    if gamma is None or np.isnan(gamma):
        print("  Skipping regime-split 2SLS (no valid threshold provided).")
        return None, None
        
    exog_cols = [c for c in CONTROL_COLS if c in df.columns]
    
    # Split the sample based on the PREDICTED kappa (the exogenous instrument index)
    # We split on predicted kappa to avoid endogeneity in the split selection itself.
    low_df = df[df['kappa_predicted'] <= gamma].copy()
    high_df = df[df['kappa_predicted'] > gamma].copy()
    
    if len(low_df) < MIN_REGIME_SIZE or len(high_df) < MIN_REGIME_SIZE:
        print("  WARNING: Regimes too small for reliable 2SLS. Skipping.")
        return None, None
        
    print(f"  Split at gamma* = {gamma:.2f}")
    print(f"  Low Regime N  = {len(low_df)}")
    print(f"  High Regime N = {len(high_df)}")
    
    def run_split(split_df, regime_name):
        exog = sm.add_constant(split_df[exog_cols])
        endog = split_df[[kappa_col]]
        instruments = split_df[feature_cols]
        dep = split_df[dep_var]
        
        try:
            iv_model = IV2SLS(dependent=dep, exog=exog, endog=endog, instruments=instruments)
            res = iv_model.fit(**_fit_kwargs(split_df))

            coef = res.params[kappa_col]
            se = res.std_errors[kappa_col]
            pval = res.pvalues[kappa_col]
            
            print(f"\n  [{regime_name} Regime] kappa coef: {coef:.6f}")
            print(f"  [{regime_name} Regime] SE:         {se:.6f}")
            print(f"  [{regime_name} Regime] p-value:    {pval:.6f}")
            
            return res
        except Exception as e:
            print(f"  [{regime_name} Regime] 2SLS failed: {e}")
            return None
            
    res_low = run_split(low_df, "LOW")
    res_high = run_split(high_df, "HIGH")
    
    if res_low is not None and res_high is not None:
        # Quick Wald test for difference in coefficients
        diff = res_low.params[kappa_col] - res_high.params[kappa_col]
        se_diff = np.sqrt(res_low.std_errors[kappa_col]**2 + res_high.std_errors[kappa_col]**2)
        z_stat = diff / se_diff
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        print(f"\n  Difference (Low - High): {diff:.6f}")
        print(f"  Z-statistic:             {z_stat:.4f}")
        print(f"  p-value for difference:  {p_val:.6f}")
        if p_val < 0.05:
            print("  RESULT: The return to human effort (kappa) is significantly different across regimes.")
            if res_low.params[kappa_col] > res_high.params[kappa_col]:
                print("          Human effort matters MORE in the LOW complexity regime.")
            else:
                print("          Human effort matters MORE in the HIGH complexity regime.")
        else:
            print("  RESULT: Cannot reject that returns to human effort are equal across regimes.")
            
    return res_low, res_high


# ============================================================
# 6. INSTRUMENT ROBUSTNESS CHECK
# ============================================================

def run_instrument_robustness(df):
    """
    Re-runs the main 2SLS using more conservative instrument subsets
    to ensure results are not driven solely by potentially violative
    instruments (like prompt length/lexical complexity).
    """
    print("\n" + "="*70)
    print("INSTRUMENT SUBSET ROBUSTNESS CHECK")
    print("="*70)
    
    dep_var = df['pass_rate']
    endog = df[['kappa_actual']]
    exog_cols = [c for c in CONTROL_COLS if c in df.columns]
    exog = sm.add_constant(df[exog_cols])
    
    subsets = {
        "Full Set": IV_FEATURE_COLS,
        "Reduced Set (No Length/Lexical)": REDUCED_IV_COLS,
        "Conservative (Pure Structural)": CONSERVATIVE_IV_COLS
    }
    
    results = {}
    
    for name, subset in subsets.items():
        print(f"\n--- {name} ---")
        print(f"  Instruments ({len(subset)}): {', '.join(subset)}")
        instruments = df[subset]
        
        try:
            iv_model = IV2SLS(dependent=dep_var, exog=exog, endog=endog, instruments=instruments)
            res = iv_model.fit(**_fit_kwargs(df))

            coef = res.params['kappa_actual']
            se = res.std_errors['kappa_actual']
            pval = res.pvalues['kappa_actual']
            fstat = res.first_stage.diagnostics['f.stat'].iloc[0]
            
            j_stat_val, j_pval = np.nan, np.nan
            if hasattr(res, 'sargan'):
                j_stat_val = res.sargan.stat
                j_pval = res.sargan.pval
                
            print(f"  kappa coef: {coef:.6f} (p={pval:.4f})")
            print(f"  1st stage F: {fstat:.2f}")
            if not np.isnan(j_pval):
                print(f"  J-test p-value: {j_pval:.4f}")
                
            results[name] = {
                'coef': coef, 'se': se, 'pval': pval,
                'fstat': fstat, 'j_pval': j_pval
            }
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = None
            
    return results


# ============================================================
# MAIN
# ============================================================

def run_stage2_analysis(enriched_file, model_path):
    """Run the complete Stage 2 analysis battery."""
    df, feature_cols = load_data_for_stage2(enriched_file, model_path)
    
    results = {}
    
    # 1. Proper 2SLS
    ols_naive, iv_results = run_proper_2sls(df, feature_cols)
    results['ols_naive'] = ols_naive
    results['iv_results'] = iv_results
    
    # 2. Hausman test
    h_stat, h_pval = hausman_test(df, feature_cols, iv_results)
    results['hausman_stat'] = h_stat
    results['hausman_pval'] = h_pval
    
    # 3. Instrument Robustness Check
    robustness_results = run_instrument_robustness(df)
    results['instrument_robustness'] = robustness_results
    
    # 4. Hansen threshold test
    gamma, sup_wald, p_hansen, ci_lower, ci_upper, wald_curve = hansen_threshold_test(df)
    results['threshold_gamma'] = gamma
    results['threshold_sup_wald'] = sup_wald
    results['threshold_pval'] = p_hansen
    results['threshold_ci_lower'] = ci_lower
    results['threshold_ci_upper'] = ci_upper
    results['wald_curve'] = wald_curve
    
    # 5. Placebo test
    placebo_gammas, placebo_walds = placebo_test(df)
    results['placebo_gammas'] = placebo_gammas
    results['placebo_walds'] = placebo_walds
    
    # Compare real threshold vs placebo
    if gamma is not None and placebo_gammas:
        print("\n" + "="*70)
        print("REAL vs PLACEBO COMPARISON")
        print("="*70)
        placebo_mean = np.mean(placebo_gammas)
        placebo_std = np.std(placebo_gammas)
        if placebo_std > 0:
            z_score = (gamma - placebo_mean) / placebo_std
            print(f"  Real threshold:    {gamma:.2f}")
            print(f"  Placebo mean:      {placebo_mean:.2f}")
            print(f"  Z-score:           {z_score:.2f}")
        
        if placebo_walds and sup_wald is not None:
            p_placebo = np.mean(np.array(placebo_walds) >= sup_wald)
            print(f"  Placebo p-value (sup-Wald): {p_placebo:.4f}")
            if p_placebo < 0.05:
                print(f"  RESULT: Real kink sup-Wald exceeds placebo distribution. Kink is REAL.")
            else:
                print(f"  RESULT: Real kink sup-Wald within placebo range. Kink may be SPURIOUS.")
    
    # 5. Regime-Split 2SLS Analysis
    res_low, res_high = run_regime_split_2sls(df, feature_cols, gamma)
    results['regime_low_iv'] = res_low
    results['regime_high_iv'] = res_high

    # 6. Survivorship robustness: re-run the threshold test using the
    # all-samples Stage 1 predictor. Stage 1's primary predictor is fit on
    # pass_rate == 1.0 only, which means it extrapolates onto the failure
    # regime. If that extrapolation is manufacturing the kink, the all-samples
    # predictor ,  which is trained on failures too ,  will produce a different
    # threshold location or a much weaker sup-Wald. Agreement across the two
    # predictors is the strongest single sign that the kink is structural
    # rather than a filter artefact.
    if (
        'kappa_predicted_all_samples' in df.columns
        and df['kappa_predicted_all_samples'].notna().any()
    ):
        print("\n" + "=" * 70)
        print("ROBUSTNESS: THRESHOLD TEST WITH ALL-SAMPLES STAGE 1 PREDICTOR")
        print("=" * 70)
        robust_out = hansen_threshold_test(
            df, dep_col='pass_rate', kappa_col='kappa_predicted_all_samples'
        )
        (robust_gamma, robust_sup_wald, robust_pval,
         robust_ci_lo, robust_ci_hi, _) = robust_out
        results['robust_gamma'] = robust_gamma
        results['robust_sup_wald'] = robust_sup_wald
        results['robust_pval'] = robust_pval
        results['robust_ci_lower'] = robust_ci_lo
        results['robust_ci_upper'] = robust_ci_hi
    else:
        print("\n  [Skipped all-samples robustness: predictor column not present. "
              "Re-run train_stage1_iv.py to generate it.]")
    
    # Save results summary
    summary_path = os.path.join(OUTPUT_DIR, "stage2_results_summary.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(summary_path, 'w') as f:
        f.write("Stage 2 Results Summary\n")
        f.write("=" * 50 + "\n\n")
        if iv_results is not None:
            f.write(f"2SLS kappa coefficient: {iv_results.params['kappa_actual']:.6f}\n")
            f.write(f"2SLS kappa p-value: {iv_results.pvalues['kappa_actual']:.6f}\n")
            f.write(f"First-stage F-stat: {iv_results.first_stage.diagnostics['f.stat'].iloc[0]:.2f}\n")
            if hasattr(iv_results, 'j_stat_val'):
                f.write(f"Sargan-Hansen J-stat: {iv_results.j_stat_val:.4f}\n")
                f.write(f"Sargan-Hansen J-pval: {iv_results.j_pval:.6f}\n")
        f.write(f"\nHausman statistic: {h_stat}\n")
        f.write(f"Hausman p-value: {h_pval}\n")
        f.write(f"\nHansen threshold (gamma): {gamma}\n")
        f.write(f"Hansen bootstrap p-value: {p_hansen}\n")
        if ci_lower is not None:
            f.write(f"Hansen threshold 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]\n")
            
        f.write("\nINSTRUMENT ROBUSTNESS:\n")
        for name, r in robustness_results.items():
            if r is not None:
                f.write(f"{name}: coef={r['coef']:.6f} (p={r['pval']:.4f}), F={r['fstat']:.2f}, J-p={r['j_pval']:.4f}\n")
        
        if res_low is not None and res_high is not None:
            f.write("\nREGIME-SPLIT 2SLS:\n")
            f.write(f"Low Regime kappa coef:  {res_low.params['kappa_actual']:.6f} (p={res_low.pvalues['kappa_actual']:.4f})\n")
            f.write(f"High Regime kappa coef: {res_high.params['kappa_actual']:.6f} (p={res_high.pvalues['kappa_actual']:.4f})\n")
            
    print(f"\nResults summary saved to {summary_path}")
    
    return df, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: 2SLS + Hansen + Hausman + Placebo")
    parser.add_argument("--input", default=DEFAULT_OOF_FILE,
                        help="Path to OOF-enriched JSONL file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to Stage 1 model")
    args = parser.parse_args()
    
    run_stage2_analysis(args.input, args.model)
