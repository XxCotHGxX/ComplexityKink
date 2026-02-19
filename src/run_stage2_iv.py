"""
Stage 2: The corrected analysis pipeline.

Implements:
1. Proper 2SLS via linearmodels.iv.IV2SLS (correct standard errors)
2. Hausman endogeneity test (OLS vs 2SLS)
3. Bootstrap Hansen threshold test (sup-Wald with null distribution)
4. Placebo test (shuffled instruments)
5. Continuous pass rate as dependent variable
6. Memorization control (m_mem_jaccard) as regressor
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
    DEFAULT_ENRICHED_FILE, DEFAULT_MODEL_PATH, OUTPUT_DIR,
    IV_FEATURE_COLS, CONTROL_COLS,
    THRESHOLD_GRID_START, THRESHOLD_GRID_END, THRESHOLD_GRID_STEP,
    HANSEN_BOOTSTRAP_ITERATIONS, PLACEBO_ITERATIONS, MIN_REGIME_SIZE
)


def _compute_pass_rate(item):
    """Compute pass rate from the status field when pass_rate is not stored."""
    if 'pass_rate' in item:
        return float(item['pass_rate'])
    status = item.get('status', [])
    if isinstance(status, list) and len(status) > 0:
        n_pass = sum(1 for s in status if isinstance(s, str) and 'pass' in s.lower())
        return n_pass / len(status)
    return 0.0


def load_data_for_stage2(enriched_file, model_path):
    """Load data and generate predicted kappa from Stage 1 model."""
    print("Loading data and model...")
    data = []
    with open(enriched_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            row = dict(item.get('iv_features', {}))
            row['id'] = item.get('id', '')
            row['kappa_actual'] = item.get('kappa_cyclomatic', 1)
            row['is_success'] = item.get('is_success', 0)
            row['pass_rate'] = _compute_pass_rate(item)
            row['e_norm'] = item.get('e_norm', 0.0)
            row['m_mem_jaccard'] = item.get('m_mem_jaccard', 0.0)
            row['coupling_depth'] = item.get('coupling_depth', 0)
            row['lang'] = item.get('lang', 'unknown')
            data.append(row)
    
    df = pd.DataFrame(data)
    model = joblib.load(model_path)
    
    # Available IV feature columns
    feature_cols = [c for c in IV_FEATURE_COLS if c in df.columns]
    
    # Generate predicted kappa (instrumented variable)
    X_features = df[feature_cols].values
    df['kappa_predicted'] = model.predict(X_features)
    
    print(f"Loaded {len(df)} samples")
    print(f"Success rate: {df['is_success'].mean():.3f}")
    print(f"Mean pass_rate: {df['pass_rate'].mean():.3f}")
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
    print("\n--- Model A: Naive OLS (uses output kappa - BIASED) ---")
    X_naive = sm.add_constant(df[['kappa_actual'] + exog_cols])
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
        iv_results = iv_model.fit(cov_type='robust')
        
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
        
    except Exception as e:
        print(f"  2SLS failed: {e}")
        print("  Falling back to manual reduced-form approach...")
        iv_results = None
    
    return ols_naive, iv_results


# ============================================================
# 2. HAUSMAN ENDOGENEITY TEST
# ============================================================

def hausman_test(df, feature_cols):
    """
    Hausman test: formally tests whether OLS and 2SLS produce
    significantly different estimates. If p < 0.05, endogeneity is confirmed
    and 2SLS should be preferred over OLS.
    """
    print("\n" + "="*70)
    print("HAUSMAN ENDOGENEITY TEST")
    print("="*70)
    
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
    
    thresholds = np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END, THRESHOLD_GRID_STEP)
    
    # Step 1: Compute observed Wald statistics across the grid
    print(f"  Searching {len(thresholds)} candidate thresholds [{THRESHOLD_GRID_START}, {THRESHOLD_GRID_END}]...")
    wald_stats = []
    for gamma in thresholds:
        w = compute_threshold_wald(df, gamma, dep_col, kappa_col)
        wald_stats.append((gamma, w))
    
    valid_stats = [(g, w) for g, w in wald_stats if not np.isnan(w)]
    if not valid_stats:
        print("  ERROR: No valid threshold found. All splits too small.")
        return None, None, None, None
    
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
    
    # Step 3: p-value = proportion of bootstrap sup-Walds exceeding observed
    if boot_sup_walds:
        p_hansen = np.mean(np.array(boot_sup_walds) >= sup_wald)
        ci_gammas = []
        
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
    
    return best_gamma, sup_wald, p_hansen, wald_stats


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
    
    thresholds = np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END, THRESHOLD_GRID_STEP)
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
    h_stat, h_pval = hausman_test(df, feature_cols)
    results['hausman_stat'] = h_stat
    results['hausman_pval'] = h_pval
    
    # 3. Hansen threshold test
    gamma, sup_wald, p_hansen, wald_curve = hansen_threshold_test(df)
    results['threshold_gamma'] = gamma
    results['threshold_sup_wald'] = sup_wald
    results['threshold_pval'] = p_hansen
    results['wald_curve'] = wald_curve
    
    # 4. Placebo test
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
        f.write(f"Hausman statistic: {h_stat}\n")
        f.write(f"Hausman p-value: {h_pval}\n")
        f.write(f"Hansen threshold (gamma): {gamma}\n")
        f.write(f"Hansen bootstrap p-value: {p_hansen}\n")
    print(f"\nResults summary saved to {summary_path}")
    
    return df, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: 2SLS + Hansen + Hausman + Placebo")
    parser.add_argument("--input", default=DEFAULT_ENRICHED_FILE,
                        help="Path to enriched JSONL file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to Stage 1 model")
    args = parser.parse_args()
    
    run_stage2_analysis(args.input, args.model)
