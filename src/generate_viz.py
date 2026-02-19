"""
Visualization suite for the Complexity Kink Research.
Generates publication-quality figures with dynamically computed thresholds.
"""
import json
import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEFAULT_ENRICHED_FILE, DEFAULT_MODEL_PATH, OUTPUT_DIR,
    IV_FEATURE_COLS, CONTROL_COLS,
    THRESHOLD_GRID_START, THRESHOLD_GRID_END, THRESHOLD_GRID_STEP,
    MIN_REGIME_SIZE
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


def load_and_predict(enriched_file, model_path):
    """Load data and generate predicted kappa."""
    print("Loading data for visualization...")
    data = []
    with open(enriched_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            row = dict(item.get('iv_features', {}))
            row['kappa_actual'] = item.get('kappa_cyclomatic', 1)
            row['is_success'] = item.get('is_success', 0)
            row['pass_rate'] = _compute_pass_rate(item)
            row['e_norm'] = item.get('e_norm', 0.0)
            data.append(row)
    
    df = pd.DataFrame(data)
    model = joblib.load(model_path)
    
    feature_cols = [c for c in IV_FEATURE_COLS if c in df.columns]
    df['kappa_predicted'] = model.predict(df[feature_cols].values)
    
    return df


def find_threshold(df, dep_col='pass_rate', kappa_col='kappa_predicted'):
    """Find the best threshold dynamically using segmented regression."""
    import statsmodels.api as sm
    
    thresholds = np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END, THRESHOLD_GRID_STEP)
    best_gamma = None
    best_fstat = -np.inf
    
    exog_cols = [c for c in CONTROL_COLS if c in df.columns and c in df.columns]
    
    X_pooled = sm.add_constant(df[[kappa_col] + exog_cols].dropna())
    if len(X_pooled) < MIN_REGIME_SIZE * 2:
        return 6.0  # fallback
    
    res_pooled = sm.OLS(df[dep_col], X_pooled).fit()
    
    for gamma in thresholds:
        low = df[df[kappa_col] <= gamma]
        high = df[df[kappa_col] > gamma]
        
        if len(low) < MIN_REGIME_SIZE or len(high) < MIN_REGIME_SIZE:
            continue
        
        try:
            X_low = sm.add_constant(low[[kappa_col] + exog_cols])
            X_high = sm.add_constant(high[[kappa_col] + exog_cols])
            res_low = sm.OLS(low[dep_col], X_low).fit()
            res_high = sm.OLS(high[dep_col], X_high).fit()
            
            rss_split = res_low.ssr + res_high.ssr
            rss_pooled = res_pooled.ssr
            k = len(res_low.params)
            n = len(df)
            f_stat = ((rss_pooled - rss_split) / k) / (rss_split / (n - 2 * k))
            
            if f_stat > best_fstat:
                best_fstat = f_stat
                best_gamma = gamma
        except Exception:
            continue
    
    return best_gamma if best_gamma is not None else 6.0


def generate_visualizations(enriched_file, model_path, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_and_predict(enriched_file, model_path)
    
    # Dynamically compute the threshold
    threshold = find_threshold(df)
    print(f"Computed threshold for visualization: {threshold:.1f}")
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # ================================================================
    # 1. The Paradox: Naive vs Predicted (Side-by-Side)
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Naive View
    naive_stats = df[df['kappa_actual'] <= 15].groupby('kappa_actual')['pass_rate'].mean()
    sns.barplot(x=naive_stats.index, y=naive_stats.values, ax=ax1, palette="Reds_d")
    ax1.set_title("The 'Reverse Threshold' (Naive Measurement)\n"
                  "Artificially low pass rate at kappa=1", fontsize=14)
    ax1.set_ylabel("Mean Pass Rate")
    ax1.set_xlabel("Output Cyclomatic Complexity (kappa)")

    # Corrected View
    df['kappa_rounded'] = df['kappa_predicted'].round().astype(int)
    corrected_stats = df[df['kappa_rounded'] <= 15].groupby('kappa_rounded')['pass_rate'].mean()
    sns.barplot(x=corrected_stats.index, y=corrected_stats.values, ax=ax2, palette="Blues_d")
    
    # Place kink line at the COMPUTED threshold (not hardcoded)
    kink_idx = threshold - corrected_stats.index.min()
    ax2.axvline(x=kink_idx, color='red', linestyle='--', linewidth=2,
                label=f'Complexity Kink (gamma={threshold:.1f})')
    ax2.set_title("The 'Complexity Kink' (Instrumented Variable)\n"
                  "True reliability limit recovered", fontsize=14)
    ax2.set_ylabel("Mean Pass Rate")
    ax2.set_xlabel("Predicted Target Complexity (kappa_hat)")
    ax2.legend()

    plt.tight_layout()
    path1 = os.path.join(output_dir, "paradox_vs_kink.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path1}")

    # ================================================================
    # 2. Phase Diagram Heatmap
    # ================================================================
    plt.figure(figsize=(10, 8))
    df['E_bin'] = pd.cut(df['e_norm'], bins=15)
    df['K_bin'] = pd.cut(df['kappa_predicted'], bins=15)
    
    heatmap_data = df.pivot_table(index='K_bin', columns='E_bin', 
                                   values='pass_rate', aggfunc='mean')
    heatmap_data = heatmap_data.iloc[::-1]
    
    sns.heatmap(heatmap_data, cmap="RdYlGn", center=0.5, annot=False)
    plt.title("LLM Performance Phase Diagram\n"
              "Green = Safe Zone | Red = Logic Collapse", fontsize=16)
    plt.ylabel("Target Structural Complexity (kappa_hat)")
    plt.xlabel("Normalized Instruction Entropy (E_norm)")
    
    path2 = os.path.join(output_dir, "performance_phase_diagram.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path2}")

    # ================================================================
    # 3. Wald Statistic Curve (Hansen Threshold Selection)
    # ================================================================
    import statsmodels.api as sm
    
    thresholds = np.arange(THRESHOLD_GRID_START, THRESHOLD_GRID_END, THRESHOLD_GRID_STEP)
    exog_cols = [c for c in CONTROL_COLS if c in df.columns]
    
    X_pooled = sm.add_constant(df[['kappa_predicted'] + exog_cols])
    res_pooled = sm.OLS(df['pass_rate'], X_pooled).fit()
    
    wald_values = []
    for gamma in thresholds:
        low = df[df['kappa_predicted'] <= gamma]
        high = df[df['kappa_predicted'] > gamma]
        
        if len(low) < MIN_REGIME_SIZE or len(high) < MIN_REGIME_SIZE:
            wald_values.append(np.nan)
            continue
        
        try:
            X_low = sm.add_constant(low[['kappa_predicted'] + exog_cols])
            X_high = sm.add_constant(high[['kappa_predicted'] + exog_cols])
            res_low = sm.OLS(low['pass_rate'], X_low).fit()
            res_high = sm.OLS(high['pass_rate'], X_high).fit()
            
            rss_split = res_low.ssr + res_high.ssr
            k = len(res_low.params)
            n = len(df)
            f_stat = ((res_pooled.ssr - rss_split) / k) / (rss_split / (n - 2 * k))
            wald_values.append(f_stat)
        except Exception:
            wald_values.append(np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, wald_values, 'b-', linewidth=2)
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Optimal gamma = {threshold:.1f}')
    plt.xlabel("Candidate Threshold (gamma)", fontsize=14)
    plt.ylabel("Wald F-statistic", fontsize=14)
    plt.title("Hansen Threshold Selection\n"
              "Peak = optimal structural break location", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    path3 = os.path.join(output_dir, "hansen_wald_curve.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--input", default=DEFAULT_ENRICHED_FILE,
                        help="Path to enriched JSONL file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH,
                        help="Path to Stage 1 model")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory to save visualizations")
    args = parser.parse_args()
    
    generate_visualizations(args.input, args.model, args.output_dir)
