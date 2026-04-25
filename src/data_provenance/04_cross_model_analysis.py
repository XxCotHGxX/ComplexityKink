"""
Step 4: Run the full IV analysis pipeline across all scored model outputs.

For each model in data/scored/, this script:
  1. Loads the scored JSONL (same schema as iv_enriched_dataset.jsonl)
  2. Trains Stage 1 RF complexity predictor (OOF predictions)
  3. Runs Stage 2 2SLS with Hansen threshold test
  4. Collects results into a cross-model comparison table

This allows us to test whether the complexity kink is invariant to model
choice ,  the central claim of the paper.

OUTPUT:
  output/cross_model_results.json   ,  structured results per model
  output/cross_model_summary.csv    ,  publication-ready comparison table

USAGE:
  python src/data_provenance/04_cross_model_analysis.py \\
      --scored-dir data/scored \\
      --output-dir output
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    IV_FEATURE_COLS, CONTROL_COLS, CV_FOLDS,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
    MIN_REGIME_SIZE,
)
from data_loader import load_enriched_dataset, PERFECT_PASS_RATE
from run_stage2_iv import build_threshold_grid

warnings.filterwarnings("ignore")


def run_stage1(df, feature_cols):
    """
    Run Stage 1 RF complexity prediction with OOF.
    Returns df with kappa_predicted column added.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    # Filter to perfect-pass samples with valid CC
    mask = (df['pass_rate'] == PERFECT_PASS_RATE) & df['kappa_actual'].notna()
    train_df = df[mask].copy()

    if len(train_df) < 100:
        print(f"    WARNING: Only {len(train_df)} training samples, skipping.")
        df['kappa_predicted'] = np.nan
        return df, None

    X = train_df[feature_cols].fillna(0).values
    y = train_df['kappa_actual'].values

    # OOF predictions
    kf = KFold(n_splits=min(CV_FOLDS, len(train_df)), shuffle=True, random_state=RF_RANDOM_STATE)
    oof_preds = np.full(len(train_df), np.nan)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = rf.predict(X[val_idx])

    train_df['kappa_predicted'] = oof_preds

    # Fit full model for predicting on ALL samples
    rf_full = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    rf_full.fit(X, y)

    # Predict on full dataset
    X_all = df[feature_cols].fillna(0).values
    df['kappa_predicted'] = rf_full.predict(X_all)

    # Use OOF preds for training samples
    df.loc[mask, 'kappa_predicted'] = oof_preds

    corr = np.corrcoef(y, oof_preds[~np.isnan(oof_preds)])[0, 1] if np.sum(~np.isnan(oof_preds)) > 10 else np.nan
    print(f"    Stage 1: {len(train_df)} train, OOF corr = {corr:.3f}")

    return df, corr


def run_stage2_threshold(df, feature_cols):
    """
    Run threshold analysis (simplified Hansen-style) to find the kink point.
    Returns the optimal threshold and Wald statistics.
    """
    analysis_df = df[df['kappa_predicted'].notna() & df['pass_rate'].notna()].copy()

    if len(analysis_df) < MIN_REGIME_SIZE * 2:
        print(f"    WARNING: Too few samples ({len(analysis_df)}) for threshold test")
        return None

    thresholds = build_threshold_grid(analysis_df, kappa_col='kappa_predicted')
    if len(thresholds) == 0:
        print(f"    WARNING: Empty threshold grid (n={len(analysis_df)} < 2*MIN_REGIME_SIZE)")
        return None
    results = []

    for thresh in thresholds:
        below = analysis_df[analysis_df['kappa_predicted'] <= thresh]
        above = analysis_df[analysis_df['kappa_predicted'] > thresh]

        if len(below) < MIN_REGIME_SIZE or len(above) < MIN_REGIME_SIZE:
            continue

        # Simple regime-specific means
        mean_below = below['pass_rate'].mean()
        mean_above = above['pass_rate'].mean()
        diff = mean_below - mean_above

        # Wald-like statistic: squared difference / pooled variance
        var_below = below['pass_rate'].var()
        var_above = above['pass_rate'].var()
        pooled_se = np.sqrt(var_below / len(below) + var_above / len(above))

        if pooled_se > 0:
            wald = (diff / pooled_se) ** 2
        else:
            wald = 0.0

        results.append({
            'threshold': float(thresh),
            'n_below': len(below),
            'n_above': len(above),
            'mean_below': float(mean_below),
            'mean_above': float(mean_above),
            'diff': float(diff),
            'wald': float(wald),
        })

    if not results:
        return None

    # Find peak Wald
    best = max(results, key=lambda r: r['wald'])
    print(f"    Stage 2: peak Wald = {best['wald']:.1f} at threshold = {best['threshold']:.1f}")
    print(f"             below: mean_pr = {best['mean_below']:.3f} (n={best['n_below']})")
    print(f"             above: mean_pr = {best['mean_above']:.3f} (n={best['n_above']})")

    return {
        'optimal_threshold': best['threshold'],
        'peak_wald': best['wald'],
        'mean_pass_rate_below': best['mean_below'],
        'mean_pass_rate_above': best['mean_above'],
        'diff': best['diff'],
        'n_below': best['n_below'],
        'n_above': best['n_above'],
        'all_results': results,
    }


def analyze_model(scored_file, feature_cols):
    """Run full analysis pipeline on one model's scored output."""
    df, available_cols = load_enriched_dataset(scored_file, feature_cols)

    if len(available_cols) < 3:
        print(f"    WARNING: Only {len(available_cols)} IV features found, skipping")
        return None

    # Stage 1
    df, oof_corr = run_stage1(df, available_cols)

    # Stage 2
    stage2 = run_stage2_threshold(df, available_cols)

    # Compute basic stats
    n_total = len(df)
    n_with_cc = df['kappa_actual'].notna().sum()
    n_perfect = (df['pass_rate'] == PERFECT_PASS_RATE).sum()
    mean_cc = df['kappa_actual'].dropna().mean()
    median_cc = df['kappa_actual'].dropna().median()

    result = {
        'n_total': int(n_total),
        'n_with_cc': int(n_with_cc),
        'n_perfect_pass': int(n_perfect),
        'perfect_pass_rate': float(n_perfect / max(n_total, 1)),
        'mean_cc': float(mean_cc) if not np.isnan(mean_cc) else None,
        'median_cc': float(median_cc) if not np.isnan(median_cc) else None,
        'oof_correlation': float(oof_corr) if oof_corr is not None and not np.isnan(oof_corr) else None,
    }

    if stage2:
        result['optimal_threshold'] = stage2['optimal_threshold']
        result['peak_wald'] = stage2['peak_wald']
        result['mean_pr_below_threshold'] = stage2['mean_pass_rate_below']
        result['mean_pr_above_threshold'] = stage2['mean_pass_rate_above']
        result['performance_drop'] = stage2['diff']
    else:
        result['optimal_threshold'] = None
        result['peak_wald'] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="Cross-model IV analysis")
    parser.add_argument("--scored-dir", default=os.path.join("data", "scored"))
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find scored files
    scored_files = sorted([
        f for f in os.listdir(args.scored_dir)
        if f.endswith('.jsonl') and not f.startswith('_')
    ])

    if args.only:
        scored_files = [f for f in scored_files if any(m in f for m in args.only)]

    print(f"Found {len(scored_files)} scored model files")
    print()

    all_results = {}
    for sf in scored_files:
        model_name = sf.replace('.jsonl', '')
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        scored_path = os.path.join(args.scored_dir, sf)
        result = analyze_model(scored_path, IV_FEATURE_COLS)

        if result:
            result['model_id'] = model_name
            all_results[model_name] = result
        print()

    # Save full results
    results_path = os.path.join(args.output_dir, "cross_model_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create summary table
    if all_results:
        summary_df = pd.DataFrame(all_results.values())
        cols = [
            'model_id', 'n_total', 'n_perfect_pass', 'perfect_pass_rate',
            'mean_cc', 'oof_correlation', 'optimal_threshold', 'peak_wald',
            'mean_pr_below_threshold', 'mean_pr_above_threshold', 'performance_drop',
        ]
        cols = [c for c in cols if c in summary_df.columns]
        summary_df = summary_df[cols]

        csv_path = os.path.join(args.output_dir, "cross_model_summary.csv")
        summary_df.to_csv(csv_path, index=False)

        print("=" * 80)
        print("CROSS-MODEL SUMMARY")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print()
        print(f"Full results: {results_path}")
        print(f"Summary CSV:  {csv_path}")

        # Key finding: is the threshold consistent across models?
        thresholds = summary_df['optimal_threshold'].dropna()
        if len(thresholds) > 1:
            print(f"\nThreshold range: {thresholds.min():.1f} - {thresholds.max():.1f}")
            print(f"Threshold mean:  {thresholds.mean():.1f} +/- {thresholds.std():.1f}")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()
