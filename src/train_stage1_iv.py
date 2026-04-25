"""
Stage 1: Train a complexity predictor using instruction-based features.
Uses 10-fold cross-validation with out-of-fold predictions for valid inference.
Includes sensitivity analysis: perfect-pass vs. all-samples training.

DATA PROVENANCE: Imports data loading from ``data_loader`` (single source
of truth).  Training set is filtered to ``pass_rate == 1.0`` ,  only fully
correct code has trustworthy cyclomatic complexity as a training target.
"""
import json
import argparse
import sys
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEFAULT_ENRICHED_FILE, DEFAULT_MODEL_PATH, DEFAULT_OOF_FILE,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE, CV_FOLDS,
    IV_FEATURE_COLS
)
from data_loader import load_enriched_dataset, PERFECT_PASS_RATE


def train_with_cross_validation(df, feature_cols):
    """
    Train Stage 1 predictor with 10-fold cross-validation.
    Uses out-of-fold predictions so every sample gets a prediction
    from a model that never saw it (prevents overfitting leakage).
    
    CRITICAL DESIGN CHOICE: Train only on samples with pass_rate == 1.0.
    Rationale: Only fully correct outputs (all tests pass) have trustworthy
    kappa_actual.  Partially correct code (e.g. 80% pass) may have missing
    branches, incomplete loops, or absent error handling ,  all of which
    deflate cyclomatic complexity relative to the intended solution.
    """
    # --- Filter to trustworthy training samples ---
    has_valid_kappa = df['kappa_actual'].notna()
    has_perfect_pass = df['pass_rate'] == PERFECT_PASS_RATE
    train_mask = has_valid_kappa & has_perfect_pass
    train_set = df[train_mask].copy()
    
    print(f"\n--- Stage 1: Perfect-Pass Training ---")
    print(f"Training samples (pass_rate == 1.0 & valid kappa): {len(train_set)}")
    print(f"Excluded: {(~train_mask).sum()} "
          f"({(~has_valid_kappa).sum()} NaN kappa, "
          f"{(has_valid_kappa & ~has_perfect_pass).sum()} imperfect pass)")
    
    X_train = train_set[feature_cols].values
    y_train = train_set['kappa_actual'].values
    
    # 10-fold cross-validation for honest R^2
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RF_RANDOM_STATE)
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    print(f"\n10-Fold Cross-Validated R-squared:")
    print(f"  Mean: {cv_scores.mean():.4f}")
    print(f"  Std:  {cv_scores.std():.4f}")
    print(f"  Min:  {cv_scores.min():.4f}")
    print(f"  Max:  {cv_scores.max():.4f}")
    
    cv_mae_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    print(f"\n10-Fold Cross-Validated MAE:")
    print(f"  Mean: {-cv_mae_scores.mean():.4f}")
    
    # Out-of-fold predictions for training set
    oof_predictions = cross_val_predict(model, X_train, y_train, cv=kf)
    oof_r2 = r2_score(y_train, oof_predictions)
    print(f"\nOut-of-Fold R-squared: {oof_r2:.4f}")
    
    # Fit final model on all successes for prediction on failures
    model.fit(X_train, y_train)
    
    # Feature Importance
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nTop Complexity Predictors in Instructions:")
    print(importances.to_string())
    
    # Apply predictions to the FULL dataset
    print("\nGenerating predicted kappa for full dataset...")
    X_full = df[feature_cols].values
    df['kappa_predicted'] = model.predict(X_full)
    
    # For training samples, use out-of-fold predictions to avoid overfitting
    df.loc[train_mask, 'kappa_predicted_oof'] = oof_predictions
    df.loc[~train_mask, 'kappa_predicted_oof'] = model.predict(
        df.loc[~train_mask, feature_cols].values
    )
    
    return model, df


def sensitivity_analysis(df, feature_cols):
    """Fit the all-samples variant of the Stage 1 predictor.

    The primary predictor is fit on ``pass_rate == 1.0`` only because
    partially-correct code has systematically deflated kappa. That filter
    is correct for the main result, but it also means the primary model
    never sees the failure regime during training ,  it extrapolates there.
    An extrapolation artefact near the pass/fail boundary could look like
    a structural break in pass_rate, so every run also produces an
    all-samples predictor and attaches its predictions to the dataset under
    ``kappa_predicted_all_samples``. Stage 2 re-runs the threshold analysis
    with this column as a robustness check; if the kink location and
    significance are similar under both, the result is not a survivorship
    artefact.
    """
    print("\n--- Sensitivity Analysis: All-Samples Training ---")

    valid_mask = df['kappa_actual'].notna()
    df_valid = df[valid_mask]

    X_all = df_valid[feature_cols].values
    y_all = df_valid['kappa_actual'].values

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RF_RANDOM_STATE)
    model_all = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1
    )

    cv_scores_all = cross_val_score(model_all, X_all, y_all, cv=kf, scoring='r2')
    print(f"All-Samples CV R-squared: {cv_scores_all.mean():.4f} (+/- {cv_scores_all.std():.4f})")

    model_all.fit(X_all, y_all)

    # Persist the all-samples predictor output for every row so Stage 2 can
    # run its robustness pass without having to refit here.
    df['kappa_predicted_all_samples'] = model_all.predict(df[feature_cols].values)

    non_perfect = df[df['pass_rate'] < PERFECT_PASS_RATE]
    if len(non_perfect) > 0:
        corr = np.corrcoef(
            non_perfect['kappa_predicted'].values,
            non_perfect['kappa_predicted_all_samples'].values,
        )[0, 1]
        print(f"Correlation of non-perfect predictions (perfect-only vs all-sample): {corr:.4f}")
        print(f"  Values near 1 indicate the perfect-pass filter does not materially")
        print(f"  change the predicted complexity for failing samples. Values well")
        print(f"  below 1 would mean the two models disagree in the regime where")
        print(f"  the primary model must extrapolate ,  flagged for follow-up.")

    return model_all


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train complexity predictor")
    parser.add_argument("--input", default=DEFAULT_ENRICHED_FILE,
                        help="Path to enriched JSONL file")
    parser.add_argument("--output-model", default=DEFAULT_MODEL_PATH,
                        help="Path to save trained model")
    parser.add_argument("--output-data", default=DEFAULT_OOF_FILE,
                        help="Path to save dataset with out-of-fold predictions")
    args = parser.parse_args()
    
    df, available_cols = load_enriched_dataset(args.input)
    
    if not available_cols:
        print("ERROR: No IV feature columns found in data.")
        sys.exit(1)
    print(f"Using {len(available_cols)} instrument features: {available_cols}")
    
    # Primary training (perfect-pass only, cross-validated).
    model, df = train_with_cross_validation(df, available_cols)

    # Save the primary model.
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(model, args.output_model)
    print(f"\nModel saved to {args.output_model}")

    # Sanity check: predicted kappa for NaN-kappa samples (parse failures).
    # Run before persistence so the summary numbers reflect what is written.
    nan_kappa = df[df['kappa_actual'].isna()]
    if len(nan_kappa) > 0:
        print(f"\nSanity Check: Predicted kappa for NaN-kappa samples (parse failures / missing code):")
        print(f"  Count: {len(nan_kappa)}")
        print(f"  Mean Predicted Kappa: {nan_kappa['kappa_predicted'].mean():.2f}")
        print(f"  Median Predicted Kappa: {nan_kappa['kappa_predicted'].median():.2f}")

    # Fit the all-samples variant and attach its predictions to the dataframe.
    # This must happen before we write the enriched file so both predictor
    # columns land in the JSONL in lock step.
    sensitivity_analysis(df, available_cols)

    # Write OOF predictions plus the all-samples predictor column. The OOF
    # predictor is the primary regressor for Stage 2; the all-samples column
    # is used only for the robustness pass that stress-tests the survivorship
    # assumption.
    out_lines = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            item['kappa_predicted'] = float(df.iloc[i]['kappa_predicted_oof'])
            if 'kappa_predicted_all_samples' in df.columns:
                item['kappa_predicted_all_samples'] = float(
                    df.iloc[i]['kappa_predicted_all_samples']
                )
            out_lines.append(json.dumps(item) + '\n')

    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)
    with open(args.output_data, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
    print(f"Data with OOF + all-samples predictions saved to {args.output_data}")

    return df


if __name__ == "__main__":
    main()
