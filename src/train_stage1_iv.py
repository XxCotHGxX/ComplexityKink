"""
Stage 1: Train a complexity predictor using instruction-based features.
Uses 10-fold cross-validation with out-of-fold predictions for valid inference.
Includes sensitivity analysis: success-only vs. all-samples training.
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
    DEFAULT_ENRICHED_FILE, DEFAULT_MODEL_PATH,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE, CV_FOLDS,
    IV_FEATURE_COLS
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


def load_data(file_path):
    """Load enriched dataset into a DataFrame with IV features flattened."""
    print("Loading enriched dataset...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            row = dict(item.get('iv_features', {}))
            row['kappa_actual'] = item.get('kappa_cyclomatic', 1)
            row['is_success'] = item.get('is_success', 0)
            row['pass_rate'] = _compute_pass_rate(item)
            row['e_norm'] = item.get('e_norm', 0.0)
            row['m_mem_jaccard'] = item.get('m_mem_jaccard', 0.0)
            row['coupling_depth'] = item.get('coupling_depth', 0)
            row['lang'] = item.get('lang', 'unknown')
            data.append(row)
    
    df = pd.DataFrame(data)
    print(f"Total samples loaded: {len(df)}")
    return df


def train_with_cross_validation(df, feature_cols):
    """
    Train Stage 1 predictor with 10-fold cross-validation.
    Uses out-of-fold predictions so every sample gets a prediction
    from a model that never saw it (prevents overfitting leakage).
    
    CRITICAL DESIGN CHOICE: Train on successes only.
    Rationale: Only successful outputs have trustworthy kappa_actual as targets.
    Failed outputs produce truncated/broken code with misleading kappa.
    """
    # --- PRIMARY MODEL: Train on successes only ---
    success_mask = df['is_success'] == 1
    train_set = df[success_mask].copy()
    
    print(f"\n--- Stage 1: Success-Only Training ---")
    print(f"Training samples (successes): {len(train_set)}")
    print(f"Excluded samples (failures): {(~success_mask).sum()}")
    
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
    
    # For success samples, use out-of-fold predictions to avoid overfitting
    df.loc[success_mask, 'kappa_predicted_oof'] = oof_predictions
    df.loc[~success_mask, 'kappa_predicted_oof'] = model.predict(
        df.loc[~success_mask, feature_cols].values
    )
    
    return model, df


def sensitivity_analysis(df, feature_cols):
    """
    Sensitivity check: What if we train on ALL samples (including failures)?
    This helps assess survivorship bias in the success-only approach.
    """
    print("\n--- Sensitivity Analysis: All-Samples Training ---")
    
    X_all = df[feature_cols].values
    y_all = df['kappa_actual'].values
    
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
    preds_all = model_all.predict(X_all)
    
    # Compare predictions for failures between the two approaches
    fail_mask = df['is_success'] == 0
    if fail_mask.any():
        corr = np.corrcoef(df.loc[fail_mask, 'kappa_predicted'].values, 
                           model_all.predict(df.loc[fail_mask, feature_cols].values))[0, 1]
        print(f"Correlation of failure predictions (success-only vs all-sample): {corr:.4f}")
        print(f"  High correlation (>0.9) suggests survivorship bias is minimal.")
        print(f"  Low correlation (<0.7) suggests survivorship bias is a concern.")
    
    return model_all


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train complexity predictor")
    parser.add_argument("--input", default=DEFAULT_ENRICHED_FILE,
                        help="Path to enriched JSONL file")
    parser.add_argument("--output-model", default=DEFAULT_MODEL_PATH,
                        help="Path to save trained model")
    args = parser.parse_args()
    
    df = load_data(args.input)
    
    # Determine available feature columns
    available_cols = [c for c in IV_FEATURE_COLS if c in df.columns]
    if not available_cols:
        print("ERROR: No IV feature columns found in data.")
        sys.exit(1)
    print(f"Using {len(available_cols)} instrument features: {available_cols}")
    
    # Primary training (success-only, cross-validated)
    model, df = train_with_cross_validation(df, available_cols)
    
    # Save the model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    joblib.dump(model, args.output_model)
    print(f"\nModel saved to {args.output_model}")
    
    # Sanity check on fake-simple failures
    k1_fails = df[(df['kappa_actual'] == 1) & (df['is_success'] == 0)]
    if len(k1_fails) > 0:
        print(f"\nSanity Check: Predicted kappa for 'fake simple' failures (kappa_actual=1, fail):")
        print(f"  Count: {len(k1_fails)}")
        print(f"  Mean Predicted Kappa: {k1_fails['kappa_predicted'].mean():.2f}")
        print(f"  Median Predicted Kappa: {k1_fails['kappa_predicted'].median():.2f}")
        print(f"  Max Predicted Kappa: {k1_fails['kappa_predicted'].max():.2f}")
    
    # Sensitivity analysis
    sensitivity_analysis(df, available_cols)
    
    return df


if __name__ == "__main__":
    main()
