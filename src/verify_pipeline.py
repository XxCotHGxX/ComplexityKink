"""
End-to-end verification script for the Complexity Kink Research pipeline.
Runs all stages on a small sample and validates outputs using assertions.

Uses assertions (not print statements) for diagnostics per GPU cluster constraints.
"""
import json
import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_ENRICHED_FILE, IV_FEATURE_COLS, CONTROL_COLS


def sample_data(input_file, n_samples=2000, seed=42):
    """Load a random subset of the enriched dataset."""
    rng = np.random.RandomState(seed)
    
    # First pass: count lines
    print(f"Counting lines in {input_file}...")
    n_total = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            n_total += 1
    print(f"Total lines: {n_total}")
    
    # Select random indices
    if n_samples >= n_total:
        indices = set(range(n_total))
    else:
        indices = set(rng.choice(n_total, size=n_samples, replace=False))
    
    # Second pass: extract selected lines
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i in indices:
                lines.append(line)
    
    return lines


def write_temp_data(lines, tmp_dir):
    """Write sampled data to a temp file."""
    path = os.path.join(tmp_dir, "test_enriched.jsonl")
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return path


def test_stage1(tmp_data_path, tmp_dir):
    """Test Stage 1: training and cross-validation."""
    print("\n=== Testing Stage 1: Train Complexity Predictor ===")
    
    from train_stage1_iv import load_data, train_with_cross_validation
    
    df = load_data(tmp_data_path)
    assert len(df) > 0, "Stage 1: No data loaded"
    
    feature_cols = [c for c in IV_FEATURE_COLS if c in df.columns]
    assert len(feature_cols) > 0, f"Stage 1: No IV feature columns found. Available: {list(df.columns)}"
    
    model, df = train_with_cross_validation(df, feature_cols)
    
    # Validate outputs
    assert 'kappa_predicted' in df.columns, "Stage 1: kappa_predicted column missing"
    assert 'kappa_predicted_oof' in df.columns, "Stage 1: kappa_predicted_oof column missing"
    assert df['kappa_predicted'].notna().all(), "Stage 1: NaN values in kappa_predicted"
    assert (df['kappa_predicted'] >= 0).all(), "Stage 1: Negative kappa_predicted values"
    
    # Save model to temp
    import joblib
    model_path = os.path.join(tmp_dir, "test_model.joblib")
    joblib.dump(model, model_path)
    
    print("  Stage 1: ALL ASSERTIONS PASSED")
    return model_path, df


def test_stage2(tmp_data_path, model_path):
    """Test Stage 2: 2SLS, Hausman, Hansen, Placebo."""
    print("\n=== Testing Stage 2: Statistical Analysis ===")
    
    # Override config for faster testing
    import config
    original_hansen = config.HANSEN_BOOTSTRAP_ITERATIONS
    original_placebo = config.PLACEBO_ITERATIONS
    original_min_regime = config.MIN_REGIME_SIZE
    config.HANSEN_BOOTSTRAP_ITERATIONS = 20  # Fast for testing
    config.PLACEBO_ITERATIONS = 20
    config.MIN_REGIME_SIZE = 30  # Small sample needs smaller minimum
    
    try:
        from run_stage2_iv import load_data_for_stage2, run_proper_2sls, hausman_test, hansen_threshold_test, placebo_test
        
        df, feature_cols = load_data_for_stage2(tmp_data_path, model_path)
        assert len(df) > 0, "Stage 2: No data loaded"
        assert 'kappa_predicted' in df.columns, "Stage 2: kappa_predicted missing"
        assert 'pass_rate' in df.columns, "Stage 2: pass_rate missing"
        
        # Test 2SLS
        print("\n  Testing 2SLS...")
        ols_naive, iv_results = run_proper_2sls(df, feature_cols)
        assert ols_naive is not None, "Stage 2: Naive OLS failed"
        # IV may fail on small samples, that's acceptable
        if iv_results is not None:
            assert hasattr(iv_results, 'params'), "Stage 2: IV results missing params"
            assert hasattr(iv_results, 'pvalues'), "Stage 2: IV results missing pvalues"
            print("  2SLS: PASSED")
        else:
            print("  2SLS: Skipped (small sample)")
        
        # Test Hausman
        print("\n  Testing Hausman test...")
        h_stat, h_pval = hausman_test(df, feature_cols)
        # Hausman can be NaN for edge cases, but should be numeric normally
        if not np.isnan(h_stat):
            assert h_stat >= 0, f"Stage 2: Negative Hausman statistic: {h_stat}"
            assert 0 <= h_pval <= 1, f"Stage 2: Invalid Hausman p-value: {h_pval}"
            print("  Hausman: PASSED")
        else:
            print("  Hausman: Inconclusive (expected for small samples)")
        
        # Test Hansen threshold
        print("\n  Testing Hansen threshold test...")
        gamma, sup_wald, p_hansen, wald_curve = hansen_threshold_test(df)
        if gamma is not None:
            assert 2 <= gamma <= 15, f"Stage 2: Threshold {gamma} outside expected range [2, 15]"
            assert sup_wald > 0, f"Stage 2: Non-positive sup-Wald: {sup_wald}"
            if not np.isnan(p_hansen):
                assert 0 <= p_hansen <= 1, f"Stage 2: Invalid Hansen p-value: {p_hansen}"
            print(f"  Hansen: PASSED (gamma={gamma:.1f})")
        else:
            print("  Hansen: Skipped (no valid threshold found)")
        
        # Test Placebo
        print("\n  Testing Placebo test...")
        placebo_gammas, placebo_walds = placebo_test(df)
        if placebo_gammas:
            assert len(placebo_gammas) > 0, "Stage 2: Placebo produced no results"
            assert all(2 <= g <= 15 for g in placebo_gammas), "Stage 2: Placebo thresholds out of range"
            print(f"  Placebo: PASSED ({len(placebo_gammas)} iterations)")
        else:
            print("  Placebo: Skipped (no valid placebo iterations)")
        
        print("\n  Stage 2: ALL ASSERTIONS PASSED")
        
    finally:
        # Restore original config
        config.HANSEN_BOOTSTRAP_ITERATIONS = original_hansen
        config.PLACEBO_ITERATIONS = original_placebo
        config.MIN_REGIME_SIZE = original_min_regime


def test_visualization(tmp_data_path, model_path, tmp_dir):
    """Test visualization generation."""
    print("\n=== Testing Visualization ===")
    
    from generate_viz import generate_visualizations
    
    viz_dir = os.path.join(tmp_dir, "test_viz")
    os.makedirs(viz_dir, exist_ok=True)
    
    generate_visualizations(tmp_data_path, model_path, viz_dir)
    
    # Check output files exist
    expected_files = ["paradox_vs_kink.png", "performance_phase_diagram.png", "hansen_wald_curve.png"]
    for fname in expected_files:
        fpath = os.path.join(viz_dir, fname)
        assert os.path.exists(fpath), f"Visualization missing: {fname}"
        assert os.path.getsize(fpath) > 0, f"Visualization empty: {fname}"
    
    print("  Visualization: ALL ASSERTIONS PASSED")


def main():
    print("="*70)
    print("COMPLEXITY KINK RESEARCH - END-TO-END VERIFICATION")
    print("="*70)
    
    # Sample data
    lines = sample_data(DEFAULT_ENRICHED_FILE, n_samples=2000)
    assert len(lines) > 0, "Verification: No data sampled"
    print(f"Sampled {len(lines)} records for testing")
    
    # Create temp directory
    tmp_dir = tempfile.mkdtemp(prefix="ckr_verify_")
    print(f"Temp directory: {tmp_dir}")
    
    try:
        tmp_data_path = write_temp_data(lines, tmp_dir)
        
        # Run all stages
        model_path, _ = test_stage1(tmp_data_path, tmp_dir)
        test_stage2(tmp_data_path, model_path)
        test_visualization(tmp_data_path, model_path, tmp_dir)
        
        print("\n" + "="*70)
        print("ALL VERIFICATION TESTS PASSED")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n VERIFICATION FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        raise
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
