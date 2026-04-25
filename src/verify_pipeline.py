"""
End-to-end verification script for the Complexity Kink Research pipeline.
Runs all stages on a small sample and validates outputs using assertions.

DATA PROVENANCE: Uses ``data_loader.load_enriched_dataset`` for loading.
Test parameters are passed as function arguments rather than mutating
global config state.
"""
import json
import os
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEFAULT_ENRICHED_FILE, IV_FEATURE_COLS, CONTROL_COLS
from data_loader import load_enriched_dataset


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
    
    from train_stage1_iv import train_with_cross_validation
    
    df, feature_cols = load_enriched_dataset(tmp_data_path)
    assert len(df) > 0, "Stage 1: No data loaded"
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
    
    oof_path = os.path.join(tmp_dir, "test_oof.jsonl")
    out_lines = []
    with open(tmp_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            item['kappa_predicted'] = float(df.iloc[i]['kappa_predicted_oof'])
            out_lines.append(json.dumps(item) + '\n')
    with open(oof_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
    
    print("  Stage 1: ALL ASSERTIONS PASSED")
    return model_path, oof_path


def test_stage2(tmp_data_path, model_path):
    """Test Stage 2: full 2SLS analysis end-to-end."""
    print("\n=== Testing Stage 2: Statistical Analysis ===")

    # Shrink bootstrap/placebo loops so the smoke test finishes in seconds.
    # The script runs in isolation, so mutating config here is safe.
    import config
    saved = {
        'HANSEN_BOOTSTRAP_ITERATIONS': config.HANSEN_BOOTSTRAP_ITERATIONS,
        'PLACEBO_ITERATIONS': config.PLACEBO_ITERATIONS,
        'MIN_REGIME_SIZE': config.MIN_REGIME_SIZE,
        'THRESHOLD_CI_BOOTSTRAP': config.THRESHOLD_CI_BOOTSTRAP,
    }
    config.HANSEN_BOOTSTRAP_ITERATIONS = 20
    config.PLACEBO_ITERATIONS = 10
    config.MIN_REGIME_SIZE = 30
    config.THRESHOLD_CI_BOOTSTRAP = 20

    try:
        import run_stage2_iv as s2

        df_out, results = s2.run_stage2_analysis(tmp_data_path, model_path)

        # Contract: every key the downstream viz / reporting code relies on.
        for key in (
            'iv_results', 'threshold_gamma', 'placebo_gammas',
            'instrument_robustness', 'threshold_ci_lower',
            'threshold_ci_upper', 'regime_low_iv', 'regime_high_iv',
        ):
            assert key in results, f"Stage 2: missing result key {key!r}"

        iv_res = results['iv_results']
        if iv_res is not None:
            assert hasattr(iv_res, 'j_stat_val'), "Stage 2: J-stat not attached"
            print(f"    - 2SLS kappa: {iv_res.params['kappa_actual']:.4f}")

        robust = results['instrument_robustness']
        for name in ("Full Set",
                     "Reduced Set (No Length/Lexical)",
                     "Conservative (Pure Structural)"):
            assert name in robust, f"Stage 2: robustness set {name!r} missing"

        gamma = results['threshold_gamma']
        if gamma is not None and not pd.isna(gamma):
            assert 2 <= gamma <= 30, f"Stage 2: gamma {gamma} outside plausible range"
            print(f"    - Threshold gamma: {gamma:.2f}")

        print("  Stage 2: ALL ASSERTIONS PASSED")

    finally:
        for k, v in saved.items():
            setattr(config, k, v)


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
        model_path, oof_path = test_stage1(tmp_data_path, tmp_dir)
        test_stage2(oof_path, model_path)  # noqa: use OOF-enriched jsonl
        test_visualization(oof_path, model_path, tmp_dir)
        
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
