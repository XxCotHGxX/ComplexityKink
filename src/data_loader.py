"""
Canonical data loading module for the Complexity Kink Research pipeline.

SINGLE SOURCE OF TRUTH for:
  - Pass rate computation
  - JSONL loading & field extraction
  - Kappa handling (NaN for missing/unparsable ,  NEVER silently default to 1)

Every pipeline stage MUST import from here.  No file should re-implement
these functions locally.

Design decisions:
  - ``kappa_actual`` is np.nan when the source value is null or missing.
    A default of 1 would push unparsable / failed generations onto the
    low end of the complexity axis and manufacture exactly the pass-rate
    collapse at low kappa that the paper is trying to measure honestly.
  - There is no ``is_success`` concept. Stage 1 filters directly on
    ``pass_rate == 1.0`` because only fully correct code has a trustworthy
    cyclomatic complexity: partial solutions have missing branches and
    incomplete loops, both of which deflate CC relative to the intended
    solution.
  - ``pass_rate`` is computed once by the feature extractor and stored in
    the JSONL. Downstream stages read it; they never recompute it, so the
    pipeline cannot silently apply different definitions at different stages.
"""
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Only samples where ALL tests pass have trustworthy cyclomatic complexity.
# Stage 1 uses this to select its training set.
PERFECT_PASS_RATE = 1.0


# ---------------------------------------------------------------------------
# Pass-rate computation  (called ONLY by the feature extractor at write time)
# ---------------------------------------------------------------------------

def compute_pass_rate(status_field):
    """
    Compute pass rate from a test-status field.

    Parameters
    ----------
    status_field : list[str] | str | None
        Raw ``status`` value from a scored JSONL record.

    Returns
    -------
    float
        Value in [0.0, 1.0].

    Notes
    -----
    This function is the ONLY place pass-rate logic should live.
    ``feature_extractor_iv.py`` calls it once when writing the enriched
    JSONL.  Every downstream file reads the stored ``pass_rate`` value
    instead of recomputing.
    """
    if status_field is None:
        return 0.0

    # Handle stringified list (e.g. "['pass', 'fail']")
    if isinstance(status_field, str):
        try:
            import ast
            status_field = ast.literal_eval(status_field)
        except (ValueError, SyntaxError):
            # Bare string like "pass" or "fail"
            return 1.0 if 'pass' in status_field.lower() else 0.0

    if isinstance(status_field, list) and len(status_field) > 0:
        n_pass = sum(
            1 for s in status_field
            if isinstance(s, str) and 'pass' in s.lower()
        )
        return n_pass / len(status_field)

    return 0.0


# ---------------------------------------------------------------------------
# Dataset loading  (called by stage1, stage2, viz, verify, check scripts)
# ---------------------------------------------------------------------------

def load_enriched_dataset(file_path, feature_cols=None):
    """
    Canonical loader for the enriched JSONL dataset.

    Parameters
    ----------
    file_path : str
        Path to the enriched (or OOF-enriched) JSONL file.
    feature_cols : list[str] | None
        IV feature column names to look for.  Defaults to
        ``config.IV_FEATURE_COLS``.

    Returns
    -------
    df : pd.DataFrame
        One row per sample.  ``kappa_actual`` is np.nan when the source
        record has no valid cyclomatic complexity.
    available_cols : list[str]
        Subset of *feature_cols* actually present in the data.
    """
    from config import IV_FEATURE_COLS
    if feature_cols is None:
        feature_cols = IV_FEATURE_COLS

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            row = dict(item.get('iv_features', {}))

            # --- Cyclomatic complexity ---
            # None / null  to  np.nan  (unknown / unparsable)
            # Numeric      to  float   (known)
            kappa_raw = item.get('kappa_cyclomatic', None)
            row['kappa_actual'] = (
                float(kappa_raw) if kappa_raw is not None else np.nan
            )

            # --- Pass rate (read stored value ,  never recompute) ---
            pr = item.get('pass_rate', None)
            row['pass_rate'] = float(pr) if pr is not None else 0.0

            # --- Predicted kappa from Stage 1 OOF (may be absent) ---
            kp = item.get('kappa_predicted', None)
            row['kappa_predicted'] = (
                float(kp) if kp is not None else np.nan
            )

            # --- Alternative Stage 1 predictor fit on all samples.
            # Present only after train_stage1_iv has been rerun with the
            # sensitivity pass. Stage 2 uses this for the robustness run
            # that stress-tests the perfect-pass survivorship filter.
            kp_all = item.get('kappa_predicted_all_samples', None)
            row['kappa_predicted_all_samples'] = (
                float(kp_all) if kp_all is not None else np.nan
            )

            # --- Metadata ---
            row['lang'] = item.get('lang', 'unknown')
            row['id'] = item.get('id', '')

            data.append(row)

    df = pd.DataFrame(data)

    # Determine which IV features are actually present
    available_cols = [c for c in feature_cols if c in df.columns]

    n_valid_kappa = df['kappa_actual'].notna().sum()
    n_nan_kappa = df['kappa_actual'].isna().sum()
    n_perfect = (df['pass_rate'] == PERFECT_PASS_RATE).sum()

    print(f"Loaded {len(df)} samples from {file_path}")
    print(f"  kappa_actual : {n_valid_kappa} valid, {n_nan_kappa} NaN")
    print(f"  pass_rate    : mean {df['pass_rate'].mean():.3f}")
    print(f"  perfect pass : {n_perfect} ({n_perfect/len(df)*100:.1f}%)")
    print(f"  IV features  : {len(available_cols)}/{len(feature_cols)}")

    return df, available_cols
