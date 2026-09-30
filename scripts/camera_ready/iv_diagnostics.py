"""Output-complexity and candidate-IV diagnostics for any outcome definition.

Recomputes, at the prompt level, the IV-related statistics reported in the
manuscript (the IV-diagnostics discussion, the appendix "Additional Econometric
Diagnostics", and the within-task-type Sargan table) for whichever scored panel
is passed via ``--scored-dir``. The per-model scored JSONL files are read with
the loaders in ``src/analyze_kink.py`` so the rows match the paper exactly;
``pass_rate`` is the outcome and ``kappa_cyclomatic`` is the generated-output
cyclomatic complexity (output CC). Nothing here depends on how ``pass_rate``
was defined, so the same script serves the reviewed outcome and the
independent audit.

Computed quantities
-------------------
1. Direct OLS of prompt mean pass on prompt mean output CC (HC1 SE) and the
   candidate-IV 2SLS with the six rubric dimensions as instruments
   (``linearmodels.IV2SLS``, robust SE), with the robust first-stage Wald
   chi2(6), partial R^2, the classical and HC1-robust first-stage joint F,
   first-stage coefficients (HC1), and the number of prompts whose mean uses
   all 21 model CC values.
2. Sargan J (df 5) and the robust Wooldridge score test for the full model.
3. Random-subsample Sargan J: ``--draws`` draws without replacement at each
   ``--subsample-sizes`` n, plus the deterministic full-sample row.
4. Just-identified single-instrument 2SLS for each rubric dimension.
5. Principal-component rotation of the six dimensions: variance shares,
   PC1/PC2 loadings, the PC1-only IV fit, and J for leading-k PCs, k = 2..6.
6. Post hoc instrument-subset search (``--subset-sizes``, default 2..5, which
   contains the requested 2- and 3-dimension search): Sargan p and robust
   first-stage Wald per subset, and the non-rejecting subsets (p > 0.05).
7. Full six-dimension Sargan J within each task category with at least
   ``--min-task-type-n`` prompts (labels: ``task_type`` in
   data/public_release/prompts, keyed by ``prompt_id``).

Implementation choices that reproduce the reviewed values
---------------------------------------------------------
These were checked against results/robustness_summary.json, the manuscript
appendix, and the rebuttal-stage outputs that produced them
(src/rebuttal/02_overid_diagnostics.py and 09_threshold_with_task_type.py,
which are not in this snapshot). With the reviewed outcome, every point
estimate, standard error, J statistic, first-stage statistic, loading, and
subsample mean agrees with those outputs to floating-point precision.

* Prompt means come from ``build_combined_df``. The outcome is the mean
  ``pass_rate`` over all 21 panel generations of the prompt (the mean-pooled
  outcome used throughout the paper). Generations without computable output CC
  still contribute their pass rate, and ``load_scored_model`` codes a missing
  ``pass_rate`` as 0. The regressor is the mean ``kappa_cyclomatic`` over the
  generations with computable CC (pandas skips missing values). So only the CC
  mean is restricted to generations with computable CC. Restricting the pass
  mean to those generations as well does not reproduce the reviewed values.
  Block ``prompt_mean_sensitivity`` reports that variant for items 1-2.
* The 2SLS and first-stage covariance is linearmodels ``cov_type="robust"``,
  which is heteroskedasticity-robust with no small-sample correction (HC0). The
  "robust first-stage Wald chi2(k)" is linearmodels'
  ``first_stage.diagnostics["f.stat"]`` under that covariance. It is a Wald
  statistic, not divided by k. ``partial.rsquared`` comes from the same table.
  Direct OLS and the first-stage coefficient table use statsmodels HC1.
* The appendix's "classical joint F(6,4993) = 2,265.4" is the HC1-robust joint
  F (the HC1 Wald chi2 divided by 6), reported as ``hc1_robust_joint_f``. The
  homoskedastic classical F is 1,779.4 (``classical_joint_f``).
* Sargan is linearmodels ``IVResults.sargan``, and the robust statistic is
  ``IVResults.wooldridge_overid``. P-values come from normal and chi-square
  survival functions. linearmodels computes ``1 - cdf``, which has about 1e-16
  absolute error. So the two agree to that error, and far-tail p-values here
  are more accurate and do not underflow to 0.
* Subsampling uses ``numpy.random.RandomState(--seed)`` (default 42) and one
  stream over the prompt frame sorted by ``prompt_id``, taken in the order of
  ``--subsample-sizes``. Each draw is ``rng.choice(N, n, replace=False)``. With
  the defaults this is the draw sequence of the rebuttal script, so the
  reviewed subsample means reproduce exactly, not just within Monte Carlo
  noise. Every without-replacement draw at n = N is a permutation of the full
  sample, so the n = N row is the full-sample J (one deterministic fit).
* PCA is an SVD of the column-centred, unstandardised six dimensions over the
  analysed prompts. This reproduces PC1 = 78.5% and PC1+PC2 = 87.7%.
  Standardised dimensions do not (they give 78.0% and 86.0%). Component signs
  are arbitrary, so each component is oriented with a nonnegative branching
  loading. That orientation matches the reported PC2 contrast and makes PC1
  load positively on all six dimensions. Signs do not affect any IV or J value.
* The release parquet's ``task_type`` equals the rebuttal's ``primary_type``
  label for all 5,000 prompts, so the within-category tests are unchanged.

Relative paths are resolved against the current working directory. Defaults
are anchored at the repository root. ``--scored-dir`` falls back to
``$CK_SCORED_DIR`` before ``data/stage_d/scored_combined``, following the other
camera-ready scripts. The panel is validated strictly: 21 models, one row per
prompt per model, and full prompt, rubric, and task-type coverage.
``--allow-incomplete-panel`` downgrades these checks to warnings.

Usage (from the repository root):
    python scripts/camera_ready/iv_diagnostics.py \
        --scored-dir data/stage_d/scored_combined \
        --out results/camera_ready/reviewed_outcome/iv_diagnostics.json
    python scripts/camera_ready/iv_diagnostics.py \
        --scored-dir data/stage_d/scored_independent_audit \
        --out results/camera_ready/independent_audit/iv_diagnostics.json
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats

import linearmodels
from linearmodels.iv import IV2SLS

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_kink import (  # noqa: E402
    RUBRIC_DIMS,
    build_combined_df,
    discover_models,
    load_rubric_scores,
    load_scored_model,
)
from config import STAGE_C_EXCLUDED_MODELS  # noqa: E402

OUTCOME = "pass_rate"
REGRESSOR = "kappa_cyclomatic"
ALPHA = 0.05
EXPECTED_MODELS = 21
DEFAULT_SEED = 42
DEFAULT_DRAWS = 200
DEFAULT_SUBSAMPLE_SIZES = (250, 500, 1000, 2000, 3000, 4000)
DEFAULT_SUBSET_SIZES = (2, 3, 4, 5)
DEFAULT_MIN_TASK_TYPE_N = 300


# ---------------------------------------------------------------------------
# Command line and small helpers
# ---------------------------------------------------------------------------

def _int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Output-CC and candidate-IV diagnostics for any outcome definition.",
    )
    parser.add_argument(
        "--scored-dir", type=Path,
        default=Path(os.environ.get("CK_SCORED_DIR", ROOT / "data" / "stage_d" / "scored_combined")),
        help="Per-model scored JSONL directory (pass_rate = outcome). "
             "Default: $CK_SCORED_DIR, else data/stage_d/scored_combined.",
    )
    parser.add_argument(
        "--rubric", type=Path,
        default=ROOT / "data" / "stage_d" / "ensemble_scores_current_aggregated.jsonl",
    )
    parser.add_argument(
        "--prompts", type=Path, default=ROOT / "data" / "stage_d" / "stage_d_prompts.jsonl",
        help="Prompt manifest; its prompt ids define the expected analysis set.",
    )
    parser.add_argument(
        "--task-types", type=Path, default=ROOT / "data" / "public_release" / "prompts",
        help="Parquet file or directory with prompt_id and task_type columns.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--subsample-sizes", type=_int_tuple, default=DEFAULT_SUBSAMPLE_SIZES)
    parser.add_argument("--subset-sizes", type=_int_tuple, default=DEFAULT_SUBSET_SIZES)
    parser.add_argument("--min-task-type-n", type=int, default=DEFAULT_MIN_TASK_TYPE_N)
    parser.add_argument("--expected-models", type=int, default=EXPECTED_MODELS)
    parser.add_argument(
        "--allow-incomplete-panel", action="store_true",
        help="Report panel-coverage failures as warnings instead of aborting.",
    )
    args = parser.parse_args(argv)
    if args.draws <= 0:
        parser.error("--draws must be positive")
    bad = [k for k in args.subset_sizes if not 2 <= k < len(RUBRIC_DIMS)]
    if bad:
        parser.error(f"--subset-sizes must lie in 2..{len(RUBRIC_DIMS) - 1}; got {bad}")
    return args


def _display_path(path: Path) -> str:
    """Repository-relative path when possible (symlinks are not resolved)."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _p_normal(z: float) -> float:
    return float(2.0 * stats.norm.sf(abs(z)))


def _p_chi2(stat: float, df: int) -> float:
    return float(stats.chi2.sf(stat, df))


def _clean(obj):
    """Convert numpy scalars and non-finite floats into strict-JSON values."""
    if isinstance(obj, dict):
        return {str(key): _clean(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(value) for value in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def load_prompt_ids(path: Path) -> tuple[list[str], int]:
    ids = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(json.loads(line)["prompt_id"])
    return ids, len(ids) - len(set(ids))


def load_task_types(path: Path) -> tuple[pd.Series, int]:
    labels = pd.read_parquet(path, columns=["prompt_id", "task_type"])
    duplicates = int(labels["prompt_id"].duplicated().sum())
    labels = labels.drop_duplicates("prompt_id", keep="last")
    return labels.set_index("prompt_id")["task_type"], duplicates


def audit_scored_file(path: str, rubric_ids: set[str]) -> dict:
    """Raw-file facts that ``load_scored_model`` absorbs silently."""
    rows = decode_errors = not_in_rubric = null_pass = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                decode_errors += 1
                continue
            if record.get("id") not in rubric_ids:
                not_in_rubric += 1
                continue
            if record.get("pass_rate") is None:
                null_pass += 1
    return {
        "raw_rows": rows,
        "json_decode_errors_skipped": decode_errors,
        "rows_without_rubric_match_dropped": not_in_rubric,
        "missing_pass_rate_coded_zero": null_pass,
    }


def load_panel(args: argparse.Namespace) -> dict:
    scored_dir = Path(args.scored_dir)
    for path in (scored_dir, args.rubric, args.prompts, args.task_types):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    prompt_ids, duplicate_prompt_ids = load_prompt_ids(args.prompts)
    rubric = load_rubric_scores(str(args.rubric))
    rubric_ids = set(rubric)
    task_types, duplicate_task_ids = load_task_types(args.task_types)

    model_frames: dict[str, pd.DataFrame] = {}
    model_files: dict[str, dict] = {}
    excluded = []
    for name, path in discover_models(str(scored_dir)):
        if name in STAGE_C_EXCLUDED_MODELS:
            excluded.append(name)
            continue
        frame = load_scored_model(path, rubric)
        frame[REGRESSOR] = pd.to_numeric(frame[REGRESSOR], errors="raise").astype(float)
        model_frames[name] = frame
        model_files[name] = {
            "file": Path(path).name,
            "sha256": _sha256(Path(path)),
            **audit_scored_file(path, rubric_ids),
            "analysed_rows": int(len(frame)),
            "unique_prompt_ids": int(frame["prompt_id"].nunique()),
            "missing_output_cc": int(frame[REGRESSOR].isna().sum()),
        }
    if not model_frames:
        raise RuntimeError(f"no scored model files found in {scored_dir}")

    return {
        "scored_dir": scored_dir,
        "prompt_ids": prompt_ids,
        "duplicate_prompt_ids": duplicate_prompt_ids,
        "rubric": rubric,
        "task_types": task_types,
        "duplicate_task_type_ids": duplicate_task_ids,
        "model_frames": model_frames,
        "model_files": model_files,
        "excluded_models": excluded,
    }


def build_prompt_frames(panel: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prompt-level frames: the paper definition and the CC-matched variant."""
    model_frames = panel["model_frames"]
    long = pd.concat(
        [df[["prompt_id", OUTCOME, REGRESSOR]].assign(model=name) for name, df in model_frames.items()],
        ignore_index=True,
    )

    # Paper definition (reproduces the reviewed values): pass averaged over all
    # panel generations, output CC averaged over generations where it is computable.
    frame = build_combined_df(model_frames)
    frame = frame.sort_values("prompt_id").reset_index(drop=True)
    grouped = long.groupby("prompt_id")
    frame["n_generations"] = frame["prompt_id"].map(grouped["model"].count())
    frame["n_cc_values"] = frame["prompt_id"].map(grouped[REGRESSOR].count())
    frame["task_type"] = frame["prompt_id"].map(panel["task_types"])

    # Sensitivity variant: both means over the generations with computable CC.
    computable = long.dropna(subset=[REGRESSOR])
    matched = computable.groupby("prompt_id").agg(
        **{OUTCOME: (OUTCOME, "mean"), REGRESSOR: (REGRESSOR, "mean")}
    ).reset_index()
    matched = frame[["prompt_id", *RUBRIC_DIMS]].merge(matched, on="prompt_id", how="inner")
    matched = matched.sort_values("prompt_id").reset_index(drop=True)
    return frame, matched, long


def validate_panel(panel: dict, frame: pd.DataFrame, long: pd.DataFrame,
                   expected_models: int) -> tuple[dict, list[str]]:
    prompt_set = set(panel["prompt_ids"])
    n_prompts = len(prompt_set)
    failures = []

    if panel["duplicate_prompt_ids"]:
        failures.append(f"prompt manifest has {panel['duplicate_prompt_ids']} duplicate ids")
    missing_rubric = len(prompt_set - set(panel["rubric"]))
    if missing_rubric:
        failures.append(f"{missing_rubric} manifest prompts lack rubric scores")
    if len(panel["model_frames"]) != expected_models:
        failures.append(f"found {len(panel['model_frames'])} models, expected {expected_models}")
    for name, frame_m in panel["model_frames"].items():
        ids = set(frame_m["prompt_id"])
        if len(frame_m) != n_prompts or len(ids) != len(frame_m) or ids != prompt_set:
            failures.append(
                f"{name}: {len(frame_m)} rows / {len(ids)} unique ids; "
                f"{len(prompt_set - ids)} manifest prompts missing, {len(ids - prompt_set)} extra"
            )
    analysed = set(frame["prompt_id"])
    if analysed != prompt_set:
        failures.append(
            f"analysis frame has {len(analysed)} prompts; {len(prompt_set - analysed)} manifest "
            f"prompts missing and {len(analysed - prompt_set)} extra"
        )
    missing_task = int(frame["task_type"].isna().sum())
    if missing_task:
        failures.append(f"{missing_task} analysed prompts lack a task_type label")
    if panel["duplicate_task_type_ids"]:
        failures.append(f"task-type file has {panel['duplicate_task_type_ids']} duplicate prompt ids")
    total_null_pass = sum(v["missing_pass_rate_coded_zero"] for v in panel["model_files"].values())
    total_decode = sum(v["json_decode_errors_skipped"] for v in panel["model_files"].values())
    if total_decode:
        failures.append(f"{total_decode} undecodable JSON lines were skipped by the loader")

    cc_counts = frame["n_cc_values"].value_counts().sort_index(ascending=False)
    validation = {
        "manifest_prompts": n_prompts,
        "analysed_prompts": int(len(frame)),
        "expected_models": expected_models,
        "analysed_models": len(panel["model_frames"]),
        "excluded_models": sorted(panel["excluded_models"]),
        "models": sorted(panel["model_frames"]),
        "analysed_generations": int(len(long)),
        "generations_with_computable_output_cc": int(long[REGRESSOR].notna().sum()),
        "generations_without_computable_output_cc": int(long[REGRESSOR].isna().sum()),
        "missing_pass_rate_coded_zero": int(total_null_pass),
        "prompts_without_any_output_cc": int((frame["n_cc_values"] == 0).sum()),
        "prompts_with_all_models_cc": int((frame["n_cc_values"] == len(panel["model_frames"])).sum()),
        "prompts_with_fewer_models_cc": int((frame["n_cc_values"] < len(panel["model_frames"])).sum()),
        "prompts_by_number_of_cc_values": {str(k): int(v) for k, v in cc_counts.items()},
        "task_type_counts": {str(k): int(v) for k, v in frame["task_type"].value_counts().items()},
        "failures": failures,
    }
    return validation, failures


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _analysis_rows(frame: pd.DataFrame, instruments: Sequence[str]) -> pd.DataFrame:
    return frame.dropna(subset=[OUTCOME, REGRESSOR, *instruments])


def _iv_results(data: pd.DataFrame, instruments: Sequence[str], cov_type: str = "robust"):
    exog = pd.DataFrame({"const": 1.0}, index=data.index)
    model = IV2SLS(data[OUTCOME], exog, data[[REGRESSOR]], data[list(instruments)])
    return model.fit(cov_type=cov_type)


def iv_fit(frame: pd.DataFrame, instruments: Sequence[str], wooldridge: bool = False) -> dict:
    """2SLS of prompt mean pass on prompt mean output CC with ``instruments``."""
    data = _analysis_rows(frame, instruments)
    res = _iv_results(data, instruments)
    n = int(res.nobs)
    k = len(instruments)
    coef = float(res.params[REGRESSOR])
    se = float(res.std_errors[REGRESSOR])
    first = res.first_stage.diagnostics.loc[REGRESSOR]
    wald = float(first["f.stat"])
    out = {
        "instruments": list(instruments),
        "n": n,
        "coefficient": coef,
        "robust_se": se,
        "z": coef / se,
        "p_value": _p_normal(coef / se),
        "first_stage_robust_wald_chi2": wald,
        "first_stage_df": k,
        "first_stage_p_value": _p_chi2(wald, k),
        "partial_r_squared": float(first["partial.rsquared"]),
    }
    if k > 1:
        sargan = res.sargan
        j = float(sargan.stat)
        out.update({
            "sargan_j": j,
            "sargan_df": int(sargan.df),
            "sargan_p_value": _p_chi2(j, int(sargan.df)),
            "sargan_j_per_obs": j / n,
            "sargan_rejects_at_0_05": bool(_p_chi2(j, int(sargan.df)) < ALPHA),
        })
        if wooldridge:
            score = res.wooldridge_overid
            out.update({
                "wooldridge_robust_score": float(score.stat),
                "wooldridge_df": int(score.df),
                "wooldridge_p_value": _p_chi2(float(score.stat), int(score.df)),
            })
    return out


def sargan_only(data: pd.DataFrame, instruments: Sequence[str]) -> tuple[float, int]:
    # J does not depend on the covariance estimator, so skip the robust one.
    sargan = _iv_results(data, instruments, cov_type="unadjusted").sargan
    return float(sargan.stat), int(sargan.df)


def direct_ols(frame: pd.DataFrame) -> dict:
    data = frame.dropna(subset=[OUTCOME, REGRESSOR])
    design = sm.add_constant(data[[REGRESSOR]], has_constant="add")
    res = sm.OLS(data[OUTCOME], design).fit(cov_type="HC1")
    coef = float(res.params[REGRESSOR])
    se = float(res.bse[REGRESSOR])
    return {
        "n": int(res.nobs),
        "coefficient": coef,
        "hc1_se": se,
        "z": coef / se,
        "p_value": _p_normal(coef / se),
        "intercept": float(res.params["const"]),
        "r_squared": float(res.rsquared),
    }


def first_stage_ols(frame: pd.DataFrame) -> dict:
    data = frame.dropna(subset=[REGRESSOR, *RUBRIC_DIMS])
    design = sm.add_constant(data[RUBRIC_DIMS], has_constant="add")
    hc1 = sm.OLS(data[REGRESSOR], design).fit(cov_type="HC1")
    classical = sm.OLS(data[REGRESSOR], design).fit()
    df_num, df_denom = int(classical.df_model), int(classical.df_resid)
    return {
        "n": int(hc1.nobs),
        "coefficients_hc1": {
            name: {
                "coefficient": float(hc1.params[name]),
                "hc1_se": float(hc1.bse[name]),
                "p_value": _p_normal(float(hc1.params[name] / hc1.bse[name])),
            }
            for name in ["const", *RUBRIC_DIMS]
        },
        "r_squared": float(classical.rsquared),
        "joint_f_df": [df_num, df_denom],
        "classical_joint_f": float(classical.fvalue),
        "classical_joint_f_p_value": float(stats.f.sf(classical.fvalue, df_num, df_denom)),
        "hc1_robust_joint_f": float(hc1.fvalue),
        "hc1_robust_joint_f_p_value": float(stats.f.sf(hc1.fvalue, df_num, df_denom)),
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def subsample_overid(frame: pd.DataFrame, full_fit: dict, sizes: Sequence[int],
                     draws: int, seed: int) -> dict:
    data = _analysis_rows(frame, RUBRIC_DIMS).reset_index(drop=True)
    n_total = len(data)
    rng = np.random.RandomState(seed)
    rows = []
    for n in sizes:
        if n > n_total:
            rows.append({"n": n, "skipped": f"exceeds the {n_total} analysed prompts"})
            continue
        j_values, p_values, failures = [], [], 0
        df = None
        for _ in range(draws):
            idx = rng.choice(n_total, size=n, replace=False)
            try:
                j, df = sargan_only(data.iloc[idx], RUBRIC_DIMS)
            except Exception:  # noqa: BLE001 - record and continue, as in the rebuttal
                failures += 1
                continue
            j_values.append(j)
            p_values.append(_p_chi2(j, df))
        j_arr = np.asarray(j_values, dtype=float)
        p_arr = np.asarray(p_values, dtype=float)
        rows.append({
            "n": n,
            "draws": draws,
            "successful_draws": int(len(j_arr)),
            "failed_draws": failures,
            "df": df,
            "mean_j": float(j_arr.mean()),
            "mean_j_per_obs": float(j_arr.mean() / n),
            "median_j": float(np.median(j_arr)),
            "sd_j": float(j_arr.std(ddof=1)) if len(j_arr) > 1 else None,
            "median_p_value": float(np.median(p_arr)),
            "share_rejecting_at_0_05": float(np.mean(p_arr < ALPHA)),
        })
    rows.append({
        "n": n_total,
        "draws": 1,
        "full_sample": True,
        "note": "Every without-replacement draw of size N is a permutation of the full "
                "sample, and J is permutation invariant.",
        "df": full_fit["sargan_df"],
        "mean_j": full_fit["sargan_j"],
        "mean_j_per_obs": full_fit["sargan_j_per_obs"],
        "median_p_value": full_fit["sargan_p_value"],
        "share_rejecting_at_0_05": float(full_fit["sargan_p_value"] < ALPHA),
    })
    return {
        "seed": seed,
        "rng": "numpy.random.RandomState(seed); one stream over sizes in the listed order; "
               "rng.choice(N, n, replace=False) per draw over the prompt frame sorted by prompt_id",
        "draws_per_n": draws,
        "null_expectation_of_j": len(RUBRIC_DIMS) - 1,
        "rows": rows,
    }


def principal_components(frame: pd.DataFrame) -> dict:
    data = _analysis_rows(frame, RUBRIC_DIMS).copy()
    x = data[RUBRIC_DIMS].to_numpy(dtype=float)
    centred = x - x.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    vt = vt * np.where(vt[:, 0] < 0, -1.0, 1.0)[:, None]  # branching loading >= 0
    scores = centred @ vt.T
    shares = singular ** 2 / np.sum(singular ** 2)
    names = [f"pc{j + 1}" for j in range(len(RUBRIC_DIMS))]
    for j, name in enumerate(names):
        data[name] = scores[:, j]

    fits = []
    for k in range(1, len(names) + 1):
        fit = iv_fit(data, names[:k])
        fit["k"] = k
        fit["cumulative_variance_explained"] = float(shares[:k].sum())
        fits.append(fit)
    return {
        "decomposition": "SVD of column-centred, unstandardised rubric dimensions",
        "orientation": "each component signed so its branching loading is nonnegative",
        "variance_explained": [float(v) for v in shares],
        "variance_explained_pc1": float(shares[0]),
        "variance_explained_pc1_pc2": float(shares[:2].sum()),
        "loadings": {
            name.upper(): {dim: float(vt[j, i]) for i, dim in enumerate(RUBRIC_DIMS)}
            for j, name in enumerate(names)
        },
        "pc1_only": fits[0],
        "leading_k_fits": fits,
        "leading_k_overid": [
            {
                "k": fit["k"],
                "cumulative_variance_explained": fit["cumulative_variance_explained"],
                "sargan_j": fit["sargan_j"],
                "sargan_df": fit["sargan_df"],
                "sargan_p_value": fit["sargan_p_value"],
            }
            for fit in fits[1:]
        ],
    }


def subset_search(frame: pd.DataFrame, sizes: Sequence[int]) -> dict:
    fits = []
    for size in sizes:
        for combo in itertools.combinations(RUBRIC_DIMS, size):
            fit = iv_fit(frame, list(combo))
            fit["size"] = size
            fits.append(fit)
    passing = [fit for fit in fits if fit["sargan_p_value"] > ALPHA]
    walds = [fit["first_stage_robust_wald_chi2"] for fit in passing]
    return {
        "sizes_searched": list(sizes),
        "subsets_tested": len(fits),
        "subsets_tested_by_size": {str(s): sum(1 for f in fits if f["size"] == s) for s in sizes},
        "non_rejecting_rule": f"Sargan p > {ALPHA}",
        "non_rejecting_count": len(passing),
        "non_rejecting_count_by_size": {str(s): sum(1 for f in passing if f["size"] == s) for s in sizes},
        "non_rejecting": [
            {
                "instruments": fit["instruments"],
                "size": fit["size"],
                "sargan_j": fit["sargan_j"],
                "sargan_df": fit["sargan_df"],
                "sargan_p_value": fit["sargan_p_value"],
                "first_stage_robust_wald_chi2": fit["first_stage_robust_wald_chi2"],
                "coefficient": fit["coefficient"],
                "robust_se": fit["robust_se"],
                "p_value": fit["p_value"],
            }
            for fit in passing
        ],
        "non_rejecting_first_stage_wald_range": [min(walds), max(walds)] if walds else None,
        "all_subsets": fits,
    }


def within_task_type(frame: pd.DataFrame, min_n: int) -> dict:
    tested, untested = [], {}
    for task_type, subset in frame.groupby("task_type", sort=True):
        if len(subset) < min_n:
            untested[str(task_type)] = int(len(subset))
            continue
        fit = iv_fit(subset, RUBRIC_DIMS)
        tested.append({
            "task_type": str(task_type),
            "n": fit["n"],
            "sargan_j": fit["sargan_j"],
            "sargan_df": fit["sargan_df"],
            "sargan_p_value": fit["sargan_p_value"],
            "sargan_j_per_obs": fit["sargan_j_per_obs"],
            "rejects_at_0_05": fit["sargan_rejects_at_0_05"],
            "first_stage_robust_wald_chi2": fit["first_stage_robust_wald_chi2"],
            "coefficient": fit["coefficient"],
            "robust_se": fit["robust_se"],
        })
    tested.sort(key=lambda row: (-row["n"], row["task_type"]))
    return {
        "minimum_prompts": min_n,
        "categories_tested": len(tested),
        "categories_rejecting_at_0_05": sum(1 for row in tested if row["rejects_at_0_05"]),
        "tested": tested,
        "not_tested_below_minimum": untested,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse(args: argparse.Namespace) -> dict:
    started = time.time()
    panel = load_panel(args)
    frame, matched, long = build_prompt_frames(panel)
    validation, failures = validate_panel(panel, frame, long, args.expected_models)
    if failures:
        message = "panel validation failed: " + "; ".join(failures)
        if not args.allow_incomplete_panel:
            raise RuntimeError(message + " (use --allow-incomplete-panel to continue)")
        print("WARNING: " + message, file=sys.stderr)

    n_models = len(panel["model_frames"])
    full = iv_fit(frame, RUBRIC_DIMS, wooldridge=True)
    ols = direct_ols(frame)

    matched_full = iv_fit(matched, RUBRIC_DIMS, wooldridge=True)
    matched_ols = direct_ols(matched)

    report = {
        "schema_version": 1,
        "script": "scripts/camera_ready/iv_diagnostics.py",
        "script_sha256": _sha256(Path(__file__)),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "linearmodels": linearmodels.__version__,
        },
        "inputs": {
            "scored_dir": _display_path(panel["scored_dir"]),
            "rubric": _display_path(args.rubric),
            "rubric_sha256": _sha256(Path(args.rubric)),
            "prompts": _display_path(args.prompts),
            "prompts_sha256": _sha256(Path(args.prompts)),
            "task_types": _display_path(args.task_types),
            "task_type_column": "task_type",
            "outcome_field": OUTCOME,
            "regressor_field": REGRESSOR,
            "scored_files": panel["model_files"],
        },
        "method": {
            "unit": "prompt",
            "outcome": "mean pass_rate over all panel generations of the prompt "
                       "(build_combined_df; a missing pass_rate is coded 0 by load_scored_model)",
            "regressor": "mean kappa_cyclomatic over the panel generations with computable output CC",
            "instruments": list(RUBRIC_DIMS),
            "ols_covariance": "statsmodels HC1",
            "iv_covariance": "linearmodels IV2SLS cov_type='robust' "
                             "(heteroskedasticity-robust, no small-sample correction)",
            "first_stage_wald": "linearmodels first_stage.diagnostics f.stat under the robust "
                                "covariance: Wald chi2(k), not divided by k",
            "partial_r_squared": "linearmodels first_stage.diagnostics partial.rsquared",
            "overidentification": "linearmodels IVResults.sargan and IVResults.wooldridge_overid",
            "p_values": "normal / chi-square survival functions",
            "alpha": ALPHA,
        },
        "validation": validation,
        "output_cc_regressions": {
            "n_prompts": ols["n"],
            "n_models": n_models,
            "prompts_with_all_models_cc": validation["prompts_with_all_models_cc"],
            "prompts_with_fewer_models_cc": validation["prompts_with_fewer_models_cc"],
            "direct_ols": ols,
            "candidate_iv_2sls": {
                key: full[key]
                for key in ("instruments", "n", "coefficient", "robust_se", "z", "p_value",
                            "first_stage_robust_wald_chi2", "first_stage_df", "first_stage_p_value",
                            "partial_r_squared")
            },
            "first_stage_ols": first_stage_ols(frame),
        },
        "overidentification": {
            "n": full["n"],
            "sargan_j": full["sargan_j"],
            "sargan_df": full["sargan_df"],
            "sargan_p_value": full["sargan_p_value"],
            "sargan_j_per_obs": full["sargan_j_per_obs"],
            "wooldridge_robust_score": full["wooldridge_robust_score"],
            "wooldridge_df": full["wooldridge_df"],
            "wooldridge_p_value": full["wooldridge_p_value"],
        },
        "subsample_overidentification": subsample_overid(
            frame, full, args.subsample_sizes, args.draws, args.seed
        ),
        "just_identified": {dim: iv_fit(frame, [dim]) for dim in RUBRIC_DIMS},
        "principal_components": principal_components(frame),
        "subset_search": subset_search(frame, args.subset_sizes),
        "within_task_type": within_task_type(frame, args.min_task_type_n),
        "prompt_mean_sensitivity": {
            "definition": "pass_rate and kappa_cyclomatic both averaged over only the generations "
                          "with computable output CC (not the paper definition)",
            "n_prompts": matched_ols["n"],
            "direct_ols": matched_ols,
            "candidate_iv_2sls": {
                key: matched_full[key]
                for key in ("n", "coefficient", "robust_se", "p_value",
                            "first_stage_robust_wald_chi2", "partial_r_squared")
            },
            "sargan_j": matched_full["sargan_j"],
            "sargan_p_value": matched_full["sargan_p_value"],
            "wooldridge_robust_score": matched_full["wooldridge_robust_score"],
            "wooldridge_p_value": matched_full["wooldridge_p_value"],
        },
    }
    report["runtime_seconds"] = round(time.time() - started, 1)
    return _clean(report)


def print_summary(report: dict) -> None:
    reg = report["output_cc_regressions"]
    ols, iv = reg["direct_ols"], reg["candidate_iv_2sls"]
    ov = report["overidentification"]
    print(f"prompts={reg['n_prompts']} models={reg['n_models']} "
          f"all-model CC={reg['prompts_with_all_models_cc']} fewer={reg['prompts_with_fewer_models_cc']}")
    print(f"OLS   beta={ols['coefficient']:.10f} HC1 SE={ols['hc1_se']:.8f} "
          f"p={ols['p_value']:.5g} R2={ols['r_squared']:.6f}")
    print(f"2SLS  beta={iv['coefficient']:.8f} robust SE={iv['robust_se']:.8f} p={iv['p_value']:.4g} "
          f"first-stage Wald={iv['first_stage_robust_wald_chi2']:.1f} partial R2={iv['partial_r_squared']:.4f}")
    print(f"Sargan J={ov['sargan_j']:.4f} (df {ov['sargan_df']}, p={ov['sargan_p_value']:.3g}); "
          f"Wooldridge={ov['wooldridge_robust_score']:.2f}")
    for row in report["subsample_overidentification"]["rows"]:
        if "mean_j" in row:
            print(f"  n={row['n']:>5}: mean J={row['mean_j']:.2f} J/n={row['mean_j_per_obs']:.4f} "
                  f"reject={row['share_rejecting_at_0_05']:.3f}")
    for dim, fit in report["just_identified"].items():
        print(f"  just-identified {dim:>16}: beta={fit['coefficient']:+.5f} p={fit['p_value']:.3g}")
    pcs = report["principal_components"]
    pc1 = pcs["pc1_only"]
    print(f"PC1 share={pcs['variance_explained_pc1']:.4f} PC1+2={pcs['variance_explained_pc1_pc2']:.4f}; "
          f"PC1-only beta={pc1['coefficient']:.6f} SE={pc1['robust_se']:.6f} p={pc1['p_value']:.3f} "
          f"Wald={pc1['first_stage_robust_wald_chi2']:.1f}")
    for row in pcs["leading_k_overid"]:
        print(f"  PC1..PC{row['k']}: J={row['sargan_j']:.2f} p={row['sargan_p_value']:.3g}")
    search = report["subset_search"]
    print(f"subsets tested={search['subsets_tested']} non-rejecting={search['non_rejecting_count']}")
    for row in search["non_rejecting"]:
        print(f"  {'+'.join(row['instruments']):>34}: p={row['sargan_p_value']:.3f} "
              f"Wald={row['first_stage_robust_wald_chi2']:.1f}")
    for row in report["within_task_type"]["tested"]:
        print(f"  {row['task_type']:>28} n={row['n']:>5} J={row['sargan_j']:.2f} "
              f"p={row['sargan_p_value']:.3g} J/n={row['sargan_j_per_obs']:.4f}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = analyse(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"Wrote {out} ({report['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
