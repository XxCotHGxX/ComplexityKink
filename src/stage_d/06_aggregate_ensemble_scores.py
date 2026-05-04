"""
Stage D step 6: aggregate long-format ensemble rubric scores.

Outputs:
  - data/stage_d/ensemble_scores_aggregated.jsonl
      one row per prompt_id with mean dimension scores, composite mean/std,
      dimension variances, and scorer count
  - data/stage_d/ensemble_reliability_report.json
      scorer coverage, pairwise correlations/agreement, and ICC estimates

The reliability calculations are intentionally transparent and lightweight:
ICC(2,1)-style two-way random effects estimates are reported separately for
each rubric dimension and the composite. They are not a substitute for human
calibration, but they are enough to find unstable dimensions before the human
subsample is scored.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "stage_d" / "ensemble_scores_long.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "stage_d" / "ensemble_scores_aggregated.jsonl"
DEFAULT_REPORT = ROOT / "data" / "stage_d" / "ensemble_reliability_report.json"
RUBRIC_DIMS = [
    "branching", "iteration", "state",
    "data_structures", "edge_cases", "composition",
]


def load_long_scores(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("error") or not rec.get("scores"):
                    continue
                row = {
                    "prompt_id": rec["prompt_id"],
                    "scorer_id": rec["scorer_id"],
                    "composite": float(rec["composite"]),
                    "rubric_hash": rec.get("rubric_hash"),
                }
                for dim in RUBRIC_DIMS:
                    row[dim] = float(rec["scores"][dim])
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["prompt_id", "scorer_id", "composite", *RUBRIC_DIMS])
    return pd.DataFrame(rows).drop_duplicates(["prompt_id", "scorer_id"], keep="last")


def icc_2_1(pivot: pd.DataFrame) -> float | None:
    """Two-way random-effects single-rater ICC approximation.

    Rows are targets/prompts, columns are scorers. Requires a complete matrix.
    """
    complete = pivot.dropna(axis=0, how="any")
    if complete.shape[0] < 2 or complete.shape[1] < 2:
        return None
    x = complete.to_numpy(dtype=float)
    n, k = x.shape
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((x - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + (k * (ms_cols - ms_err) / n)
    if denom <= 0:
        return None
    return float((ms_rows - ms_err) / denom)


def pairwise_metrics(df: pd.DataFrame, value_col: str) -> list[dict]:
    out = []
    scorers = sorted(df["scorer_id"].unique())
    for a, b in itertools.combinations(scorers, 2):
        pa = df[df["scorer_id"] == a][["prompt_id", value_col]]
        pb = df[df["scorer_id"] == b][["prompt_id", value_col]]
        merged = pa.merge(pb, on="prompt_id", suffixes=("_a", "_b"))
        if len(merged) < 2:
            continue
        va = merged[f"{value_col}_a"].to_numpy(dtype=float)
        vb = merged[f"{value_col}_b"].to_numpy(dtype=float)
        corr = np.corrcoef(va, vb)[0, 1] if np.std(va) > 0 and np.std(vb) > 0 else np.nan
        out.append({
            "scorer_a": a,
            "scorer_b": b,
            "n_overlap": int(len(merged)),
            "pearson": None if math.isnan(corr) else float(corr),
            "mean_abs_diff": float(np.mean(np.abs(va - vb))),
            "exact_agreement": float(np.mean(va == vb)),
            "within_one": float(np.mean(np.abs(va - vb) <= 1.0)),
        })
    return out


def aggregate(df: pd.DataFrame) -> list[dict]:
    rows = []
    for pid, group in df.groupby("prompt_id", sort=False):
        rec = {
            "prompt_id": pid,
            "n_scorers": int(group["scorer_id"].nunique()),
            "scorer_ids": sorted(group["scorer_id"].unique()),
            "rubric_hashes": sorted(h for h in group["rubric_hash"].dropna().unique()),
            "scores_mean": {},
            "scores_std": {},
            "scores_variance": {},
        }
        for dim in RUBRIC_DIMS:
            values = group[dim].astype(float)
            rec["scores_mean"][dim] = float(values.mean())
            rec["scores_std"][dim] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            rec["scores_variance"][dim] = float(values.var(ddof=1)) if len(values) > 1 else 0.0
        comp = group["composite"].astype(float)
        rec["composite_mean"] = float(comp.mean())
        rec["composite_std"] = float(comp.std(ddof=1)) if len(comp) > 1 else 0.0
        rec["composite_variance"] = float(comp.var(ddof=1)) if len(comp) > 1 else 0.0
        rec["mean_dimension_variance"] = float(
            np.mean(list(rec["scores_variance"].values()))
        )
        rows.append(rec)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage D ensemble rubric scores.")
    parser.add_argument("--input", nargs="+", default=[str(DEFAULT_INPUT)])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    df = load_long_scores([Path(path) for path in args.input])
    if df.empty:
        raise SystemExit("No successful ensemble score rows found.")

    agg_rows = aggregate(df)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in agg_rows:
            f.write(json.dumps(row) + "\n")

    reliability = {}
    for col in ["composite", *RUBRIC_DIMS]:
        pivot = df.pivot_table(index="prompt_id", columns="scorer_id", values=col, aggfunc="first")
        reliability[col] = {
            "icc_2_1_complete_cases": icc_2_1(pivot),
            "complete_case_n": int(pivot.dropna(axis=0, how="any").shape[0]),
            "pairwise": pairwise_metrics(df, col),
        }

    coverage = {
        "n_prompts": int(df["prompt_id"].nunique()),
        "n_scorers": int(df["scorer_id"].nunique()),
        "rows": int(len(df)),
        "rows_by_scorer": dict(Counter(df["scorer_id"])),
        "scorers_per_prompt": {
            str(k): int(v)
            for k, v in Counter(df.groupby("prompt_id")["scorer_id"].nunique()).items()
        },
        "rubric_hashes": sorted(h for h in df["rubric_hash"].dropna().unique()),
    }
    report = {"coverage": coverage, "reliability": reliability}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Aggregated ensemble scores")
    print(f"  Input rows: {len(df):,}")
    print(f"  Prompts:    {df['prompt_id'].nunique():,}")
    print(f"  Scorers:    {df['scorer_id'].nunique():,}")
    print(f"  Output:     {output}")
    print(f"  Report:     {args.report}")
    print("  Composite ICC:", reliability["composite"]["icc_2_1_complete_cases"])


if __name__ == "__main__":
    main()
