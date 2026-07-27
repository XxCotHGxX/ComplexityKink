"""Regenerate the public tail-extension summary tables from retained raw data.

The manuscript uses integer bins only for descriptive displays. This script
assigns a continuous prompt-composite score ``x`` to bin
``floor(x + 0.5)``, so bin ``b`` represents ``[b - 0.5, b + 0.5)`` and exact
half-point boundaries are assigned upward. Breakpoint estimation remains
unbinned and is outside this script.

The large raw prompt and generation bundles are retained separately from the
Git snapshot. Point ``--data-root`` at the root of that retained bundle when it
is not available below this repository.

Examples:
    python scripts/regenerate_tail_extension_tables.py
    python scripts/regenerate_tail_extension_tables.py --data-root D:/data-copy
    python scripts/regenerate_tail_extension_tables.py --data-root D:/data-copy --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from display_bins import half_open_integer_bin  # noqa: E402


MATCHED_MODELS = (
    "azure_deepseek-v3.2-speciale",
    "azure_gpt-oss-120b",
    "azure_kimi-k2.5",
    "azure_llama-3.3-70b",
    "azure_mistral-large-3",
)
FRONTIER_MODELS = (
    ("cli/claude-opus-4.6", "Claude Opus 4.6"),
    ("cli/gpt-5.4", "GPT-5.4"),
    ("cli/gemini-3.1-pro", "Gemini 3.1 Pro Preview"),
)
EXPECTED_EXTENSION_BINS = {15: 218, 16: 133, 17: 11, 18: 3}
EXPECTED_PANEL_PROMPTS = 5000
EXPECTED_EXTENSION_PROMPTS = 365
EXPECTED_FRONTIER_ANCHORS = 150


def read_jsonl(path: Path) -> Iterable[dict]:
    """Yield nonempty JSON objects from an append-only JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_extension_metadata(data_root: Path) -> pd.DataFrame:
    path = data_root / "data/rebuttal/tail_topup/tail_topup_final.jsonl"
    rows = list(read_jsonl(path))
    prompt_ids = [row["prompt_id"] for row in rows]
    require(
        len(rows) == EXPECTED_EXTENSION_PROMPTS,
        f"Expected {EXPECTED_EXTENSION_PROMPTS} retained extension prompts, "
        f"found {len(rows)}",
    )
    require(
        len(prompt_ids) == len(set(prompt_ids)),
        "Retained extension metadata contains duplicate prompt IDs",
    )

    frame = pd.DataFrame(
        {
            "prompt_id": prompt_ids,
            "composite": [float(row["rubric_composite"]) for row in rows],
        }
    )
    frame["bin"] = half_open_integer_bin(frame["composite"])
    counts = {
        int(bin_id): int(count)
        for bin_id, count in frame.groupby("bin").size().items()
    }
    require(
        counts == EXPECTED_EXTENSION_BINS,
        f"Unexpected extension support after half-open binning: {counts}",
    )
    return frame


def load_matched_frame(data_root: Path, extension_meta: pd.DataFrame) -> pd.DataFrame:
    tail_dir = data_root / "data/rebuttal/tail_topup/scored"
    panel_dir = data_root / "data/stage_d/scored_combined"
    ensemble_path = (
        data_root / "data/stage_d/ensemble_scores_current_aggregated.jsonl"
    )

    tail_stems = {path.stem for path in tail_dir.glob("*.jsonl")}
    panel_stems = {path.stem for path in panel_dir.glob("*.jsonl")}
    matched = tuple(sorted(tail_stems & panel_stems))
    require(
        matched == tuple(sorted(MATCHED_MODELS)),
        "Expected exactly the five locked matched models; "
        f"found {list(matched)}",
    )

    extension_composite = dict(
        zip(extension_meta["prompt_id"], extension_meta["composite"])
    )
    extension_ids = set(extension_composite)
    extension_rows: list[dict] = []
    for model in MATCHED_MODELS:
        path = tail_dir / f"{model}.jsonl"
        seen: set[str] = set()
        for row in read_jsonl(path):
            prompt_id = row.get("prompt_id")
            if prompt_id not in extension_ids or row.get("pass_rate") is None:
                continue
            require(
                prompt_id not in seen,
                f"Duplicate successful extension row for {model}/{prompt_id}",
            )
            seen.add(prompt_id)
            composite = float(row["rubric_composite"])
            require(
                np.isclose(composite, extension_composite[prompt_id]),
                f"Composite mismatch for extension prompt {prompt_id}",
            )
            extension_rows.append(
                {
                    "prompt_id": prompt_id,
                    "model": model,
                    "source": "Audit-clean extension",
                    "composite": composite,
                    "pass_rate": float(row["pass_rate"]),
                }
            )
        require(
            seen == extension_ids,
            f"Extension model {model} covers {len(seen)} of "
            f"{EXPECTED_EXTENSION_PROMPTS} retained prompts",
        )

    ensemble: dict[str, float] = {}
    for row in read_jsonl(ensemble_path):
        prompt_id = row["prompt_id"]
        require(
            prompt_id not in ensemble,
            f"Duplicate ensemble score for prompt {prompt_id}",
        )
        ensemble[prompt_id] = float(row["composite_mean"])
    require(
        len(ensemble) == EXPECTED_PANEL_PROMPTS,
        f"Expected {EXPECTED_PANEL_PROMPTS} original ensemble scores, "
        f"found {len(ensemble)}",
    )

    panel_ids = set(ensemble)
    panel_rows: list[dict] = []
    for model in MATCHED_MODELS:
        path = panel_dir / f"{model}.jsonl"
        seen: set[str] = set()
        for row in read_jsonl(path):
            prompt_id = row.get("id")
            if prompt_id not in panel_ids or row.get("pass_rate") is None:
                continue
            require(
                prompt_id not in seen,
                f"Duplicate successful original row for {model}/{prompt_id}",
            )
            seen.add(prompt_id)
            panel_rows.append(
                {
                    "prompt_id": prompt_id,
                    "model": model,
                    "source": "Original benchmark",
                    "composite": ensemble[prompt_id],
                    "pass_rate": float(row["pass_rate"]),
                }
            )
        require(
            seen == panel_ids,
            f"Original model {model} covers {len(seen)} of "
            f"{EXPECTED_PANEL_PROMPTS} prompts",
        )

    generation_frame = pd.DataFrame(extension_rows + panel_rows)
    require(
        len(generation_frame)
        == len(MATCHED_MODELS)
        * (EXPECTED_PANEL_PROMPTS + EXPECTED_EXTENSION_PROMPTS),
        "Matched generation frame has an unexpected row count",
    )
    prompt_frame = (
        generation_frame.groupby(["prompt_id", "source"], as_index=False)
        .agg(
            composite=("composite", "first"),
            pass_rate=("pass_rate", "mean"),
            model_count=("model", "nunique"),
        )
        .sort_values(["source", "prompt_id"])
        .reset_index(drop=True)
    )
    require(
        bool((prompt_frame["model_count"] == len(MATCHED_MODELS)).all()),
        "Every prompt must have exactly five matched-model outcomes",
    )
    prompt_frame["bin"] = half_open_integer_bin(prompt_frame["composite"])
    return prompt_frame


def summarize(prompt_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    curve = (
        prompt_frame.groupby("bin", as_index=False)
        .agg(
            n=("pass_rate", "size"),
            mean_pass=("pass_rate", "mean"),
            sem=("pass_rate", "sem"),
        )
        .sort_values("bin")
        .reset_index(drop=True)
    )

    split_parts = []
    for source, low_bin, high_bin in (
        ("Audit-clean extension", 15, 18),
        ("Original benchmark", 9, 18),
    ):
        part = prompt_frame[
            (prompt_frame["source"] == source)
            & prompt_frame["bin"].between(low_bin, high_bin)
        ]
        part = (
            part.groupby("bin", as_index=False)
            .agg(
                n=("pass_rate", "size"),
                mean_pass=("pass_rate", "mean"),
                sem=("pass_rate", "sem"),
            )
            .sort_values("bin")
        )
        part.insert(0, "source", source)
        split_parts.append(part)
    source_split = pd.concat(split_parts, ignore_index=True)

    replication_rows = []
    for bin_id in (15, 16):
        original = prompt_frame.loc[
            (prompt_frame["source"] == "Original benchmark")
            & (prompt_frame["bin"] == bin_id),
            "pass_rate",
        ]
        extension = prompt_frame.loc[
            (prompt_frame["source"] == "Audit-clean extension")
            & (prompt_frame["bin"] == bin_id),
            "pass_rate",
        ]
        _, p_value = stats.ttest_ind(original, extension, equal_var=False)
        for source, values in (
            ("Original benchmark", original),
            ("Audit-clean extension", extension),
        ):
            replication_rows.append(
                {
                    "source": source,
                    "bin": bin_id,
                    "n": len(values),
                    "mean_pass": values.mean(),
                    "sem": values.sem(),
                    "welch_p": float(p_value),
                }
            )
    replication = pd.DataFrame(replication_rows)

    return {
        "tail_extension_curve.csv": curve,
        "tail_extension_source_split.csv": source_split,
        "tail_extension_replication.csv": replication,
    }


def load_fixed_version_frame(
    data_root: Path, extension_meta: pd.DataFrame
) -> pd.DataFrame:
    scored_dir = data_root / "data/rebuttal/frontier_tail/scored"
    extension_ids = set(extension_meta["prompt_id"])
    successful: dict[tuple[str, str], dict] = {}

    for path in sorted(scored_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            model = row.get("model_id")
            if model not in dict(FRONTIER_MODELS):
                continue
            if row.get("group") == "tail" and row["prompt_id"] not in extension_ids:
                continue
            if row.get("pass_rate") is None:
                continue
            key = (model, row["prompt_id"])
            require(
                key not in successful,
                f"Duplicate successful fixed-version row for {model}/{row['prompt_id']}",
            )
            successful[key] = row

    expected_prompts = EXPECTED_FRONTIER_ANCHORS + EXPECTED_EXTENSION_PROMPTS
    prompt_sets = []
    for model, _ in FRONTIER_MODELS:
        model_rows = {
            prompt_id: row
            for (row_model, prompt_id), row in successful.items()
            if row_model == model
        }
        require(
            len(model_rows) == expected_prompts,
            f"Fixed-version model {model} covers {len(model_rows)} of "
            f"{expected_prompts} prompts",
        )
        tail_count = sum(row.get("group") == "tail" for row in model_rows.values())
        require(
            tail_count == EXPECTED_EXTENSION_PROMPTS,
            f"Fixed-version model {model} has {tail_count} retained tail prompts",
        )
        prompt_sets.append(set(model_rows))
    require(
        all(prompt_set == prompt_sets[0] for prompt_set in prompt_sets[1:]),
        "Fixed-version models do not cover the same prompt IDs",
    )

    rows = []
    for model, display_name in FRONTIER_MODELS:
        for (row_model, prompt_id), row in successful.items():
            if row_model != model:
                continue
            rows.append(
                {
                    "model": display_name,
                    "prompt_id": prompt_id,
                    "composite": float(row["rubric_composite"]),
                    "pass_rate": float(row["pass_rate"]),
                }
            )
    frame = pd.DataFrame(rows)
    frame["bin"] = half_open_integer_bin(frame["composite"])

    output_rows = []
    for _, display_name in FRONTIER_MODELS:
        for bin_id in (15, 16):
            values = frame.loc[
                (frame["model"] == display_name) & (frame["bin"] == bin_id),
                "pass_rate",
            ]
            expected_n = 228 if bin_id == 15 else 133
            require(
                len(values) == expected_n,
                f"Unexpected fixed-version support for {display_name}, "
                f"bin {bin_id}: {len(values)}",
            )
            output_rows.append(
                {
                    "model": display_name,
                    "bin": bin_id,
                    "n": len(values),
                    "mean_pass": values.mean(),
                }
            )
    return pd.DataFrame(output_rows)


def check_table(path: Path, expected: pd.DataFrame) -> None:
    require(path.exists(), f"Missing expected output: {path}")
    actual = pd.read_csv(path)
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise RuntimeError(f"Regenerated table differs from {path}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT,
        help="Repository or retained-data root containing data/ (default: repo root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results",
        help="Destination for regenerated CSVs (default: results/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare regenerated tables with existing CSVs without writing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extension_meta = load_extension_metadata(args.data_root)
    prompt_frame = load_matched_frame(args.data_root, extension_meta)
    tables = summarize(prompt_frame)
    tables["tail_extension_fixed_version.csv"] = load_fixed_version_frame(
        args.data_root, extension_meta
    )

    if args.check:
        for filename, table in tables.items():
            check_table(args.output_dir / filename, table)
        print(
            "Verified four tail-extension tables from 365 retained prompts "
            "and five matched models."
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        path = args.output_dir / filename
        table.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
