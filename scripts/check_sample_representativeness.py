"""
Representativeness check for the locked 5,000-prompt experimental set.

The experimental sample was drawn with --scan-limit 200000 (a prefix scan
over source-ordered shards); see docs/reproduction_guide.md. This script
draws an independent full-pool reference sample and tests whether the two
draws differ on the sampling-frame axes that could bias downstream analysis.

Usage:
    python scripts/check_sample_representativeness.py \
        --extracted data/oci_extracted.jsonl \
        --experiment data/experiment_prompts.jsonl \
        --n-ref 5000 \
        --seed 2026

No model is queried; this only characterizes the sampling frame.
"""
import argparse
import json
import os
import sys
import random
import subprocess
import tempfile
from collections import Counter

import numpy as np
from scipy import stats


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def draw_reference_sample(extracted_path, n_ref, seed):
    """Draw a full-pool stratified reference sample via 01_select_prompts.py.

    We shell out rather than re-implement stratification so the reference
    draw uses the exact same bucketing logic as the experimental draw.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    selector = os.path.normpath(os.path.join(here, "..", "src", "data_provenance",
                                             "01_select_prompts.py"))
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name
    cmd = [
        sys.executable, selector,
        "--input", extracted_path,
        "--output", out_path,
        "--n-prompts", str(n_ref),
        "--seed", str(seed),
        "--scan-limit", "0",  # full pool
    ]
    print(f"Drawing reference sample: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path


def dist_tokens(records):
    return np.array([
        len((r.get("instruction") or r.get("prompt") or "").split())
        for r in records
    ])


def dist_reference_cc(records):
    vals = [r.get("reference_cc") or r.get("cc") for r in records]
    return np.array([v for v in vals if v is not None], dtype=float)


def dist_test_count(records):
    out = []
    for r in records:
        tests = r.get("tests") or r.get("unit_tests") or []
        if isinstance(tests, list):
            out.append(len(tests))
    return np.array(out, dtype=float)


def dist_seed_lang(records):
    return Counter(r.get("seed_lang") or r.get("source_lang") or "unknown"
                   for r in records)


def ks_report(name, a, b):
    if len(a) == 0 or len(b) == 0:
        print(f"  {name:22s}  insufficient data")
        return
    ks = stats.ks_2samp(a, b)
    print(f"  {name:22s}  n_exp={len(a):>5}  n_ref={len(b):>5}  "
          f"KS D={ks.statistic:.4f}  p={ks.pvalue:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extracted", required=True,
                    help="Full extracted OpenCodeInstruct JSONL (pre-selection)")
    ap.add_argument("--experiment", default="data/experiment_prompts.jsonl")
    ap.add_argument("--n-ref", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026,
                    help="Different from experimental seed (42) by design")
    args = ap.parse_args()

    exp = load_jsonl(args.experiment)
    ref_path = draw_reference_sample(args.extracted, args.n_ref, args.seed)
    ref = load_jsonl(ref_path)

    print(f"\nExperiment set: {len(exp)}  Reference set: {len(ref)}\n")
    print("Kolmogorov-Smirnov two-sample tests (reject H0 at p < 0.05):\n")
    ks_report("instruction tokens",   dist_tokens(exp),        dist_tokens(ref))
    ks_report("reference CC",         dist_reference_cc(exp),  dist_reference_cc(ref))
    ks_report("test count per prompt", dist_test_count(exp),   dist_test_count(ref))

    print("\nSeed-language distribution (Chi-squared):")
    exp_l = dist_seed_lang(exp)
    ref_l = dist_seed_lang(ref)
    keys = sorted(set(exp_l) | set(ref_l))
    exp_v = np.array([exp_l.get(k, 0) for k in keys])
    ref_v = np.array([ref_l.get(k, 0) for k in keys])
    print("  languages: " + ", ".join(f"{k}" for k in keys))
    print(f"  exp counts: {exp_v.tolist()}")
    print(f"  ref counts: {ref_v.tolist()}")
    if exp_v.sum() > 0 and ref_v.sum() > 0:
        chi2, p, dof, _ = stats.chi2_contingency(np.array([exp_v, ref_v]))
        print(f"  chi2={chi2:.2f}  dof={dof}  p={p:.4f}")

    print("\nInterpretation: p > 0.05 on every axis means the scan-limit "
          "prefix draw is statistically indistinguishable from a full-pool "
          "draw on the sampling frame; no re-sampling required.\n")

    os.unlink(ref_path)


if __name__ == "__main__":
    main()
