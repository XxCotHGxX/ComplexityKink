"""
Step 1: Select a stratified prompt set for multi-model generation.

Takes the full OpenCodeInstruct extraction (final_results_scored.jsonl),
computes cyclomatic complexity on the Qwen2.5 solutions, then selects a
stratified sample of prompts that spans the full CC range.

This ensures:
  - The SAME prompts are sent to every model
  - Every complexity level is well-represented (not dominated by CC=1)
  - Unit tests from NVIDIA are carried along for later scoring

OUTPUT:
  data/experiment_prompts.jsonl  ,  one record per selected prompt:
    {
      "prompt_id":    str,    # unique prompt identifier
      "input":        str,    # the coding instruction
      "unit_tests":   str,    # NVIDIA's unit tests (JSON-encoded list)
      "reference_cc": int|null, # CC of the Qwen2.5 solution (for stratification)
      "lang":         str,    # always "python" for OpenCodeInstruct
    }

USAGE:
  python src/data_provenance/01_select_prompts.py [--n-prompts 5000] [--seed 42]
"""
import os
import sys
import json
import argparse
import random

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lizard


def compute_cc_quick(code):
    """Compute cyclomatic complexity of a Python code snippet using lizard."""
    if not code or not isinstance(code, str) or len(code.strip()) < 10:
        return None
    try:
        analysis = lizard.analyze_file.analyze_source_code("snippet.py", code)
        if not analysis.function_list:
            return None
        total_cc = sum(f.cyclomatic_complexity for f in analysis.function_list)
        return total_cc if total_cc > 0 else None
    except Exception:
        return None


def clean_code_from_markdown(text):
    """Strip markdown code fences from model output (takes LAST block)."""
    import re
    pattern = r'```(?:\w+)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def is_trivial_test(tests_str):
    """Detect broken tests where every assertion is `assert X == None`."""
    if not tests_str or tests_str.strip() in ('', '[]'):
        return True
    if '== None' in tests_str and tests_str.count('assert') == tests_str.count('== None'):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Select stratified prompt set")
    parser.add_argument("--input", default=os.path.join("data", "final_results_scored.jsonl"),
                        help="Path to full extraction JSONL")
    parser.add_argument("--output", default=os.path.join("data", "experiment_prompts.jsonl"),
                        help="Path to output prompt set JSONL")
    parser.add_argument("--n-prompts", type=int, default=5000,
                        help="Total number of prompts to select")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    # Default is 0 (scan the full pool). OpenCodeInstruct shards have
    # ordering structure by seed corpus and generator, so a partial scan can
    # silently bias the language mix and difficulty distribution. The
    # original 5,000-prompt draw used --scan-limit 200000; that decision is
    # documented in docs/reproduction_guide.md and validated post-hoc.
    parser.add_argument("--scan-limit", type=int, default=0,
                        help="Records to scan (0 = full pool, recommended).")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"=== Prompt Selection ===")
    print(f"  Input:     {args.input}")
    print(f"  Output:    {args.output}")
    print(f"  N prompts: {args.n_prompts}")
    scan_msg = "full pool" if args.scan_limit == 0 else f"first {args.scan_limit:,} records"
    print(f"  Scan:      {scan_msg}")
    print(f"  Seed:      {args.seed}")
    print()

    # Phase 1: Scan records, compute CC, bucket by complexity
    # CC bins: [1], [2], [3], [4], [5], [6-7], [8-10], [11-15], [16+], [None]
    bins = {
        '1': [], '2': [], '3': [], '4': [], '5': [],
        '6-7': [], '8-10': [], '11-15': [], '16+': [], 'none': []
    }

    def get_bin(cc):
        if cc is None:
            return 'none'
        if cc <= 5:
            return str(cc)
        if cc <= 7:
            return '6-7'
        if cc <= 10:
            return '8-10'
        if cc <= 15:
            return '11-15'
        return '16+'

    print("Scanning records and computing CC...")
    scanned = 0
    with_tests = 0

    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if args.scan_limit and scanned >= args.scan_limit:
                break
            scanned += 1

            rec = json.loads(line)

            # Must have unit tests for scoring
            tests = rec.get('unit_tests', '')
            if not tests or tests == '[]':
                continue

            # QUALITY FILTER: reject trivial tests (e.g. all `assert X == None`)
            if is_trivial_test(tests):
                continue

            # QUALITY FILTER: reference solution must pass its own tests
            exec_status = rec.get('tests_execution_status', [])
            if exec_status and any(s != 'pass' for s in exec_status):
                continue

            with_tests += 1

            # Must have non-trivial input
            prompt = rec.get('input', '')
            if not prompt or len(prompt.strip()) < 20:
                continue

            # Compute CC on the Qwen2.5 solution
            code = rec.get('code_cleaned', '')
            if not code:
                code = clean_code_from_markdown(rec.get('output', ''))
            cc = compute_cc_quick(code)

            b = get_bin(cc)
            bins[b].append({
                'prompt_id': rec.get('id', f'scan_{scanned}'),
                'input': prompt,
                'unit_tests': tests,
                'reference_cc': cc,
                'lang': rec.get('lang', 'python'),
            })

            if scanned % 50000 == 0:
                total_in_bins = sum(len(v) for v in bins.values())
                print(f"  Scanned {scanned:,}, candidates: {total_in_bins:,}")

    # Phase 2: Stratified selection
    print(f"\nScan complete. {scanned:,} scanned, {with_tests:,} have tests.")
    print("Bin distribution:")
    for k, v in bins.items():
        print(f"  CC {k:>5}: {len(v):,}")

    # Exclude 'none' bin ,  we want prompts where CC is measurable
    named_bins = {k: v for k, v in bins.items() if k != 'none'}
    total_available = sum(len(v) for v in named_bins.values())
    print(f"\nTotal with measurable CC: {total_available:,}")

    # Allocate: equal share per bin, remainder to largest bins
    n_bins = len(named_bins)
    per_bin = args.n_prompts // n_bins
    remainder = args.n_prompts % n_bins

    selected = []
    for i, (k, v) in enumerate(sorted(named_bins.items(), key=lambda x: len(x[1]), reverse=True)):
        alloc = per_bin + (1 if i < remainder else 0)
        take = min(alloc, len(v))
        chosen = random.sample(v, take)
        selected.extend(chosen)
        print(f"  CC {k:>5}: allocated {alloc}, took {take}")

    random.shuffle(selected)
    print(f"\nTotal selected: {len(selected)}")

    # Phase 3: Write output
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for rec in selected:
            f.write(json.dumps(rec) + '\n')

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
