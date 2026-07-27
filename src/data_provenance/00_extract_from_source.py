"""
Step 0: Re-extract samples from OpenCodeInstruct using NVIDIA's own test results.

This script REPLACES the original 4-step pipeline (01-04) by:
  1. Downloading parquet shards directly from HuggingFace
  2. Using the dataset's NATIVE `tests_execution_status` and `average_test_score`
     fields instead of re-running tests locally
  3. Using the dataset's NATIVE `lang` field (if available) or falling back to
     a stricter language detection heuristic
  4. Writing a single output JSONL with clean provenance

WHY THIS EXISTS:
  The original pipeline (04_score_tests.py) only executed Python and JavaScript
  tests.  Go, Java, and C++ results were FABRICATED:
      result = "pass" if kappa_cyclomatic < 5 else "fail"
  This made ~44% of pass/fail outcomes a direct function of the complexity
  variable the paper studies ,  invalidating any analysis on those languages.

  OpenCodeInstruct already ran all tests across all languages.  We use their
  ground-truth results.

OUTPUT SCHEMA (one JSON object per line):
  {
    "id":                       str,   # original dataset ID
    "lang":                     str,   # programming language
    "input":                    str,   # natural-language instruction
    "output":                   str,   # model-generated code
    "unit_tests":               str,   # unit test code (JSON-encoded list)
    "tests_execution_status":   list,  # per-test ["pass","fail",...] from NVIDIA
    "average_test_score":       float, # aggregate score from NVIDIA
    "status":                   list,  # same as tests_execution_status (compat)
    "pass_rate":                float  # computed from tests_execution_status
  }

USAGE:
  pip install huggingface_hub pyarrow pandas
  python src/data_provenance/00_extract_from_source.py

REPRODUCIBILITY:
  - Shards are processed in deterministic order (0..49)
  - Per-language cap uses random_state=42
  - All filtering criteria are explicit below
"""
import os
import json
import ast
import io
import argparse
import pandas as pd
import pyarrow.parquet as pq

try:
    from huggingface_hub import hf_hub_url
    import requests
    HAS_HF = True
except ImportError:
    HAS_HF = False

# -- Configuration --------------------------------------------------------
DATASET_NAME = "nvidia/OpenCodeInstruct"
NUM_SHARDS = 50
LANGUAGES = ['python', 'java', 'javascript', 'cpp', 'go']
TARGET_PER_LANG = None         # None = no cap, use all valid samples
RANDOM_STATE = 42              # for reproducible operations
MIN_CODE_LENGTH = 20           # minimum characters in generated code
MIN_TEST_COUNT = 1             # minimum number of test assertions

def clean_code_from_markdown(code):
    """Strip markdown code fences if present, returning raw code."""
    import re
    if not code or not isinstance(code, str):
        return ''
    code = code.strip()
    # Match ```lang\n...\n```
    match = re.match(r'^```\w*\n(.*?)```\s*$', code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code


OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_results_scored.jsonl")


def compute_pass_rate(status_list):
    """Compute pass rate from a list of per-test outcomes."""
    if not status_list or not isinstance(status_list, list):
        return 0.0
    n_pass = sum(1 for s in status_list if isinstance(s, str) and 'pass' in s.lower())
    return n_pass / len(status_list) if len(status_list) > 0 else 0.0


def parse_status_field(raw):
    """Parse tests_execution_status from various formats."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        # Try JSON
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError):
            pass
    return []


def detect_language(row):
    """
    Determine programming language for a record.

    Preference order:
      1. Dataset's own 'language' or 'lang' field (if present and in our list)
      2. Markdown code fence language tag (```python ... ```)
      3. Heuristic detection from code content
    """
    # Check native field first
    for field in ['language', 'lang']:
        native = row.get(field, '')
        if isinstance(native, str):
            native_lower = native.strip().lower()
            lang_map = {
                'python': 'python', 'py': 'python', 'python3': 'python',
                'javascript': 'javascript', 'js': 'javascript',
                'java': 'java',
                'cpp': 'cpp', 'c++': 'cpp',
                'go': 'go', 'golang': 'go',
            }
            if native_lower in lang_map:
                return lang_map[native_lower]

    # Check markdown code fence tag
    import re
    code = str(row.get('output', ''))
    fence_match = re.match(r'```(\w+)', code.strip())
    if fence_match:
        tag = fence_match.group(1).lower()
        fence_map = {
            'python': 'python', 'py': 'python', 'python3': 'python',
            'javascript': 'javascript', 'js': 'javascript', 'typescript': 'javascript',
            'java': 'java',
            'cpp': 'cpp', 'c++': 'cpp', 'c': 'cpp',
            'go': 'go', 'golang': 'go',
        }
        if tag in fence_map:
            return fence_map[tag]

    # Fallback: heuristic from code (stricter patterns)
    code_lower = code.lower()

    if 'package main' in code_lower and 'func ' in code_lower:
        return 'go'
    if '#include' in code_lower and ('std::' in code_lower or 'cout' in code_lower or 'cin' in code_lower):
        return 'cpp'
    if 'public class' in code_lower and 'public static' in code_lower:
        return 'java'
    if 'def ' in code_lower and ('import ' in code_lower or 'print(' in code_lower or ':' in code_lower):
        return 'python'
    if ('function ' in code_lower or '=>' in code_lower) and ('{' in code_lower):
        return 'javascript'

    return None


def process_shard(shard_idx, counts, existing_ids):
    """Download and process a single parquet shard. Returns list of valid records."""
    if not HAS_HF:
        raise ImportError("huggingface_hub and requests are required. pip install huggingface_hub requests")

    file_path = f"data/train-{shard_idx:05d}-of-{NUM_SHARDS:05d}.parquet"
    url = hf_hub_url(repo_id=DATASET_NAME, filename=file_path, repo_type="dataset")

    resp = requests.get(url, timeout=120)
    if resp.status_code != 200:
        print(f"  Shard {shard_idx}: HTTP {resp.status_code}, skipping")
        return []

    table = pq.read_table(io.BytesIO(resp.content))
    df = table.to_pandas()

    records = []
    for _, entry in df.iterrows():
        row = entry.to_dict()

        # --- Language detection ---
        lang = detect_language(row)
        if lang is None or lang not in LANGUAGES:
            continue

        # --- Skip if language already at cap (if cap is set) ---
        if TARGET_PER_LANG is not None and counts.get(lang, 0) >= TARGET_PER_LANG:
            continue

        # --- Deduplicate ---
        record_id = row.get('id', '')
        if record_id in existing_ids:
            continue

        # --- Extract fields ---
        input_text = str(row.get('input', ''))
        output_text = str(row.get('output', ''))
        code_cleaned = clean_code_from_markdown(output_text)
        unit_tests = row.get('unit_tests', '')

        # --- Basic quality filters ---
        if len(code_cleaned) < MIN_CODE_LENGTH:
            continue
        if not unit_tests or len(str(unit_tests)) < 10:
            continue

        # --- Parse NVIDIA's test results ---
        status_raw = row.get('tests_execution_status', [])
        status_list = parse_status_field(status_raw)
        if len(status_list) < MIN_TEST_COUNT:
            continue

        avg_score_raw = row.get('average_test_score', None)
        try:
            avg_score = float(avg_score_raw) if avg_score_raw is not None else None
        except (ValueError, TypeError):
            avg_score = None

        pass_rate = compute_pass_rate(status_list)

        record = {
            'id': record_id,
            'lang': lang,
            'input': input_text,
            'output': output_text,
            'code_cleaned': code_cleaned,
            'unit_tests': unit_tests if isinstance(unit_tests, str) else json.dumps(unit_tests),
            'tests_execution_status': status_list,
            'average_test_score': avg_score,
            'status': status_list,
            'pass_rate': pass_rate,
        }

        records.append(record)
        existing_ids.add(record_id)
        counts[lang] = counts.get(lang, 0) + 1

    return records


def main():
    parser = argparse.ArgumentParser(description="Extract samples from OpenCodeInstruct with NVIDIA test results")
    parser.add_argument('--output', default=OUTPUT_FILE, help='Output JSONL path')
    parser.add_argument('--target-per-lang', type=int, default=None,
                        help='Cap per language (default: None = no cap, use all)')
    parser.add_argument('--dry-run', action='store_true', help='Print config and exit')
    args = parser.parse_args()

    if args.target_per_lang is not None:
        global TARGET_PER_LANG
        TARGET_PER_LANG = args.target_per_lang

    print("=" * 60)
    print("OpenCodeInstruct Extraction (with native test results)")
    print("=" * 60)
    print(f"  Dataset:        {DATASET_NAME}")
    print(f"  Shards:         {NUM_SHARDS}")
    print(f"  Languages:      {LANGUAGES}")
    print(f"  Target/lang:    {TARGET_PER_LANG or 'No cap (all valid samples)'}")
    print(f"  Output:         {args.output}")
    print(f"  Random state:   {RANDOM_STATE}")
    print()

    if args.dry_run:
        print("Dry run ,  exiting.")
        return

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    all_records = []
    existing_ids = set()
    counts = {lang: 0 for lang in LANGUAGES}

    # Process shards in DETERMINISTIC order (0, 1, 2, ..., 49)
    for shard_idx in range(NUM_SHARDS):
        print(f"Processing shard {shard_idx}/{NUM_SHARDS-1}...")
        try:
            shard_records = process_shard(shard_idx, counts, existing_ids)
            all_records.extend(shard_records)
            print(f"  +{len(shard_records)} records. Totals: {counts}")
        except Exception as e:
            print(f"  Shard {shard_idx} failed: {e}")

        # Early exit if all languages at cap (only if cap is set)
        if TARGET_PER_LANG is not None and all(counts.get(lang, 0) >= TARGET_PER_LANG for lang in LANGUAGES):
            print("All languages at cap ,  stopping early.")
            break

    # --- Summary before write ---
    print(f"\nTotal records collected: {len(all_records)}")
    for lang in LANGUAGES:
        n = sum(1 for r in all_records if r['lang'] == lang)
        n_perfect = sum(1 for r in all_records if r['lang'] == lang and r['pass_rate'] == 1.0)
        print(f"  {lang:12s}: {n:6d} total, {n_perfect:6d} perfect pass")

    # --- Write output ---
    with open(args.output, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record) + '\n')

    print(f"\nWrote {len(all_records)} records to {args.output}")


if __name__ == "__main__":
    main()
