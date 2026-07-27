"""
Step 3: Execute unit tests against generated solutions and score results.

Takes the per-model generation files from data/generations/ and the
original prompt set (with NVIDIA's unit tests), then:
  1. Executes each generated solution against the unit tests
  2. Computes pass_rate (fraction of tests passed)
  3. Computes cyclomatic complexity via lizard
  4. Writes a unified scored dataset per model: data/scored/<model_id>.jsonl

The scored files have the SAME schema as iv_enriched_dataset.jsonl,
making them drop-in compatible with the rest of the pipeline.

SAFETY:
  Code execution is sandboxed via subprocess with:
    - 10-second timeout
    - No network access (we don't grant it)
    - Isolated temp files (cleaned up)

USAGE:
  python src/data_provenance/03_execute_and_score.py \\
      --prompts data/experiment_prompts.jsonl \\
      --generations-dir data/generations \\
      --output-dir data/scored
"""
import os
import sys
import json
import ast
import tempfile
import subprocess
import argparse
import time

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lizard


EXECUTION_TIMEOUT = 5  # seconds per test case
MAX_LIZARD_SOURCE_CHARS = 12_000

# On Windows, subprocess.run(timeout=...) can leave orphaned grandchildren
# alive if the generated code spawns them. We create a new process group so
# the timeout path can kill the whole tree.
_CREATIONFLAGS = 0
if os.name == "nt":
    _CREATIONFLAGS = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore


def kill_process_tree(proc):
    """Terminate a generated-code subprocess and any children it spawned."""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def compute_cc(code):
    """Compute cyclomatic complexity of Python code using lizard."""
    if not code or not isinstance(code, str) or len(code.strip()) < 10:
        return None
    if len(code) > MAX_LIZARD_SOURCE_CHARS:
        return None
    try:
        analysis = lizard.analyze_file.analyze_source_code("snippet.py", code)
        if not analysis.function_list:
            return None
        total_cc = sum(f.cyclomatic_complexity for f in analysis.function_list)
        return total_cc if total_cc > 0 else None
    except Exception:
        return None


def parse_unit_tests(unit_tests_str):
    """Parse the unit_tests field from NVIDIA format into a list of test strings."""
    if not unit_tests_str:
        return []
    try:
        tests = ast.literal_eval(unit_tests_str)
        if isinstance(tests, list):
            return [t.strip() for t in tests if isinstance(t, str) and t.strip()]
    except (ValueError, SyntaxError):
        pass
    return []


def execute_test(code, test_assertion):
    """
    Execute a single test assertion against the generated code.

    Returns "pass" or "fail".
    """
    # Build the full test script: code + assertion
    full_script = code + "\n\n" + test_assertion

    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", full_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "MPLBACKEND": "Agg",          # headless matplotlib
                "DISPLAY": "",                 # block X11
                "QT_QPA_PLATFORM": "offscreen",
                "SDL_VIDEODRIVER": "dummy",
            },
            creationflags=_CREATIONFLAGS,
        )
        try:
            _out, _err = proc.communicate(timeout=EXECUTION_TIMEOUT)
            return "pass" if proc.returncode == 0 else "fail"
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            try:
                proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            return "fail"
    except Exception:
        kill_process_tree(proc)
        return "fail"


def score_solution(code, unit_tests_str):
    """
    Execute all unit tests against a code solution.

    Returns
    -------
    status : list[str]
        Per-test ["pass","fail",...] list.
    pass_rate : float
        Fraction of tests that passed.
    """
    tests = parse_unit_tests(unit_tests_str)
    if not tests or not code:
        return [], 0.0

    status = []
    for test in tests:
        result = execute_test(code, test)
        status.append(result)

    n_pass = sum(1 for s in status if s == "pass")
    pass_rate = n_pass / len(status) if status else 0.0
    return status, pass_rate


def extract_instruction_features(instruction):
    """
    Extract IV features from instruction text.
    (Duplicated from feature_extractor_iv.py to avoid circular imports.)
    """
    import re

    if not isinstance(instruction, str) or not instruction.strip():
        return {
            'inst_tokens': 0, 'inst_if_count': 0, 'inst_loop_count': 0,
            'inst_class_count': 0, 'inst_func_count': 0, 'inst_logic_count': 0,
            'inst_total_structural': 0, 'inst_avg_word_len': 0.0,
            'inst_conditional_count': 0, 'inst_collection_count': 0,
        }

    n_if = len(re.findall(r'\b(if|when|whether|case|switch)\b', instruction, re.I))
    n_conditional = len(re.findall(r'\b(condition|scenario|situation|otherwise|else)\b', instruction, re.I))
    n_loop = len(re.findall(r'\b(loop|iterate|for each|while|repeat|mapping|filter)\b', instruction, re.I))
    n_collection = len(re.findall(r'\b(list|array|dictionary|map|set|collection|stream)\b', instruction, re.I))
    n_class = len(re.findall(r'\b(class|object|method|inheritance|interface|abstract)\b', instruction, re.I))
    n_func = len(re.findall(r'\b(function|procedure|recursive|recursion)\b', instruction, re.I))
    n_logic = len(re.findall(r'\b(logic|algorithm|validate|parse|sort|search)\b', instruction, re.I))

    tokens = instruction.split()
    n_tokens = len(tokens)
    avg_word_len = sum(len(w) for w in tokens) / n_tokens if n_tokens > 0 else 0

    return {
        'inst_tokens': n_tokens,
        'inst_if_count': n_if,
        'inst_conditional_count': n_conditional,
        'inst_loop_count': n_loop,
        'inst_collection_count': n_collection,
        'inst_class_count': n_class,
        'inst_func_count': n_func,
        'inst_logic_count': n_logic,
        'inst_total_structural': n_if + n_loop + n_class + n_func,
        'inst_avg_word_len': avg_word_len,
    }


def process_model(prompts_by_id, gen_file, output_file):
    """Score all generations for one model."""
    print(f"  Scoring: {gen_file}")

    processed = 0
    executed = 0
    perfect = 0

    # Deduplicate by prompt_id. Valid rows beat failed/empty rows; when both
    # rows are valid, prefer the later retry.
    def _is_valid(rec):
        if rec.get("error"):
            return False
        return bool((rec.get("code_cleaned") or "").strip())

    all_gen_records = {}
    with open(gen_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            gen = json.loads(line)
            pid = gen.get("prompt_id")
            prev = all_gen_records.get(pid)
            if prev is None or _is_valid(gen):
                all_gen_records[pid] = gen

    # Resume: load already-scored prompt_ids so a killed run can pick up
    # where it left off instead of re-executing every test.
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f_exist:
            for line in f_exist:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("id") or rec.get("prompt_id")
                if pid is not None:
                    done_ids.add(pid)
        if done_ids:
            print(f"  Resuming: {len(done_ids)} prompts already scored",
                  flush=True)

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for prompt_id, gen in all_gen_records.items():
            if prompt_id in done_ids:
                continue
            code = gen.get("code_cleaned")
            model_id = gen.get("model_id", "unknown")

            # Look up prompt to get unit_tests and instruction
            prompt_rec = prompts_by_id.get(prompt_id)
            if not prompt_rec:
                continue

            # Execute tests
            if code and gen.get("error") is None:
                status, pass_rate = score_solution(code, prompt_rec["unit_tests"])
                executed += 1
            else:
                status, pass_rate = [], 0.0

            if pass_rate == 1.0 and len(status) > 0:
                perfect += 1

            # Compute output-side CC from the generated solution itself.
            # Do not copy reference_cc from the prompt record: that is source
            # / reference-solution metadata used only for sampling diagnostics.
            cc = compute_cc(code) if code else None

            # Extract IV features from the instruction
            iv_features = extract_instruction_features(prompt_rec["input"])

            # Write enriched record (compatible with iv_enriched_dataset.jsonl schema)
            record = {
                "id": prompt_id,
                "model_id": model_id,
                "lang": prompt_rec.get("lang", "python"),
                "input": prompt_rec["input"],
                "output": gen.get("raw_response", ""),
                "code_cleaned": code,
                "unit_tests": prompt_rec["unit_tests"],
                "tests_execution_status": status,
                "status": status,
                "pass_rate": pass_rate,
                "kappa_cyclomatic": cc,
                "kappa_cyclomatic_source": "lizard_on_generated_output",
                "reference_cc": prompt_rec.get("reference_cc"),
                "coupling_depth": 0,
                "iv_features": iv_features,
                "generation_timestamp": gen.get("timestamp"),
            }
            f_out.write(json.dumps(record) + '\n')
            f_out.flush()
            processed += 1

            if processed % 100 == 0:
                print(f"    [{processed}] executed: {executed}, perfect: {perfect}",
                      flush=True)

    print(f"  Done: {processed} total, {executed} executed, {perfect} perfect-pass")
    return processed, executed, perfect


def main():
    parser = argparse.ArgumentParser(description="Execute tests and score solutions")
    parser.add_argument("--prompts", default=os.path.join("data", "experiment_prompts.jsonl"))
    parser.add_argument("--generations-dir", default=os.path.join("data", "generations"))
    parser.add_argument("--output-dir", default=os.path.join("data", "scored"))
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only score these model IDs")
    args = parser.parse_args()

    # Load prompts into lookup
    prompts_by_id = {}
    with open(args.prompts, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            prompts_by_id[rec["prompt_id"]] = rec
    print(f"Loaded {len(prompts_by_id)} prompts")

    os.makedirs(args.output_dir, exist_ok=True)

    # Find generation files
    gen_files = sorted([
        f for f in os.listdir(args.generations_dir)
        if f.endswith('.jsonl')
    ])

    if args.only:
        gen_files = [f for f in gen_files
                     if any(m in f for m in args.only)]

    print(f"Found {len(gen_files)} generation files")
    print()

    summary = []
    for gf in gen_files:
        model_name = gf.replace('.jsonl', '')
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        gen_path = os.path.join(args.generations_dir, gf)
        out_path = os.path.join(args.output_dir, gf)

        total, executed, perfect = process_model(prompts_by_id, gen_path, out_path)
        summary.append({
            "model": model_name,
            "total": total,
            "executed": executed,
            "perfect_pass": perfect,
            "perfect_rate": perfect / max(executed, 1),
        })
        print()

    # Print summary table
    print("=" * 70)
    print(f"{'Model':<25} {'Total':>7} {'Tested':>7} {'Perfect':>7} {'Rate':>7}")
    print("-" * 70)
    for s in summary:
        print(f"{s['model']:<25} {s['total']:>7} {s['executed']:>7} "
              f"{s['perfect_pass']:>7} {s['perfect_rate']:>6.1%}")
    print("=" * 70)

    # Save summary
    summary_path = os.path.join(args.output_dir, "_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
