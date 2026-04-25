"""
Step 5: Score prompt complexity via LLM rubric (o4-mini).

Reads experiment_prompts.jsonl and scores each prompt's structural complexity
across 6 dimensions using a rigid rubric. The scoring model (o4-mini) is NOT
in the study, avoiding circularity concerns.

Output: data/complexity_rubric_scores.jsonl
Each line: {"prompt_id": ..., "scores": {"branching": 0-4, ...}, "composite": 0-24}

USAGE:
  python src/data_provenance/05_score_complexity_rubric.py \
      --prompts data/experiment_prompts.jsonl \
      --output data/complexity_rubric_scores.jsonl \
      --workers 5
"""

import os
import sys
import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
# Optional author-side shim for CLI-tool auth stores (see 02_generate_solutions.py).
# Public users set AZURE_OPENAI_API_KEY in the environment directly.
try:
    import load_keys  # type: ignore
except ImportError:
    load_keys = None

SCORING_MODEL = "o4-mini"  # Azure deployment name

# Azure config
AZURE_ENDPOINT = "https://datapipeline0.cognitiveservices.azure.com/"
AZURE_API_VERSION = "2025-01-01-preview"

SYSTEM_PROMPT = """You are a code complexity analyst. Given a coding task description, score the STRUCTURAL COMPLEXITY that a correct solution requires.

Score complexity of the CODE STRUCTURE, NOT difficulty of understanding or solving the task. Base your scores only on what a correct implementation would need.

Score each dimension from 0 to 4:

BRANCHING (conditional paths in correct solution):
  0 = No conditionals
  1 = Single if/else
  2 = 2-3 conditional blocks
  3 = Multiple nested or chained conditionals
  4 = Deeply nested or combinatorial branching

ITERATION (loops and recursion in correct solution):
  0 = No loops
  1 = Single flat loop
  2 = Nested loop or simple recursion
  3 = Multiple loops + recursion
  4 = Nested recursion or complex multi-pass iteration

STATE (variables tracked simultaneously):
  0 = Stateless or 1 variable
  1 = 2-3 independent variables
  2 = Multiple variables with dependencies
  3 = Complex state machine or mutable shared state
  4 = Concurrent state tracking across structures

DATA_STRUCTURES (data organization required):
  0 = Primitives only
  1 = Single flat collection
  2 = Multiple collections or simple nesting
  3 = Trees, graphs, or custom classes
  4 = Multiple interacting complex structures

EDGE_CASES (boundary conditions the code must explicitly check):
  0 = No boundary checks needed
  1 = 1-2 explicit checks (empty, null)
  2 = 3-4 boundary conditions
  3 = Multiple interacting edge cases
  4 = Combinatorial boundary conditions

COMPOSITION (algorithmic steps chained together):
  0 = Single operation
  1 = 2 sequential steps
  2 = 3-4 chained steps
  3 = Multiple algorithms coordinated
  4 = Pipeline of interdependent algorithms

Respond with ONLY a JSON object, no other text:
{"branching": <0-4>, "iteration": <0-4>, "state": <0-4>, "data_structures": <0-4>, "edge_cases": <0-4>, "composition": <0-4>}"""


def score_prompt(client, prompt_text, prompt_id):
    """Score a single prompt via o4-mini."""
    try:
        response = client.chat.completions.create(
            model=SCORING_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ],
            max_completion_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks if present)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        scores = json.loads(raw)

        # Validate all 6 dimensions present and in range
        dims = ["branching", "iteration", "state", "data_structures", "edge_cases", "composition"]
        for d in dims:
            if d not in scores or not isinstance(scores[d], int) or scores[d] < 0 or scores[d] > 4:
                return prompt_id, None, f"Invalid score for {d}: {scores.get(d)}"

        composite = sum(scores[d] for d in dims)
        return prompt_id, {"scores": scores, "composite": composite}, None

    except json.JSONDecodeError as e:
        return prompt_id, None, f"JSON parse error: {e} | raw: {raw[:200]}"
    except Exception as e:
        return prompt_id, None, f"API error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Score prompt complexity via LLM rubric")
    parser.add_argument("--prompts", default="data/experiment_prompts.jsonl")
    parser.add_argument("--output", default="data/complexity_rubric_scores.jsonl")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    # Resume: keep only rows with valid scores and rewrite the output so
    # previously-failed rows do not linger alongside their retries. A second
    # run of this script therefore converges to exactly one row per
    # prompt_id ,  failed rows are dropped and re-attempted rather than
    # accumulating as duplicates.
    done_ids = set()
    if os.path.exists(args.output):
        kept = []
        total = 0
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                rec = json.loads(line)
                if rec.get("scores"):
                    if rec["prompt_id"] not in done_ids:
                        done_ids.add(rec["prompt_id"])
                        kept.append(line)
        with open(args.output, "w", encoding="utf-8") as f:
            f.writelines(kept)
        print(f"Resuming: {len(done_ids)} valid scores kept "
              f"({total - len(kept)} failed/duplicate rows dropped for retry)")

    # Load prompts
    prompts = []
    with open(args.prompts, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["prompt_id"] not in done_ids:
                prompts.append(rec)

    print(f"Prompts to score: {len(prompts)}")
    if not prompts:
        print("Nothing to do.")
        return

    # Init Azure OpenAI client. ``load_keys`` is the author-side convenience
    # shim; public users must have AZURE_OPENAI_API_KEY set directly.
    if load_keys is not None:
        load_keys.load()
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

    scored = 0
    errors = 0
    start = time.time()

    with open(args.output, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(score_prompt, client, p["input"], p["prompt_id"]): p["prompt_id"]
                for p in prompts
            }

            for future in as_completed(futures):
                prompt_id, result, error = future.result()

                if result:
                    record = {
                        "prompt_id": prompt_id,
                        "scores": result["scores"],
                        "composite": result["composite"],
                    }
                    f_out.write(json.dumps(record) + "\n")
                    f_out.flush()
                    scored += 1
                else:
                    # Write error record so we can retry later
                    record = {
                        "prompt_id": prompt_id,
                        "scores": None,
                        "composite": None,
                        "error": error,
                    }
                    f_out.write(json.dumps(record) + "\n")
                    f_out.flush()
                    errors += 1

                total = scored + errors
                if total % 100 == 0:
                    elapsed = time.time() - start
                    rate = total / elapsed * 60
                    remaining = (len(prompts) - total) / rate if rate > 0 else 0
                    print(f"  [{total}/{len(prompts)}] scored: {scored}, errors: {errors}, "
                          f"rate: {rate:.0f}/min, ETA: {remaining:.1f}min")

    elapsed = time.time() - start
    print(f"\nDone: {scored} scored, {errors} errors in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
