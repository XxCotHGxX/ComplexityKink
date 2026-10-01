"""Step 2 of the independent-audit pilot: build and execute code variants.

Runs INSIDE the sandboxed `scorer` container (no network). For each candidate
prompt it builds three kinds of variant from the reference solution and scores
every variant with the unmodified benchmark harness (`score_solution` from
src/data_provenance/03_execute_and_score.py):

* clean     - the reference, normalized through ast.unparse;
* cosmetic  - top-level functions/classes called by the tests renamed;
* bug       - single AST mutations inside function bodies.

All variants are normalized through ast.unparse so formatting does not reveal
the category. Mutants are only built for prompts whose clean variant passes
every test and whose cosmetic variant does not.

Usage (from the repository root, PowerShell):
    docker run --rm --network none -v "${PWD}/src:/src:ro" `
        -v "<data>/independent_audit/pilot:/work" --entrypoint python scorer `
        /src/audit/02_build_and_execute_variants.py --workers 8
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import random
import re
from multiprocessing import Pool

SEED = 20260928
N_MUTANTS = 8

_spec = importlib.util.spec_from_file_location(
    "harness", "/src/data_provenance/03_execute_and_score.py")
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

COMPARE_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
                ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn,
                ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
BINOP_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add,
              ast.Div: ast.Mult, ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv}
BOOLOP_SWAP = {ast.And: ast.Or, ast.Or: ast.And}


def normalize(code: str) -> str | None:
    try:
        return ast.unparse(ast.parse(code))
    except (SyntaxError, ValueError, RecursionError):
        return None


# ---------------------------------------------------------------- cosmetic

def rename_variant(code: str, unit_tests: str) -> tuple[str | None, dict]:
    tree = ast.parse(code)
    tests_text = "\n".join(harness.parse_unit_tests(unit_tests))
    defined = [n.name for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    called = [name for name in defined
              if re.search(rf"\b{re.escape(name)}\b", tests_text)]
    if not called:
        return None, {}
    taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | set(defined)
    mapping = {}
    for node in tree.body:
        if getattr(node, "name", None) in called:
            new = (node.name + "Impl") if isinstance(node, ast.ClassDef) else (node.name + "_solution")
            if new in taken:
                return None, {}
            mapping[node.name] = new

    class Renamer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in mapping:
                node.id = mapping[node.id]
            return node

    tree = Renamer().visit(tree)
    for node in tree.body:
        if getattr(node, "name", None) in mapping:
            node.name = mapping[node.name]
    return ast.unparse(tree), mapping


# ---------------------------------------------------------------- mutants

def _function_nodes(tree):
    """Yield nodes inside function bodies, skipping docstrings."""
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = fn.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                body = body[1:]
            for stmt in body:
                yield from ast.walk(stmt)


def mutation_sites(tree) -> list[tuple]:
    sites, seen = [], set()
    for node in _function_nodes(tree):
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in COMPARE_SWAP:
                    sites.append(("compare", node, i))
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in BINOP_SWAP:
            sites.append(("arith", node, None))
        elif isinstance(node, ast.BoolOp) and type(node.op) in BOOLOP_SWAP:
            sites.append(("boolean", node, None))
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            sites.append(("constant", node, None))
        elif isinstance(node, (ast.If, ast.While)):
            sites.append(("negate_condition", node, None))
    return sites


def apply_mutation(code: str, site_index: int) -> tuple[str, str]:
    tree = ast.parse(code)
    kind, node, i = mutation_sites(tree)[site_index]
    if kind == "compare":
        node.ops[i] = COMPARE_SWAP[type(node.ops[i])]()
    elif kind == "arith":
        node.op = BINOP_SWAP[type(node.op)]()
    elif kind == "boolean":
        node.op = BOOLOP_SWAP[type(node.op)]()
    elif kind == "constant":
        node.value = node.value + 1
    elif kind == "negate_condition":
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), kind


# ---------------------------------------------------------------- execution

def run_variant(v: dict) -> dict:
    status, pass_rate = harness.score_solution(v["code"], v["unit_tests"])
    out = {k: v[k] for k in v if k != "unit_tests"}
    out.update({"harness_pass_rate": pass_rate, "test_status": status})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/work/pilot_inputs.jsonl")
    ap.add_argument("--out", default="/work/variants_executed.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    # Phase A: clean and cosmetic variants for every candidate.
    phase_a = []
    for p in prompts:
        clean = normalize(p["reference_code"])
        if clean is None:
            continue
        base = {"prompt_id": p["prompt_id"], "unit_tests": p["unit_tests"]}
        phase_a.append({**base, "variant_id": f"{p['prompt_id']}:clean", "category": "clean",
                        "mutation": None, "code": clean})
        renamed, mapping = rename_variant(clean, p["unit_tests"])
        if renamed is not None:
            phase_a.append({**base, "variant_id": f"{p['prompt_id']}:cosmetic", "category": "cosmetic",
                            "mutation": {"renamed": mapping}, "code": renamed})
    with Pool(args.workers) as pool:
        results_a = pool.map(run_variant, phase_a, chunksize=4)
    print(f"phase A: {len(results_a)} variants executed", flush=True)

    by_prompt: dict[str, dict] = {}
    for r in results_a:
        by_prompt.setdefault(r["prompt_id"], {})[r["category"]] = r
    eligible = [p for p in prompts
                if by_prompt.get(p["prompt_id"], {}).get("clean", {}).get("harness_pass_rate") == 1.0
                and "cosmetic" in by_prompt[p["prompt_id"]]
                and by_prompt[p["prompt_id"]]["cosmetic"]["harness_pass_rate"] < 1.0]
    print(f"eligible for mutation: {len(eligible)} of {len(prompts)}", flush=True)

    # Phase B: up to N_MUTANTS single mutations per eligible prompt.
    phase_b = []
    for p in eligible:
        clean = by_prompt[p["prompt_id"]]["clean"]["code"]
        n_sites = len(mutation_sites(ast.parse(clean)))
        rng = random.Random(f"{SEED}-{p['prompt_id']}")
        for k, site in enumerate(sorted(rng.sample(range(n_sites), min(N_MUTANTS, n_sites)))):
            try:
                mutated, kind = apply_mutation(clean, site)
            except Exception:
                continue
            if mutated == clean:
                continue
            phase_b.append({"prompt_id": p["prompt_id"], "unit_tests": p["unit_tests"],
                            "variant_id": f"{p['prompt_id']}:bug{k}", "category": "bug",
                            "mutation": {"kind": kind, "site_index": site}, "code": mutated})
    with Pool(args.workers) as pool:
        results_b = pool.map(run_variant, phase_b, chunksize=4)
    print(f"phase B: {len(results_b)} mutants executed", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results_a + results_b:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(results_a) + len(results_b)} rows to {args.out}")


if __name__ == "__main__":
    main()
