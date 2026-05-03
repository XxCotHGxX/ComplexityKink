"""
Stage D step 8: audit unit-test coherence for selected prompt files.

The source dataset's tests are useful, but they are not guaranteed to be a
clean benchmark contract. This script flags tests that are syntactically bad,
assertion-free, duplicate-heavy, dependent on hidden fixtures/globals, or that
expect a callable name never stated in the prompt.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS = ROOT / "data" / "stage_d" / "generation_delta" / "stage_d_new_prompts.jsonl"
DEFAULT_REPORT = ROOT / "data" / "stage_d" / "unit_test_audit_report.json"
DEFAULT_FLAGS = ROOT / "data" / "stage_d" / "unit_test_audit_flags.csv"

BUILTIN_NAMES = set(dir(builtins)) | {"True", "False", "None"}
EXCLUDED_CALL_ROOTS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "max", "min", "open", "print", "range", "set", "sorted", "str",
    "sum", "tuple", "zip",
    # Common assertion helpers, modules, and Flask helpers used by tests.
    "abort", "bz2", "csv", "datetime", "gzip", "json", "jsonify", "lzma",
    "math", "mock_open", "os", "patch", "pytest", "random", "raises",
    "StringIO", "sys", "time",
}
COMMON_LOCAL_NAMES = {
    "arr", "bst", "data", "e", "expected", "graph", "grid", "head", "i",
    "item", "items", "j", "k", "key", "keys", "list1", "list2", "lst",
    "m", "matrix", "n", "node", "nums", "result", "results", "root", "s",
    "tail", "tree", "value", "values", "x", "y",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_tests(raw) -> list[str] | None:
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, str)]
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [t for t in parsed if isinstance(t, str)]
    return None


def has_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


def word_in_prompt(name: str, prompt: str) -> bool:
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, prompt, flags=re.IGNORECASE) is not None


class TestNameAudit(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loads: Counter[str] = Counter()
        self.stores: set[str] = set()
        self.imports: set[str] = set()
        self.calls: Counter[str] = Counter()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads[node.id] += 1
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.stores.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.stores.add(node.arg)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if isinstance(node.name, str):
            self.stores.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stores.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stores.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.imports.add(alias.asname or alias.name)

    def visit_Call(self, node: ast.Call) -> None:
        root = node.func
        if isinstance(root, ast.Name):
            self.calls[root.id] += 1
        elif isinstance(root, ast.Attribute):
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                self.calls[root.id] += 1
        self.generic_visit(node)


def audit_prompt(row: dict) -> dict:
    prompt_id = row["prompt_id"]
    prompt = row.get("input", "")
    tests = parse_tests(row.get("unit_tests"))
    flags: list[str] = []
    details: dict[str, object] = {}

    if tests is None:
        return {
            "prompt_id": prompt_id,
            "rubric_bin": row.get("rubric_bin"),
            "flags": ["hard_unit_tests_not_parseable"],
            "details": {},
        }
    if not tests:
        return {
            "prompt_id": prompt_id,
            "rubric_bin": row.get("rubric_bin"),
            "flags": ["hard_no_tests"],
            "details": {},
        }

    joined = "\n".join(tests)
    assert_count = sum(test.count("assert") for test in tests)
    normalized_tests = [re.sub(r"\s+", " ", test).strip() for test in tests]
    duplicate_count = len(normalized_tests) - len(set(normalized_tests))
    details.update({
        "n_tests": len(tests),
        "assert_count": assert_count,
        "duplicate_count": duplicate_count,
    })

    if has_surrogate(joined):
        flags.append("hard_invalid_unicode_surrogate")
    if assert_count == 0:
        flags.append("hard_no_asserts")
    if duplicate_count >= 5:
        flags.append("weak_many_duplicate_tests")
    elif duplicate_count > 0:
        flags.append("weak_some_duplicate_tests")

    try:
        tree = ast.parse(joined.encode("utf-8", "replace").decode("utf-8"))
    except SyntaxError as exc:
        flags.append("hard_syntax_error")
        details["syntax_error"] = exc.msg
        return {
            "prompt_id": prompt_id,
            "rubric_bin": row.get("rubric_bin"),
            "flags": flags,
            "details": details,
        }

    audit = TestNameAudit()
    audit.visit(tree)
    stores = audit.stores | audit.imports
    call_roots = sorted(
        name for name in audit.calls
        if name not in BUILTIN_NAMES
        and name not in audit.imports
        and name not in stores
        and name not in EXCLUDED_CALL_ROOTS
    )
    details["test_call_roots"] = call_roots

    if call_roots and not any(word_in_prompt(name, prompt) for name in call_roots):
        flags.append("contract_hidden_test_callable")

    explicit_io = bool(re.search(
        r"\b(read from standard input|stdin|command[- ]line|script is executed|"
        r"print the|should print|output should be printed|first line contains|"
        r"next .* lines contain)\b",
        prompt,
        flags=re.IGNORECASE,
    ))
    if explicit_io and call_roots:
        flags.append("contract_io_prompt_callable_tests")

    loads = {
        name for name in audit.loads
        if name not in BUILTIN_NAMES and name not in audit.imports
    }
    unresolved = sorted((loads - stores) - set(audit.calls))
    external = [
        name for name in unresolved
        if name not in COMMON_LOCAL_NAMES and not word_in_prompt(name, prompt)
    ]
    if external:
        flags.append("risk_external_fixture_or_global")
        details["external_names"] = external

    return {
        "prompt_id": prompt_id,
        "rubric_bin": row.get("rubric_bin"),
        "flags": sorted(set(flags)),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prompt unit tests.")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--flags", default=str(DEFAULT_FLAGS))
    args = parser.parse_args()

    rows = load_jsonl(Path(args.prompts))
    audits = [audit_prompt(row) for row in rows]
    flagged = [item for item in audits if item["flags"]]

    flag_counts: Counter[str] = Counter()
    by_bin: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, str] = {}
    for item in flagged:
        bin_label = item.get("rubric_bin") or "unknown"
        for flag in item["flags"]:
            flag_counts[flag] += 1
            by_bin[bin_label][flag] += 1
            examples.setdefault(flag, item["prompt_id"])

    report = {
        "prompts": str(Path(args.prompts)),
        "n_prompts": len(rows),
        "n_flagged_prompts": len(flagged),
        "flag_counts": dict(flag_counts),
        "flag_counts_by_bin": {
            bin_label: dict(counter)
            for bin_label, counter in sorted(by_bin.items())
        },
        "examples": examples,
        "recommendation": (
            "Drop hard_* prompts. Treat contract_hidden_test_callable and "
            "contract_io_prompt_callable_tests as exclusion flags unless the "
            "prompt is rewritten to state the tested callable/API contract."
        ),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    flags_path = Path(args.flags)
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    with flags_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_id", "rubric_bin", "flags", "details"],
        )
        writer.writeheader()
        for item in flagged:
            writer.writerow({
                "prompt_id": item["prompt_id"],
                "rubric_bin": item.get("rubric_bin"),
                "flags": ";".join(item["flags"]),
                "details": json.dumps(item["details"], sort_keys=True),
            })

    print("Stage D unit-test audit")
    print(f"  Prompts: {len(rows):,}")
    print(f"  Flagged: {len(flagged):,}")
    print(f"  Report:  {report_path}")
    print(f"  Flags:   {flags_path}")
    for flag, count in flag_counts.most_common():
        print(f"  {flag}: {count:,} (example {examples[flag]})")


if __name__ == "__main__":
    main()
