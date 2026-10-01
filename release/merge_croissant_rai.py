#!/usr/bin/env python
"""Merge draft Responsible-AI (RAI) and provenance fields into a Croissant file.

Typical use: download the Croissant JSON-LD that Hugging Face (or Harvard
Dataverse) auto-generates for the uploaded dataset, then run

    python release/merge_croissant_rai.py \
        --croissant hf_croissant.json \
        --out complexity_kink_croissant_rai.json

The script uses only the Python standard library. If ``mlcroissant`` is
importable (install it in a separate virtual environment), ``--validate``
also loads the merged file and reports Croissant errors and warnings.

Rules:
  * ``@context`` gains the ``rai``, ``prov``, and ``dct`` prefixes if missing.
  * Every key of the RAI file except ``@context`` and keys starting with ``_``
    is copied to the dataset node. Existing keys are kept unless
    ``--overwrite`` is given.
  * The output is refused while any value still contains ``TODO(author):``
    unless ``--allow-todo`` is passed (useful for dry runs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = [
    "rai:dataLimitations",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
    "rai:dataUseCases",
    "rai:dataSocialImpact",
    "rai:hasSyntheticData",
    "prov:wasDerivedFrom",
    "prov:wasGeneratedBy",
]
PREFIXES = {
    "rai": "http://mlcommons.org/croissant/RAI/",
    "prov": "http://www.w3.org/ns/prov#",
    "dct": "http://purl.org/dc/terms/",
}
RAI_CONFORMANCE = "http://mlcommons.org/croissant/RAI/1.0"
TODO = "TODO(author):"


def find_todos(value, path="$"):
    if isinstance(value, str):
        return [path] if TODO in value else []
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in find_todos(v, f"{path}[{i}]")]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in find_todos(v, f"{path}.{k}")]
    return []


def merge(croissant: dict, rai: dict, overwrite: bool, add_rai_conformance: bool) -> tuple[dict, list[str]]:
    notes = []
    out = dict(croissant)
    ctx = out.get("@context")
    if not isinstance(ctx, dict):
        raise SystemExit("Croissant @context must be a JSON object")
    ctx = dict(ctx)
    for prefix, iri in {**PREFIXES, **(rai.get("@context") or {})}.items():
        if prefix not in ctx:
            ctx[prefix] = iri
            notes.append(f"added @context prefix {prefix}")
        elif ctx[prefix] != iri:
            notes.append(f"WARNING: @context {prefix} is {ctx[prefix]!r}, expected {iri!r}; left unchanged")
    out["@context"] = ctx

    for key, value in rai.items():
        if key == "@context" or key.startswith("_"):
            continue
        if key in out and not overwrite:
            notes.append(f"kept existing {key} (use --overwrite to replace)")
            continue
        out[key] = value
        notes.append(f"set {key}")

    if add_rai_conformance:
        conf_key = "conformsTo" if "conformsTo" in out or "dct:conformsTo" not in out else "dct:conformsTo"
        current = out.get(conf_key)
        values = current if isinstance(current, list) else ([current] if current else [])
        if RAI_CONFORMANCE not in values:
            values.append(RAI_CONFORMANCE)
            notes.append(f"added {RAI_CONFORMANCE} to {conf_key}")
        out[conf_key] = values if len(values) > 1 else values[0]
    return out, notes


def validate(path: Path) -> int:
    try:
        import mlcroissant as mlc  # type: ignore
    except ImportError:
        print("mlcroissant is not installed; skipping validation (see release/MERGE_CROISSANT.md).")
        return 0
    ds = mlc.Dataset(jsonld=str(path))
    issues = ds.metadata.ctx.issues
    for e in sorted(issues.errors):
        print("ERROR:", e)
    for w in sorted(issues.warnings):
        print("WARNING:", w)
    print(f"mlcroissant: {len(issues.errors)} errors, {len(issues.warnings)} warnings")
    return 1 if issues.errors else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--croissant", type=Path, required=True, help="Auto-generated Croissant JSON-LD to extend.")
    ap.add_argument("--rai", type=Path, default=HERE / "croissant_rai_fields.json", help="RAI fields JSON.")
    ap.add_argument("--out", type=Path, required=True, help="Path for the merged Croissant JSON-LD.")
    ap.add_argument("--overwrite", action="store_true", help="Replace RAI keys already present in the input.")
    ap.add_argument("--add-rai-conformance", action="store_true",
                    help=f"Also declare conformance to {RAI_CONFORMANCE} in conformsTo.")
    ap.add_argument("--allow-todo", action="store_true", help="Write even if TODO(author) placeholders remain.")
    ap.add_argument("--validate", action="store_true", help="Validate the merged file with mlcroissant if installed.")
    args = ap.parse_args()

    croissant = json.loads(args.croissant.read_text(encoding="utf-8"))
    rai = json.loads(args.rai.read_text(encoding="utf-8"))
    merged, notes = merge(croissant, rai, args.overwrite, args.add_rai_conformance)
    for n in notes:
        print(n)

    missing = [k for k in REQUIRED if k not in merged]
    if missing:
        raise SystemExit(f"Merged file lacks required NeurIPS RAI properties: {missing}")
    todos = find_todos(merged)
    if todos and not args.allow_todo:
        print("Unresolved TODO(author) placeholders:")
        for p in todos:
            print("  ", p)
        raise SystemExit("Resolve the placeholders in the RAI file (or pass --allow-todo for a dry run).")
    if todos:
        print(f"WARNING: {len(todos)} TODO(author) placeholders written (--allow-todo).")

    args.out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if args.validate:
        sys.exit(validate(args.out))


if __name__ == "__main__":
    main()
