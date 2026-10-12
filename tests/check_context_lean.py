#!/usr/bin/env python3
"""Meaningful core Lean + row-preservation checks; no Mathlib/model/GPU run."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from lean_context_search.context import context_block, enrich_rows, load_contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", required=True, help="Installed Lean executable; no download")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parent
    template = root / 'src/lean_context_search/export_structure_context.lean'
    fixture = root / 'lean/context_fixtures.lean'
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = ["ContextFixtures." + n for n in ["Child", "Diamond", "Rich", "Dependent", "ordinary"]]
    (out / "names.txt").write_text("\n".join(names) + "\n")
    source = template.read_text() + "\n" + fixture.read_text()
    source += '\nset_option maxHeartbeats 0 in\nexport_structure_context ' + json.dumps(str(out / "names.txt")) + " " + json.dumps(str(out / "contexts.jsonl")) + "\n"
    (out / "smoke.lean").write_text(source)
    version = subprocess.run([args.lean, "--version"], capture_output=True, text=True, check=True)
    run = subprocess.run([args.lean, str(out / "smoke.lean")], capture_output=True, text=True)
    (out / "lean.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        result = {"lean_exit_code": run.returncode, "passed": False, "output": str(out / "lean.log")}
    else:
        records, manifest = load_contexts(out / "contexts.jsonl")
        child, diamond, rich, dependent = (records["ContextFixtures." + n] for n in ["Child", "Diamond", "Rich", "Dependent"])
        field = lambda r, name: next(x for x in r["fields"] if x["name"] == name)
        original = ["original " + n for n in names]
        # Character length is deliberately an artificial test counter, not model tokenization.
        enriched, audit, changed = enrich_rows(names, original, records, len, 1000000)
        overflow, over_audit, over_changed = enrich_rows(names, original, records, len, 1)
        checks = {
            "all_fixture_names_resolve": not manifest["missing_names"],
            "ordinary_theorem_is_not_enriched": names[-1] not in records and enriched[-1] == original[-1],
            "parent_parameter_substitution_preserved": "List" in field(child, "payload")["type"] and "List" in child["parents"][0]["type"],
            "dependent_field_reference_preserved": "carrier" in field(dependent, "point")["type"] and "point" in field(dependent, "law")["type"],
            "diamond_shared_fields_not_duplicated": {f["name"] for f in diamond["fields"]} == {"payload", "valid", "left", "right", "extra"} and len(diamond["fields"]) == 5,
            "both_declared_diamond_parents_retained": len(diamond["parents"]) == 2,
            "class_and_law_roles_preserved": rich["kind"] == "class" and field(rich, "law")["is_proof"] and not field(rich, "op")["is_proof"],
            "defined_law_exposed_without_erasing_original_type": "Fixes" in field(rich, "law")["type"] and "∀" in field(rich, "law")["head_expanded_type"] and "Definitional form:" in context_block(rich),
            "closed_types_and_universes_archived": all(f["closed_type"] for r in records.values() for f in r["fields"]) and bool(child["level_parameters"]),
            "one_row_per_original_name": len(enriched) == len(original) and [x["row"] for x in changed] == [0, 1, 2, 3],
            "original_passages_preserved_as_prefix": all(enriched[i].startswith(original[i] + "\n\n") for i in range(4)),
            "overflow_preserves_original_without_partial_interface": overflow == original and not over_changed and all(x["reason"] == "overflow_original_retained" for x in over_audit),
        }
        try:
            enrich_rows(names, original, {"not.indexed": child}, len, 1000000)
        except ValueError:
            checks["out_of_corpus_context_rejected"] = True
        else:
            checks["out_of_corpus_context_rejected"] = False
        (out / "example_passage.txt").write_text(enriched[0] + "\n")
        result = {"lean_exit_code": run.returncode, "lean_version": version.stdout.strip(),
                  "exported_fixture_interfaces": len(records), "checks": checks,
                  "passed": all(checks.values()), "real_model_tokenization": False,
                  "mathlib_extraction": False, "retrieval_run": False,
                  "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [template, root/'src/lean_context_search/context.py', fixture, Path(__file__)]}}
    (out / "check_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
