#!/usr/bin/env python3
"""Prepare pinned Lean extraction and render complete, row-aligned passages.

Only load metadata/texts pickle files from the user's trusted original index.
prepare uses the standard library; render needs the already installed tokenizer.
No models, network requests, GPU work or repository mutations occur in prepare.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tokenizer_files(path):
    names = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "vocab.json", "merges.txt", "config.json",
             "sentence_bert_config.json", "modules.json", "config_sentence_transformers.json"]
    return {name: sha_file(Path(path) / name) for name in names if (Path(path) / name).is_file()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def canonical_name(row):
    name = row["name"]
    return name if isinstance(name, str) else ".".join(map(str, name))


def load_db(db):
    db = Path(db).resolve()
    with (db / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)
    with (db / "texts.pkl").open("rb") as f:
        texts = pickle.load(f)
    rows = metadata["data"]
    names = [canonical_name(row) for row in rows]
    if len(rows) != len(texts) or len(set(names)) != len(names):
        raise ValueError("row/text mismatch or duplicate declaration names")
    if metadata.get("num_vectors", len(rows)) != len(rows):
        raise ValueError("metadata vector count disagrees with rows")
    if any(not name or "\n" in name for name in names):
        raise ValueError("invalid declaration name")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("passages must be strings")
    return metadata, texts, names


def prepare(args):
    db = Path(args.db_dir).resolve()
    _, _, names = load_db(db)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "names.txt").write_text("\n".join(names) + "\n")
    template = Path(__file__).with_name("export_structure_context.lean")
    source = template.read_text()
    for module in args.imports:
        if not module or any(not (c.isalnum() or c in "._'") for c in module):
            raise ValueError("invalid Lean module name")
    imports = "\n".join("import " + module for module in args.imports)
    source = source.replace("import Lean\n", "import Lean\n" + imports + "\n", 1)
    source += '\nset_option maxHeartbeats 0 in\nexport_structure_context ' + json.dumps(str(out / "names.txt"), ensure_ascii=False) + " " + json.dumps(str(out / "contexts.jsonl"), ensure_ascii=False) + "\n"
    (out / "export.lean").write_text(source)
    write_json(out / "prepare_manifest.json", {
        "schema_version": "complete-context-v1", "db_dir": str(db),
        "row_count": len(names), "imports": args.imports,
        "metadata_sha256": sha_file(db / "metadata.pkl"),
        "texts_sha256": sha_file(db / "texts.pkl"),
        "ordered_names_sha256": sha_bytes(("\n".join(names) + "\n").encode()),
        "extractor_sha256": sha_file(template), "wrapper_sha256": sha_file(out / "export.lean"),
        "snapshot_claim": args.snapshot,
        "snapshot_verified": False,
        "note": "Match toolchain/revision and archive lake-manifest before execution."
    })
    print(json.dumps({"prepared": str(out), "indexed_rows": len(names)}))


def context_block(record):
    if record.get("verification") != "projection-expressions-typechecked":
        raise ValueError("unverified interface " + record["name"])
    fields = record["fields"]
    if len({f["name"] for f in fields}) != len(fields):
        raise ValueError("duplicate effective field")
    lines = ["Lean interface specification (complete-context-v1):",
             record["kind"] + " " + record["name"] + " : " + record["declaration_type"],
             "Universe parameters: " + ", ".join(record["level_parameters"]), "Parameter context:"]
    lines.extend("  " + p["binder_info"] + " " + p["name"] + " : " + p["type"] for p in record["parameters"])
    self_binder = "[self : " if record["kind"] == "class" else "(self : "
    lines.append("  " + self_binder + record["self_type"] + ("]" if record["kind"] == "class" else ")"))
    lines.append("Direct parent interfaces in this context:")
    lines.extend("  " + p["type"] + " via " + p["projection"] for p in record["parents"])
    lines.append("Effective fields, including inherited fields:")
    for f in fields:
        lines.append("  self." + f["name"] + " [" + ("law" if f["is_proof"] else "data") + "] : " + f["type"])
        expanded = f.get("head_expanded_type", f["type"])
        if expanded != f["type"]:
            lines.append("    Definitional form: " + expanded)
    lines.append("Constructor: " + record["constructor"]["name"] + " : " + record["constructor"]["type"])
    return "\n".join(lines)


def load_contexts(path):
    records, manifest = {}, None
    with Path(path).open() as f:
        for line in f:
            item = json.loads(line)
            if item["record_kind"] == "extraction_error":
                raise ValueError("Lean extraction failed: " + str(item))
            if item["record_kind"] == "extraction_manifest":
                if manifest is not None:
                    raise ValueError("duplicate extraction manifest")
                manifest = item
            elif item["record_kind"] == "structure_context":
                if item["name"] in records:
                    raise ValueError("duplicate interface")
                records[item["name"]] = item
            else:
                raise ValueError("unknown extraction record")
    if manifest is None or manifest["error_count"] or manifest["exported_count"] != len(records):
        raise ValueError("missing or inconsistent extraction manifest")
    return records, manifest


def enrich_rows(names, originals, records, count_tokens, max_tokens):
    if set(records) - set(names):
        raise ValueError("interfaces contain declarations outside original index")
    enriched = list(originals)
    audit, changed = [], []
    for row, (name, original) in enumerate(zip(names, originals)):
        if name not in records:
            continue
        record = records[name]
        candidate = original + "\n\n" + context_block(record)
        original_count, candidate_count = count_tokens(original), count_tokens(candidate)
        eligible = candidate_count <= max_tokens
        item = {"row": row, "name": name, "field_count": len(record["fields"]),
                "parent_count": len(record["parents"]), "original_tokens": original_count,
                "complete_tokens": candidate_count, "changed": eligible,
                "reason": "complete_interface_fits" if eligible else "overflow_original_retained",
                "original_sha256": sha_bytes(original.encode()),
                "candidate_sha256": sha_bytes(candidate.encode())}
        audit.append(item)
        if eligible:
            enriched[row] = candidate
            changed.append({**item, "original_text": original, "enriched_text": candidate})
    return enriched, audit, changed


def render(args):
    prepared = Path(args.prepared_dir).resolve()
    manifest = json.loads((prepared / "prepare_manifest.json").read_text())
    db = Path(manifest["db_dir"])
    for file, key in [("metadata.pkl", "metadata_sha256"), ("texts.pkl", "texts_sha256")]:
        if sha_file(db / file) != manifest[key]:
            raise ValueError("original index changed since prepare: " + file)
    _, originals, names = load_db(db)
    records, extraction = load_contexts(prepared / "contexts.jsonl")
    if extraction["requested_count"] != len(names):
        raise ValueError("extraction requested count differs from index")
    if extraction["missing_names"] and not args.allow_unresolved_corpus_names:
        raise ValueError("unresolved indexed names; inspect snapshot before opting into scoped coverage")
    import transformers
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    def count(text):
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    enriched, audit, changed = enrich_rows(names, originals, records, count, args.max_tokens)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # These are distinct channels; texts.pkl remains the immutable reranker input.
    with (out / "retrieval_texts.pkl").open("wb") as f:
        pickle.dump(enriched, f)
    for filename, rows in [("passage_audit.jsonl", audit), ("changed_passages.jsonl", changed)]:
        with (out / filename).open("w") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_json(out / "render_manifest.json", {
        "schema_version": "complete-context-v1", "prepared_manifest": manifest,
        "extraction_manifest": extraction, "contexts_sha256": sha_file(prepared / "contexts.jsonl"),
        "row_count": len(names), "interfaces_extracted": len(records),
        "changed_rows": len(changed), "overflow_rows": len(audit) - len(changed),
        "max_tokens": args.max_tokens, "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "tokenizer_file_hashes": tokenizer_files(args.tokenizer_path),
        "transformers_version": transformers.__version__,
        "token_count_method": "AutoTokenizer, specials included, no truncation",
        "runtime_tokenization_check_required": True,
        "reranker_texts": str(db / "texts.pkl"),
        "retrieval_texts_sha256": sha_file(out / "retrieval_texts.pkl"),
        "changed_passages_sha256": sha_file(out / "changed_passages.jsonl"),
        "implementation_sha256": sha_file(__file__)
    })
    print(json.dumps({"output": str(out), "interfaces": len(records), "changed": len(changed),
                      "overflow": len(audit) - len(changed)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--db-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--imports", nargs="+", default=["Mathlib"])
    p.add_argument("--snapshot", required=True, help="Actual Lean/Mathlib revision claim; archived separately")
    p.set_defaults(func=prepare)
    p = sub.add_parser("render")
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--max-tokens", type=int, required=True, help="Verified encoder runtime budget")
    p.add_argument("--allow-unresolved-corpus-names", action="store_true")
    p.set_defaults(func=render)
    args = parser.parse_args()
    if getattr(args, "max_tokens", 1) < 1:
        parser.error("max-tokens must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
