"""Evaluate the existing development labels with binary acceptable-answer metrics.

Labels and public designated-target metrics are separate. No model selection,
training, generation or protected confirmation access occurs here.
"""
import argparse
import json
import math
from pathlib import Path
from common import HERE, rows, sha, verify_data, write
from lean_context_search.cache import text_hash

def evaluate(records, rankings, expected_split, cutoffs=(1, 5, 10, 50, 100, 200)):
    if expected_split != "development" or not records:
        raise ValueError("evaluation needs a nonempty declared held-out split")
    labels = {r["query_id"]: r for r in records}
    outputs = {r["query_id"]: r for r in rankings}
    if len(labels) != len(records) or len(outputs) != len(rankings) or set(labels) != set(outputs):
        raise ValueError("duplicate/missing/extra evaluation query IDs")
    per_query = []
    for qid, label in labels.items():
        if (label.get("split") != expected_split or label.get("judgment_status") != "accepted" or
                not label.get("judgment_provenance") or not label.get("positive_names")):
            raise ValueError("wrong split or unreviewed evaluation answers")
        output = outputs[qid]
        if output.get("query_text_sha256") != text_hash(label["query_text"]):
            raise ValueError("evaluation query text/ranking mismatch")
        names = [r["name"] for r in output["results"]]
        if len(set(names)) != len(names):
            raise ValueError("duplicate candidate names")
        gold = set(label["positive_names"])
        row = {"query_id": qid, "source_names": label.get("source_names", []),
               "positive_names": sorted(gold), "metrics": {}}
        for k in cutoffs:
            if len(names) < k:
                raise ValueError("short candidate list for requested metric cutoff")
            hits = sum(n in gold for n in names[:k])
            dcg = sum(1 / math.log2(i + 2) for i, n in enumerate(names[:k]) if n in gold)
            ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
            row["metrics"].update({f"nDCG@{k}": dcg / ideal,
                                    f"Recall@{k}": hits / len(gold), f"Hit@{k}": float(hits > 0)})
        per_query.append(row)
    means = {key: sum(r["metrics"][key] for r in per_query) / len(per_query)
             for key in per_query[0]["metrics"]}
    return {"schema": "binary-answer-retrieval-evaluation-v1", "split": expected_split,
            "queries": len(per_query), "metrics": means, "per_query": per_query}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rankings', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output exists; choose a new file')
    verify_data()
    labels = HERE / 'data/development.jsonl'
    result = evaluate(rows(labels), rows(args.rankings), 'development')
    result.update({'records_sha256': sha(labels), 'rankings_sha256': sha(args.rankings)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_query'}, indent=2))


if __name__ == '__main__':
    main()
