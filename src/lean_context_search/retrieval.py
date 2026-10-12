"""Rescore a fixed candidate pool with the ordinary cached-vector head.

This performs no encoding, label generation, graph expansion or network access.
It uses exactly the retrieval scorer and cannot add a missing candidate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .cache import Cache, ExactRetriever, normalize, stable_topk


def rerank_vector(retriever, vector, candidate_rows, top_k=None):
    raw = list(candidate_rows)
    if not raw or any(isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer)) for r in raw):
        raise ValueError("candidate rows must be a nonempty integer list")
    rows = np.asarray(raw, dtype=np.int64)
    if len(set(raw)) != len(raw) or np.any(rows < 0) or np.any(rows >= len(retriever.documents.items)):
        raise ValueError("duplicate or out-of-range candidate row")
    count = len(rows) if top_k is None else top_k
    if not isinstance(count, int) or count <= 0:
        raise ValueError("top_k must be a positive integer")
    q = normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
    if q.shape[1] != retriever.documents.vectors.shape[1]:
        raise ValueError("query dimension mismatch")
    docs = normalize(np.asarray(retriever.documents.vectors[rows]))
    scores = np.einsum("nd,d->n", docs, q[0], optimize=False)
    if retriever.alpha:
        adapted = np.einsum("nk,k->n", retriever.projected[rows], retriever.head.project(q)[0], optimize=False)
        scores = (1 - retriever.alpha) * scores + retriever.alpha * adapted
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite reranking score")
    rows, scores = stable_topk(rows, scores, count)
    return [{"row_id": int(row), "name": retriever.documents.items[row], "score": float(score)}
            for row, score in zip(rows, scores)]


def read_pools(path, queries, documents):
    pools = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        qid = row["query_id"]
        if qid in pools or qid not in queries.row:
            raise ValueError("duplicate or uncached pool query")
        if row.get("query_text_sha256") != queries.items[queries.row[qid]]["text_sha256"]:
            raise ValueError("pool query text does not match cache")
        for result in row["results"]:
            index = result["row_id"]
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(documents.items):
                raise ValueError("invalid pool row")
            if result.get("name") != documents.items[index]:
                raise ValueError("pool row/name mismatch")
        pools[qid] = row
    if set(pools) != set(queries.row):
        raise ValueError("missing query candidate pool")
    return pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--pools", required=True)
    parser.add_argument("--head")
    parser.add_argument("--projection")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    retriever = ExactRetriever(args.documents, args.head, args.projection, args.alpha)
    queries = Cache(args.queries, "queries")
    retriever.documents.compatible_queries(queries)
    pools = read_pools(args.pools, queries, retriever.documents)
    with Path(args.output).open("x") as stream:
        for i, item in enumerate(queries.items):
            pool = pools[item["query_id"]]
            result = {"query_id": item["query_id"], "query_text_sha256": item["text_sha256"],
                      "candidate_count": len(pool["results"]), "mode": "fixed_pool_head_reranking",
                      "results": rerank_vector(retriever, queries.vectors[i],
                                               [r["row_id"] for r in pool["results"]], args.top_k)}
            stream.write(json.dumps(result, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
