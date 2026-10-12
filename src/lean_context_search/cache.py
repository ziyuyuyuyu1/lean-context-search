"""Frozen-cache affine cosine head, offline document projection and exact search.

NumPy/CPU reference implementation. It never loads an encoder, test labels or
the upstream index. Frozen document vectors and original row IDs are immutable inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EPS = 1e-8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize(x, allow_zero=False):
    x = np.asarray(x)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("expected a finite [rows, dimensions] array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not allow_zero and np.any(norms <= EPS):
        raise ValueError("zero or nearly zero frozen embedding")
    return x / np.maximum(norms, EPS)


def normalization_backward(raw, unit, gradient):
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    regular = (gradient - unit * np.sum(unit * gradient, axis=1, keepdims=True))
    return np.where(norms > EPS, regular, gradient) / np.maximum(norms, EPS)


def write_cache(directory, vectors, items, role, encoder_id, representation_id, representation_family=None):
    """Export an existing array/memmap in chunks, preserving its row ordering.

    For queries, items are {query_id, text_sha256, split}; documents use names.
    IDs identify pinned encoder instructions/revisions and C representation.
    """
    if role not in {"documents", "queries"}:
        raise ValueError("invalid cache role")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if len(vectors.shape) != 2 or len(items) != len(vectors):
        raise ValueError("cache row mismatch")
    output = np.lib.format.open_memmap(directory / "vectors.npy", mode="w+",
                                        dtype=np.float32, shape=vectors.shape)
    for start in range(0, len(vectors), 2048):
        chunk = np.asarray(vectors[start:start + 2048], dtype=np.float32)
        normalize(chunk)  # Validate; store original vectors, not new embeddings.
        output[start:start + len(chunk)] = chunk
    output.flush()
    del output
    write_json(directory / "items.json", items)
    files = {name: {"path": name, "sha256": sha256(directory / name)}
             for name in ["vectors.npy", "items.json"]}
    write_json(directory / "manifest.json", {
        "schema": "frozen-vector-cache-v1", "role": role,
        "shape": list(vectors.shape), "dtype": "float32",
        "encoder_id": encoder_id, "representation_id": representation_id,
        **({"representation_family": representation_family} if representation_family else {}),
        "row_order_sha256": files["items.json"]["sha256"],
        "normalization": "raw-fp32-l2-at-score",
        "files": files,
    })
    return directory / "manifest.json"


class Cache:
    def __init__(self, manifest_path, role):
        self.path = Path(manifest_path).resolve()
        self.digest = sha256(self.path)
        self.meta = json.loads(self.path.read_text())
        if self.meta.get("schema") != "frozen-vector-cache-v1" or self.meta.get("role") != role:
            raise ValueError("unsupported cache schema/role")
        paths = {}
        for name in ["vectors.npy", "items.json"]:
            item = self.meta["files"][name]
            relative = Path(item["path"])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError("cache paths must be relative within manifest directory")
            paths[name] = self.path.parent / relative
            if not paths[name].resolve().is_relative_to(self.path.parent.resolve()):
                raise ValueError('cache path escapes its manifest directory')
            if sha256(paths[name]) != item["sha256"]:
                raise ValueError("cache checksum mismatch: " + name)
        if self.meta.get('row_order_sha256', self.meta['files']['items.json']['sha256']) != self.meta['files']['items.json']['sha256']:
            raise ValueError('cache row-order hash mismatch')
        if self.meta.get('normalization','raw-fp32-l2-at-score') != 'raw-fp32-l2-at-score':
            raise ValueError('unsupported cache normalization')
        self.vectors = np.load(paths["vectors.npy"], mmap_mode="r", allow_pickle=False)
        self.items = json.loads(paths["items.json"].read_text())
        if self.vectors.ndim != 2 or list(self.vectors.shape) != self.meta["shape"]:
            raise ValueError("cache shape mismatch")
        if self.vectors.dtype != np.float32 or len(self.items) != len(self.vectors):
            raise ValueError("cache dtype/row mismatch")
        # The file checksums bind row order. Check content as well as file layout.
        for start in range(0, len(self.vectors), 2048):
            if not np.isfinite(self.vectors[start:start + 2048]).all():
                raise ValueError("cache contains nonfinite vectors")
        if not all(isinstance(self.meta.get(k), str) and self.meta[k]
                   for k in ["encoder_id", "representation_id"]):
            raise ValueError("missing cache provenance IDs")
        if role == "documents":
            keys = self.items
            if not all(isinstance(n, str) and n for n in keys):
                raise ValueError("invalid declaration names")
        else:
            keys = [r["query_id"] for r in self.items]
            for row in self.items:
                if row.get("split") not in {"train", "development", "confirmation", "benchmark"}:
                    raise ValueError("invalid query cache split")
                if not isinstance(row.get("text_sha256"), str) or len(row["text_sha256"]) != 64:
                    raise ValueError("missing query text hash")
        if any(not isinstance(k, str) or not k for k in keys) or len(set(keys)) != len(keys):
            raise ValueError("invalid/duplicate row IDs")
        self.row = {key: i for i, key in enumerate(keys)}

    def compatible_queries(self, queries):
        if (queries.meta["encoder_id"] != self.meta["encoder_id"] or
                queries.vectors.shape[1] != self.vectors.shape[1]):
            raise ValueError("query/document encoder or dimension mismatch")


class AffineHead:
    def __init__(self, weights, bias):
        self.weights, self.bias = np.asarray(weights), np.asarray(bias)
        if (self.weights.ndim != 2 or self.bias.shape != (len(self.weights),) or
                not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all()):
            raise ValueError("invalid head parameters")

    @classmethod
    def initialize(cls, input_dim, head_dim, seed):
        if not 0 < head_dim <= input_dim:
            raise ValueError("head dimension must be in [1, input dimension]")
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0, 1 / np.sqrt(head_dim), (head_dim, input_dim)).astype(np.float32),
                   np.zeros(head_dim, dtype=np.float32))

    def project(self, unit_vectors):
        raw = unit_vectors @ self.weights.T + self.bias
        return normalize(raw, allow_zero=True)

    def save(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez(directory / "head.npz", weights=self.weights, bias=self.bias)
        write_json(directory / "manifest.json", {
            **metadata, "schema": "affine-cosine-head-v1", "epsilon": EPS,
            "input_dim": self.weights.shape[1], "head_dim": self.weights.shape[0],
            "weights_sha256": sha256(directory / "head.npz"),
        })

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = directory / "manifest.json"
        config = directory / "config.json"
        if manifest.is_file():
            metadata = json.loads(manifest.read_text())
            if metadata.get("schema") != "affine-cosine-head-v1" or metadata.get("epsilon") != EPS:
                raise ValueError("unsupported head schema/normalization")
        elif config.is_file():
            metadata = json.loads(config.read_text())
            if metadata.get("kind") != "shared_affine_cosine_head" or metadata.get("epsilon") != EPS:
                raise ValueError("unsupported distributed head contract")
            metadata = {**metadata, "input_dim": metadata["input_dimension"],
                        "head_dim": metadata["output_dimension"]}
        else:
            raise ValueError("head needs manifest.json or config.json")
        if sha256(directory / "head.npz") != metadata["weights_sha256"]:
            raise ValueError("head checksum mismatch")
        with np.load(directory / "head.npz", allow_pickle=False) as params:
            head = cls(params["weights"], params["bias"])
        if head.weights.shape != (metadata["head_dim"], metadata["input_dim"]):
            raise ValueError("head dimension mismatch")
        return head, metadata


def check_head_cache(head, metadata, documents, allow_new_corpus=False):
    encoder_matches = metadata.get("encoder_id", metadata.get("encoder_contract_id")) == documents.meta["encoder_id"]
    pinned = (metadata.get("document_cache_sha256", metadata.get("original_C_manifest_sha256")) == documents.digest and
              metadata.get("representation_id") == documents.meta["representation_id"])
    compatible_new = (allow_new_corpus and metadata.get("kind") == "shared_affine_cosine_head" and
                      metadata.get("compatible_new_corpus_representation_family") == documents.meta.get("representation_family"))
    if head.weights.shape[1] != documents.vectors.shape[1] or not encoder_matches or not (pinned or compatible_new):
        raise ValueError("head/document encoder, representation, or cache hash mismatch; explicit new-corpus mode requires a matching representation family")


def build_projection(documents, head_directory, output_directory, chunk_size=2048, allow_new_corpus=False):
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    head, metadata = AffineHead.load(head_directory)
    check_head_cache(head, metadata, documents, allow_new_corpus)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    projected = np.lib.format.open_memmap(output / "projected.npy", mode="w+", dtype=np.float32,
                                          shape=(len(documents.items), len(head.weights)))
    min_norm, max_norm, degenerate = float("inf"), 0.0, 0
    for start in range(0, len(projected), chunk_size):
        x = normalize(np.asarray(documents.vectors[start:start + chunk_size]))
        raw = x @ head.weights.T + head.bias
        norms = np.linalg.norm(raw, axis=1)
        min_norm = min(min_norm, float(norms.min()))
        max_norm = max(max_norm, float(norms.max()))
        degenerate += int(np.sum(norms <= EPS))
        projected[start:start + len(x)] = normalize(raw, allow_zero=True)
    projected.flush()
    shape = list(projected.shape)
    del projected
    write_json(output / "manifest.json", {
        "schema": "projected-document-cache-v1", "document_cache_sha256": documents.digest,
        "head_weights_sha256": metadata["weights_sha256"], "shape": shape, "dtype": "float32",
        "projected_sha256": sha256(output / "projected.npy"),
        "raw_head_norm_min": min_norm, "raw_head_norm_max": max_norm,
        "near_zero_rows": degenerate,
    })


def stable_topk(rows, scores, k):
    """Select exact top-k, resolving equal scores by original row ID."""
    if not len(rows):
        return rows, scores
    count = min(k, len(rows))
    threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
    candidates = np.flatnonzero(scores >= threshold)
    order = np.lexsort((rows[candidates], -scores[candidates]))[:count]
    chosen = candidates[order]
    return rows[chosen], scores[chosen]


class ExactRetriever:
    """Drop-in candidate-search boundary; encode queries with the frozen encoder.

    Pass raw encoder vectors to search_vector. Result names and row IDs address
    the bound passage collection; this class performs no reranking or graph expansion.
    """
    def __init__(self, document_manifest, head_directory=None, projection_directory=None,
                 alpha=0.0, chunk_size=2048, allow_new_corpus=False):
        if not 0 <= alpha <= 1 or chunk_size <= 0:
            raise ValueError("invalid alpha/chunk size")
        self.documents = Cache(document_manifest, "documents")
        self.alpha, self.chunk_size = alpha, chunk_size
        self.head = self.projected = None
        if alpha:
            if head_directory is None or projection_directory is None:
                raise ValueError("a nonzero alpha requires both head and document projection")
            self.head, meta = AffineHead.load(head_directory)
            check_head_cache(self.head, meta, self.documents, allow_new_corpus)
            directory = Path(projection_directory)
            derived = json.loads((directory / "manifest.json").read_text())
            if (derived.get("schema") != "projected-document-cache-v1" or
                    derived["document_cache_sha256"] != self.documents.digest or
                    derived["head_weights_sha256"] != meta["weights_sha256"] or
                    sha256(directory / "projected.npy") != derived["projected_sha256"]):
                raise ValueError("stale or modified document projection")
            self.projected = np.load(directory / "projected.npy", mmap_mode="r", allow_pickle=False)
            if (self.projected.dtype != np.float32 or
                    self.projected.shape != (len(self.documents.items), len(self.head.weights)) or
                    list(self.projected.shape) != derived["shape"]):
                raise ValueError("document projection shape/dtype mismatch")

    def search_vector(self, vector, top_k=100):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        q = normalize(np.asarray(vector, dtype=np.float32).reshape(1, -1))
        if q.shape[1] != self.documents.vectors.shape[1]:
            raise ValueError("query dimension mismatch")
        uq = self.head.project(q)[0] if self.alpha else None
        best_rows = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        for start in range(0, len(self.documents.items), self.chunk_size):
            x = normalize(np.asarray(self.documents.vectors[start:start + self.chunk_size]))
            # Rowwise reductions give the same arithmetic regardless of chunk boundaries.
            scores = np.einsum("nd,d->n", x, q[0], optimize=False)
            if self.alpha:
                adapted = np.einsum("nk,k->n", self.projected[start:start + len(x)], uq,
                                     optimize=False)
                scores = (1 - self.alpha) * scores + self.alpha * adapted
            if not np.isfinite(scores).all():
                raise ValueError("nonfinite retrieval score")
            rows = np.arange(start, start + len(x))
            rows, scores = stable_topk(rows, scores, top_k)
            best_rows, best_scores = stable_topk(np.concatenate([best_rows, rows]),
                                                 np.concatenate([best_scores, scores]), top_k)
        return [{"row_id": int(row), "name": self.documents.items[row], "score": float(score)}
                for row, score in zip(best_rows, best_scores)]


class FrozenEncoderSearch:
    """Connect a pinned encoder callback to the candidate-search stage.

    encode_queries([text]) must return a CPU-convertible [1, D] array, using
    exactly the cached encoder's query instruction. Choose search for a sync
    callback or asearch for an async callback. A separately configured reranker
    must look up its passages using the returned name/row_id.
    """
    def __init__(self, retriever, encode_queries, encoder_id):
        if encoder_id != retriever.documents.meta["encoder_id"] or not callable(encode_queries):
            raise ValueError("runtime encoder does not match the frozen cache")
        self.retriever, self.encode_queries = retriever, encode_queries

    def _finish(self, text, vectors, top_k):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (1, self.retriever.documents.vectors.shape[1]):
            raise ValueError("encoder callback must return [1, input_dim] CPU vectors")
        return self.retriever.search_vector(vectors[0], top_k)

    def search(self, text, top_k=200):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty query text")
        return self._finish(text, self.encode_queries([text]), top_k)

    async def asearch(self, text, top_k=200):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty query text")
        return self._finish(text, await self.encode_queries([text]), top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="precompute document-side head outputs")
    build.add_argument("--documents", required=True)
    build.add_argument("--head", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--chunk-size", type=int, default=2048)
    search = sub.add_parser("search", help="exact full-corpus search of cached queries")
    search.add_argument("--documents", required=True)
    search.add_argument("--queries", required=True)
    search.add_argument("--head")
    search.add_argument("--projection")
    search.add_argument("--alpha", type=float, default=0.0)
    search.add_argument("--top-k", type=int, default=200)
    search.add_argument("--chunk-size", type=int, default=2048)
    search.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build":
        build_projection(Cache(args.documents, "documents"), args.head, args.output, args.chunk_size)
    else:
        retriever = ExactRetriever(args.documents, args.head, args.projection, args.alpha, args.chunk_size)
        queries = Cache(args.queries, "queries")
        retriever.documents.compatible_queries(queries)
        with Path(args.output).open("x") as stream:
            for i, item in enumerate(queries.items):
                row = {"query_id": item["query_id"], "query_text_sha256": item["text_sha256"],
                       "results": retriever.search_vector(queries.vectors[i], args.top_k)}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
