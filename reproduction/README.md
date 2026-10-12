# Reproduce Lean Context Search

Run commands from the repository root. Use a clone or source archive: the wheel contains the search runtime; this directory supplies the experiment scripts and data.

| Task | Inputs | Compute |
| --- | --- | --- |
| Replay reported metrics | Included public rankings | CPU |
| Train the embedding head | Included queries, labels and frozen vectors | CPU |
| Rebuild the corpus and query caches | Pinned index, Mathlib and embedding model | CPU + one BF16 GPU |
| Run the four benchmark pipelines | Full caches and frozen Qwen reranker | CPU + one BF16 GPU |

Outputs go under `work/`. Existing outputs are preserved; choose a new destination when starting another run. The trained head in `heads/e0` and all release defaults stay unchanged.

## 1. Reproduce head training on CPU

```bash
python -m pip install -e .
python reproduction/train_embedding_head.py --check-only
python reproduction/train_embedding_head.py --output work/retrained-head
```

The second command checks every input and split without training. The third runs the recorded **100 AdamW updates**, seed `20261001`, on the shared **4096→128 affine head**. It uses the same loss, family/style sampler, initialization, clipping and optimizer operations as the measured run. The score is `0.70 × base cosine + 0.30 × head cosine`.

Saved checkpoints are at updates 0, 25, 50 and 100. The output contains losses, diagnostics, configuration and a parameter comparison with the distributed head. NumPy/BLAS differences can change rounding; parameter differences are reported explicitly.

**Included data (~13 MB):**

- `data/train.jsonl`: 88 queries, forming 44 statement/task pairs from 43 source families; accepted positives, reviewed negatives and masked uncertain alternatives.
- `data/queries/`: exact frozen vectors for those 88 queries.
- `data/documents/`: lossless slices for all 696 distinct positive/negative documents used by the loss. These use the enriched corpus representation. Negatives span other eligible training families; 43 counts query-source families only.
- `data/development.jsonl`: the existing 87 development queries and accepted answers, for evaluation only. The trainer never reads their query content. Candidate review packets are omitted.
- `data/family_policy.json` and `benchmark_exclusions.json`: the relevant historical family assignments and full public-target exclusion list. Unlisted declarations cannot enter training.
- `data/document_rows.json`, `data_manifest.json`: original corpus row mapping and checksums.

Queries and labels come from the existing **Codex-agent, LLM-only provisional review**, with no human adjudication or new generation. Masked alternatives never enter the loss. Original protected confirmation data is excluded. Development and public results were inspected during research; they are not fresh confirmation.

To evaluate a newly trained head, supply `--head work/retrained-head/checkpoint_100` to the benchmark script below. The default uses the distributed measured head.

## 2. Obtain the measured external assets

```bash
python -m pip install -e '.[inference]'
python reproduction/fetch_assets.py --asset corpus --plan
python reproduction/fetch_assets.py --asset corpus --output assets/corpus
python reproduction/fetch_assets.py --asset embedding --output assets/embedding
python reproduction/fetch_assets.py --asset reranker --output assets/reranker
```

Downloads are unauthenticated and verified against pinned file hashes before use. To check existing files without downloading, append `--verify-only`. The inference extra pins SentenceTransformers **5.2.2**, Transformers **4.57.6** and Tokenizers **0.22.2**; runtime validation checks model, tokenizer, pooling and prompt metadata before constructing a model.

| Source | Pinned revision | Download |
| --- | --- | ---: |
| [LeanSearch-v2 index](https://huggingface.co/datasets/FrenzyMath/lsv2-mathlib-v4.28.0-rc1-cuvs) | `c3757ad6c491e1afa3cc293a61119f1ff31292a0` | 5.74 GB |
| [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` | 15.15 GB |
| [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | `77d193c791ed757ca307ee72715aa132723da912` | 16.39 GB |

These revisions contain files matching the recorded artifacts. They do not retroactively establish the original index's Mathlib source commit. Historical feature and cache IDs are preserved.

Allow roughly **60 GB plus the Lean/Mathlib checkout and its build cache**: downloads total 37.3 GB; exported vectors and the two assembled caches add 15.3 GB. GPU phases run separately and release their models between processes. The recorded encoding used an NVIDIA A10; fresh inference needs a compatible BF16 CUDA environment and enough memory for one 8B model plus activations. Runtime varies with query and passage lengths.

## 3. Rebuild the typed context and paired caches

### Export the original index vectors

Use a separate environment with **cuVS 25.12.00** and its compatible CUDA dependencies. cuVS is only required for this export; it is separate from the inference extra. The exported array must match the recorded checksum.

```bash
python reproduction/export_index.py \
  --index assets/corpus/cuvs_index.bin --output work/base_vectors.npy
```

An existing verified `base_vectors.npy` can be reused. The exported rows must retain their original order.

### Extract and render context

With `git`, Lean's `elan` toolchain manager and the inference environment available:

```bash
git clone https://github.com/leanprover-community/mathlib4.git work/mathlib
git -C work/mathlib checkout 5352afccd6866369be9de43f5b7ec47203555f44
(cd work/mathlib && lake exe cache get)

lean-context-search prepare-context --db-dir assets/corpus \
  --output-dir work/context --imports Mathlib \
  --snapshot 5352afccd6866369be9de43f5b7ec47203555f44
(cd work/mathlib && lake env lean ../context/export.lean)

lean-context-search render-context --prepared-dir work/context \
  --output-dir work/passages --tokenizer-path assets/embedding \
  --max-tokens 40960 --allow-unresolved-corpus-names
```

The snapshot uses Lean **4.28.0-rc1**. The unresolved-name flag records the historical limitation: 883 indexed non-interface declarations were unresolved in this context snapshot. Extraction must still have zero interface errors. The next stage requires the exact registered passage file: **3,222 changed rows**; all other 307,357 rows retain their imported vectors.

### Encode both versions of changed passages and the public queries

```bash
python reproduction/encode.py pairs --model assets/embedding \
  --changed-passages work/passages/changed_passages.jsonl --output work/pairs

python reproduction/encode.py queries --model assets/embedding \
  --records results/public_per_query.jsonl --split benchmark --output work/public-queries

python reproduction/build_caches.py --base-vectors work/base_vectors.npy \
  --metadata assets/corpus/metadata.pkl \
  --changed-passages work/passages/changed_passages.jsonl \
  --pairs work/pairs --output work/corpus
```

Encoding uses BF16, batch size one, the recorded query instruction and a 40,960-token embedding limit. It rejects token overflow. Add `--check-only` to validate text inputs without loading a model, or `--max-new 5` for a bounded start. Continue the same job with `--resume`; input hashes and saved shards must match.

The original-text arm also uses fresh original-text vectors for the 3,222 changed rows. Re-encoding every declaration would produce a different comparison. The builder preserves the measured enriched-cache identity only when all bytes match. For fresh numerical differences, inspect the retained `build_report.json`; `--allow-numerical-drift` explicitly accepts them under a new cache identity. Evaluation then also requires `--allow-reencoded-cache` with the distributed head. A newly trained head remains bound to the historical full cache.

## 4. Run the public comparison

```bash
python reproduction/benchmark.py \
  --original-documents work/corpus/original/manifest.json \
  --enriched-documents work/corpus/enriched/manifest.json \
  --queries work/public-queries/cache/manifest.json \
  --passages assets/corpus/texts.pkl \
  --enriched-passages work/passages/changed_passages.jsonl \
  --model assets/reranker --output work/benchmark
```

Append `--plan` to preview the sequential commands without running them. For a bounded diagnostic, add `--query-ids O1_q1a_lean AG5_q1b_latex`. Omitting this option runs **all 946 queries**, and reports the predefined **810-query shared subset** alongside them.

The script searches all **310,579 declarations** independently for each retrieval arm:

1. Original-text base retrieval.
2. Enriched-text base retrieval.
3. Enriched-text retrieval with the frozen embedding head, weight 0.30.
4. Frozen Qwen reranking of arm 3's own top 200, using enriched passages and the yes-minus-no margin. No reranker head is applied.

The output includes `comparison.md`, per-query rankings and metrics, `summary.json`, input locks, commands and logs. It retains all 200 ranks for diagnostics and writes the deployed output separately as `returned_top100.jsonl`. Any differences from saved reference ranks are reported.

To reuse available Qwen scores, replace `--model assets/reranker` with `--features path/to/features.jsonl`. Each line must bind `query_id`, `query_text_sha256`, `row_id`, `passage_sha256`, `feature_contract_id` and `efficient_margin` to the measured enriched-passage contract. Missing or duplicate pairs cause an error; the script never silently starts inference. Full-corpus caches and Qwen score files are external assets.

For the 87 development queries, encode `reproduction/data/development.jsonl` separately with `--split development`, then retrieve their top 200 with `lean-context-search retrieve`. Evaluate these rankings with:

```bash
python reproduction/evaluate_development.py \
  --rankings work/development-rankings.jsonl --output work/development-metrics.json
```

This evaluator uses the historical binary acceptable-answer nDCG, Recall and Hit metrics. Development queries are never mixed into the public denominator.

## Checks and saved-result replay

```bash
python -m unittest discover -s tests -v
lean-context-search replay --evidence results/public_per_query.jsonl \
  --canonical results/public_metrics.json
node studies/cones/check_gallery_js.cjs
```

The tests include synthetic gradient, sampler, resume, feature-binding and cache-assembly checks, plus validation of the real bundled training data. Metric replay recomputes scores from saved ranks. Geometry replay checks the saved cone study. Neither replay reruns model inference. The separate cone fitting/explorer code remains under `studies/cones/`.
