# Third-party notices

- **LeanSearch-v2:** upstream Apache-2.0 text is preserved in [its notice](notices/LEANSEARCH_V2_APACHE_2_0.txt). Public designated-target queries and saved results originate from its public task.
- **Lean / Mathlib:** the context exporter targets Lean 4.28.0-rc1 and Mathlib commit `5352afccd6866369be9de43f5b7ec47203555f44`. The source checkout's [Apache-2.0 notice](notices/MATHLIB_APACHE_2_0.txt) is retained. Five abbreviated passage examples are included in [context examples](examples/context_examples.md); the original index commit is unresolved.
- **D3 7.9.0:** the offline website includes D3 with its [license](studies/cones/vendor/D3_LICENSE.txt).
- **Qwen3-Embedding-8B / Qwen3-Reranker-8B:** external model weights. The reproduction inputs include frozen Qwen embedding vectors for 88 training queries and 696 Mathlib declaration passages. These are exact slices of the recorded training caches.
- **Relevance annotations:** the reproduction inputs contain the existing Codex-agent-authored training queries and provisional reviews, plus accepted development answers. These have no human adjudication. Public benchmark queries retain their LeanSearch-v2 attribution above.

The full index, model backbones, full-corpus vector caches and protected confirmation data are not distributed here. These notices do not assign a license to the newly authored code or trained heads, or establish separate redistribution rights for the indexed excerpts and public-task evidence.
