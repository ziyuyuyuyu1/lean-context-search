# Saved results

| File | Purpose |
| --- | --- |
| [public_metrics.json](public_metrics.json) | Aggregate search metrics for all 946 public queries and the predefined 810-query shared subset, including the completed comparison arms. `release_configuration` identifies the current search setting. |
| [public_per_query.jsonl](public_per_query.jsonl) | Per-query target ranks, subset membership and comparison results used to replay the public metrics. Other relevant answers are unjudged. |
| [structural_metrics.json](structural_metrics.json) | Counts and scope of the cone fitting study; used to check and build the structural figures. These are not public search scores. |

The README table uses the 810-query subset. Both evaluation sets remain in the saved evidence.

To verify the public aggregates without loading a model, run from the repository root:

```bash
lean-context-search replay --evidence results/public_per_query.jsonl \
  --canonical results/public_metrics.json
```
