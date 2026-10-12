"""Public single-designated-target metrics; other returned declarations are unjudged."""
import json
import math
import hashlib
from pathlib import Path


METRICS = ('ndcg@1','ndcg@5','ndcg@10','recall@10','recall@50','recall@100','recall@200')


def target_metrics(rank):
    return {name: (1 / math.log2(rank + 1) if name.startswith('ndcg') else 1.0)
            if rank is not None and rank <= int(name.split('@')[1]) else 0.0
            for name in METRICS}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate(rankings, labels):
    if not labels:
        raise ValueError('evaluation requires at least one labeled query')
    known = {r['query_id']: r for r in labels}
    if len(known) != len(labels):
        raise ValueError('duplicate label query IDs')
    for label in labels:
        target = label.get('target_row')
        if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target < 0):
            raise ValueError('invalid target row ID')
        if not isinstance(label['fair'], bool):
            raise ValueError('fair subset membership must be boolean')
    seen = set()
    rows = []
    for ranking in rankings:
        qid = ranking['query_id']
        if qid in seen or qid not in known:
            raise ValueError('duplicate/unknown ranking query')
        seen.add(qid)
        label = known[qid]
        label_hash = label.get('query_text_sha256') or hashlib.sha256(label['query_text'].encode()).hexdigest()
        if ranking.get('query_text_sha256') != label_hash:
            raise ValueError('query text hash mismatch')
        results = ranking['results']
        ids = [r['row_id'] for r in results]
        if len(set(ids)) != len(ids) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
            raise ValueError('duplicate/invalid ranked row IDs')
        target = label.get('target_row')
        rank = ids.index(target) + 1 if target in ids else None
        rows.append({'query_id': qid, 'fair': bool(label['fair']), 'target_rank': rank,
                     'target_name': label['target_name'], 'metrics': target_metrics(rank)})
    if seen != set(known):
        raise ValueError('missing ranking queries')
    def aggregate(items):
        return {'queries': len(items), 'targets_missing_from_corpus': sum(known[x['query_id']].get('target_row') is None for x in items),
                **{k: sum(x['metrics'][k] for x in items) / len(items) if items else None for k in METRICS}}
    return {'schema':'lean-context-designated-target-evaluation-v1',
            'full':aggregate(rows), 'fair':aggregate([x for x in rows if x['fair']]), 'per_query':rows}


def replay_saved(evidence_path, canonical_path):
    evidence = read_jsonl(evidence_path)
    canonical = json.loads(Path(canonical_path).read_text())
    if len(evidence) != 946 or sum(x['fair'] for x in evidence) != 810:
        raise ValueError('public evidence size/fair subset mismatch')
    largest = 0.0
    comparisons = 0
    for subset, info in canonical['sets'].items():
        records = evidence if subset == 'full946' else [x for x in evidence if x['fair']]
        if len(records) != info['query_count']:
            raise ValueError('subset denominator mismatch')
        for arm, metrics in info['metrics'].items():
            for metric, expected in metrics.items():
                if '@' not in metric:
                    continue
                actual = sum(target_metrics(row['arms'][arm]['target_rank'])[metric] for row in records) / len(records)
                largest = max(largest, abs(actual - expected))
                comparisons += 1
                if abs(actual - expected) > 1e-12:
                    raise ValueError(f'canonical replay mismatch {subset}/{arm}/{metric}')
    return {'status':'passed','queries':946,'fair_queries':810,'aggregate_comparisons':comparisons,
            'max_absolute_error':largest,'model_calls':0}
