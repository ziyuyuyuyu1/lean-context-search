"""Regenerate the four README comparison rows through the installed search CLI.

Each retrieval arm searches the full corpus. Qwen reorders the embedding head's
own top 200. Subprocesses are sequential; the only optional GPU phase is reranking.
Use --features for existing portable Qwen margins, or --model for fresh inference.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from common import ROOT, read, rows, sha, write, write_rows
from lean_context_search.cache import Cache, write_cache, text_hash
from lean_context_search.metrics import evaluate

ARMS = {
    'original_retrieval': ('Base retrieval · original text', 'B_original_reencoded'),
    'enriched_retrieval': ('Base retrieval · enriched text', 'C_enriched'),
    'embedding_head': ('Embedding-head retrieval · enriched text', 'C_E0_retrieval'),
    'qwen_reranked': ('Embedding-head retrieval + Qwen reranking · enriched text', 'enriched_margin'),
}


def plan(args):
    prefix = [sys.executable, '-B', '-m', 'lean_context_search']
    output = args.output
    query_cache = output / 'queries/manifest.json'
    projection = args.projection or output / 'projection'
    commands = []
    if not args.projection:
        command = prefix + ['project', '--documents', str(args.enriched_documents), '--head', str(args.head), '--output', str(projection)]
        if args.allow_reencoded_cache:
            command.append('--allow-new-corpus')
        commands.append(('project', command))
    for arm, documents, alpha in [('original_retrieval', args.original_documents, '0'),
                                  ('enriched_retrieval', args.enriched_documents, '0'),
                                  ('embedding_head', args.enriched_documents, '0.30')]:
        command = prefix + ['retrieve', '--documents', str(documents), '--queries', str(query_cache),
                            '--alpha', alpha, '--top-k', '200', '--output', str(output / (arm + '.jsonl'))]
        if arm == 'embedding_head':
            command += ['--head', str(args.head), '--projection', str(projection)]
            if args.allow_reencoded_cache:
                command.append('--allow-new-corpus')
        commands.append((arm, command))
    rerank = prefix + ['rerank-cached' if args.features else 'rerank',
                       '--pools', str(output / 'embedding_head.jsonl'), '--passages', str(args.passages),
                       '--enriched-passages', str(args.enriched_passages), '--mode', 'margin', '--beta', '0',
                       '--return-k', '200', '--output', str(output / 'qwen_reranked.jsonl')]
    if args.features:
        rerank += ['--features', str(output / 'selected_features.jsonl')]
    else:
        rerank += ['--query-file', str(output / 'query_texts.jsonl'), '--model', str(args.model),
                   '--device', args.device, '--microbatch', str(args.microbatch)]
    commands.append(('qwen_reranked', rerank))
    return commands


def filter_features(path, pools, output):
    needed = {(p['query_id'], r['row_id']): p['query_text_sha256'] for p in pools for r in p['results']}
    found = {}
    with Path(path).open() as stream:
        for line in stream:
            value = json.loads(line)
            key = (value['query_id'], value['row_id'])
            if key in needed:
                if key in found or value['query_text_sha256'] != needed[key]:
                    raise ValueError('duplicate or mismatched cached feature')
                found[key] = value
    if set(found) != set(needed):
        raise ValueError(f'missing {len(set(needed) - set(found))} requested Qwen pairs; no automatic neural fallback')
    write_rows(output, [found[(p['query_id'], r['row_id'])] for p in pools for r in p['results']])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-documents', type=Path, required=True)
    p.add_argument('--enriched-documents', type=Path, required=True)
    p.add_argument('--queries', type=Path, required=True)
    p.add_argument('--passages', type=Path, required=True)
    p.add_argument('--enriched-passages', type=Path, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--features', type=Path, help='portable query/row/text-bound efficient margins')
    mode.add_argument('--model', type=Path, help='local pinned Qwen3-Reranker-8B; enables GPU inference')
    p.add_argument('--head', type=Path, default=ROOT / 'heads/e0')
    p.add_argument('--projection', type=Path, help='reuse an already verified full-corpus head projection')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--query-ids', nargs='+', help='explicit small diagnostic; omitted means all 946 queries')
    p.add_argument('--allow-reencoded-cache', action='store_true', help='evaluate numerically different rebuilt corpus vectors with explicit compatible-corpus provenance')
    p.add_argument('--plan', action='store_true', help='print commands; do not read caches, write files or run models')
    args = p.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if not 1 <= args.microbatch <= 4:
        p.error('microbatch must be between 1 and 4')
    commands = plan(args)
    if args.plan:
        print(json.dumps({'sequential_commands': commands, 'neural_inference_requested': bool(args.model),
                          'query_scope': args.query_ids or 'all 946 public queries', 'training': False}, indent=2))
        return
    if args.output.exists():
        raise ValueError('output exists; choose a new directory')
    evidence = rows(ROOT / 'results/public_per_query.jsonl')
    if len(evidence) != 946 or sum(r['fair'] for r in evidence) != 810:
        raise ValueError('public evidence count differs')
    requested = set(args.query_ids) if args.query_ids else {r['query_id'] for r in evidence}
    if len(requested) != len(args.query_ids or evidence) or requested - {r['query_id'] for r in evidence}:
        raise ValueError('duplicate or unknown requested query IDs')
    labels = [r for r in evidence if r['query_id'] in requested]
    queries = Cache(args.queries, 'queries')
    indices = []
    items = []
    for label in labels:
        index = queries.row[label['query_id']]
        item = queries.items[index]
        if item['text_sha256'] != text_hash(label['query_text']) or item['split'] != 'benchmark':
            raise ValueError('public query cache binding differs')
        indices.append(index)
        items.append(item)
    # Corpus row identity is independently validated before target IDs are used.
    original = Cache(args.original_documents, 'documents')
    enriched = Cache(args.enriched_documents, 'documents')
    if original.items != enriched.items or len(enriched.items) != 310579:
        raise ValueError('comparison arms need identical complete corpus row order')
    for label in labels:
        expected = enriched.row.get(label['target_name'])
        if expected != label['target_row']:
            raise ValueError('published designated target row differs from corpus order')
    args.output.mkdir(parents=True)
    (args.output / 'logs').mkdir()
    write_cache(args.output / 'queries', np.asarray(queries.vectors[indices]), items, 'queries',
                queries.meta['encoder_id'], queries.meta['representation_id'])
    write_rows(args.output / 'query_texts.jsonl', [{'query_id': l['query_id'], 'query_text': l['query_text'],
               'query_text_sha256': text_hash(l['query_text'])} for l in labels])
    write(args.output / 'input_lock.json', {'query_ids': [l['query_id'] for l in labels], 'alpha': .3,
          'reranker_mode': 'margin', 'reranker_head': None, 'pool_size': 200, 'returned_size': 100,
          'head_sha256': sha(args.head / 'head.npz'), 'corpus_manifests': {k: sha(getattr(args, k)) for k in ['original_documents', 'enriched_documents']},
          'query_manifest_sha256': sha(args.queries), 'commands': commands,
          'benchmark_role': 'previously inspected public benchmark', 'protected_confirmation_access': False})
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    for name, command in commands:
        if name == 'qwen_reranked' and args.features:
            filter_features(args.features, rows(args.output / 'embedding_head.jsonl'), args.output / 'selected_features.jsonl')
        started = time.monotonic()
        log = args.output / 'logs' / (name + '.log')
        with log.open('w') as stream:
            done = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record = {'name': name, 'argv': command, 'cwd': str(ROOT), 'exit_code': done.returncode,
                  'wall_seconds': time.monotonic() - started, 'log': str(log), 'log_sha256': sha(log),
                  'GPU_phase': name == 'qwen_reranked' and bool(args.model)}
        with (args.output / 'commands.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if done.returncode:
            raise RuntimeError('stage failed; retained ' + str(log))
    metrics = {}
    mismatches = []
    for arm, (_, saved_key) in ARMS.items():
        rankings = rows(args.output / (arm + '.jsonl'))
        metrics[arm] = evaluate(rankings, labels)
        write(args.output / (arm + '_metrics.json'), metrics[arm])
        old_by_id = {r['query_id']: r['arms'][saved_key]['target_rank'] for r in labels}
        for row in metrics[arm]['per_query']:
            if row['target_rank'] != old_by_id[row['query_id']]:
                mismatches.append({'arm': arm, 'query_id': row['query_id'], 'old_rank': old_by_id[row['query_id']], 'new_rank': row['target_rank']})
    reranked = rows(args.output / 'qwen_reranked.jsonl')
    write_rows(args.output / 'returned_top100.jsonl', [{**r, 'results': r['results'][:100]} for r in reranked])
    table = ['| Pipeline | Queries | nDCG@10 | Recall@10 | Recall@100 |', '|---|---:|---:|---:|---:|']
    for cohort in ['fair', 'full']:
        for arm, (name, _) in ARMS.items():
            values = metrics[arm][cohort]
            if values['queries']:
                table.append(f"| {name} ({cohort}) | {values['queries']} | {values['ndcg@10']:.6f} | {values['recall@10']:.6f} | {values['recall@100']:.6f} |")
    (args.output / 'comparison.md').write_text('\n'.join(table) + '\n')
    write(args.output / 'summary.json', {'status': 'completed', 'queries': len(labels), 'full_benchmark': len(labels) == 946,
          'input_scope': 'new ranking execution over existing query/document vectors',
          'Qwen_inference': bool(args.model), 'Qwen_feature_reuse': bool(args.features),
          'reference_rank_mismatches': mismatches, 'metrics': {k: {s: v[s] for s in ['full', 'fair']} for k, v in metrics.items()},
          'note': 'All 200 candidate ranks are retained for pool diagnostics; returned_top100.jsonl is the deployed output. Fresh numerical environments may differ; changes are reported, never replaced with saved ranks.'})
    print('\n'.join(table))
    print(json.dumps({'output': str(args.output), 'queries': len(labels), 'reference_rank_mismatches': len(mismatches)}))


if __name__ == '__main__':
    main()
