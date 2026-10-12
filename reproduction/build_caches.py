"""Assemble the original-text and enriched-text full-corpus caches in corpus order.

Unchanged rows retain the original index's vectors. For each changed interface,
both arms use its paired BF16 re-encoding. This matches the published comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from common import HERE, read, sha, write
from encode import prepare_pairs, shard_values
from lean_context_search.cache import write_json
from lean_context_search.context import canonical_name


def assemble(base, names, pairs, get_pair, output, encoder_id, expected):
    if len(base) != len(names) or len(names) != len(set(names)):
        raise ValueError('base/name row count or uniqueness mismatch')
    if [r['row'] for r in pairs] != sorted({r['row'] for r in pairs}):
        raise ValueError('duplicate or unordered paired rows')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cache_paths = {}
    for arm, key in [('original', 'original'), ('enriched', 'enriched')]:
        directory = output / arm
        directory.mkdir()
        target = np.lib.format.open_memmap(directory / 'vectors.npy', mode='w+', dtype=np.float32, shape=base.shape)
        for start in range(0, len(base), 2048):
            block = np.asarray(base[start:start + 2048])
            if not np.isfinite(block).all() or np.any(np.linalg.norm(block, axis=1) <= 1e-8):
                raise ValueError('invalid imported vector block')
            target[start:start + len(block)] = block
        for row in pairs:
            index = row['row']
            if not 0 <= index < len(names) or names[index] != row['name']:
                raise ValueError('paired row/name mismatch')
            vector = get_pair(row)[key]
            if (vector.shape != (base.shape[1],) or not np.isfinite(vector).all() or
                    abs(np.linalg.norm(vector) - 1) > 1e-4):
                raise ValueError('invalid paired vector')
            target[index] = vector
        target.flush()
        del target
        write_json(directory / 'items.json', names)
        files = {name: {'path': name, 'sha256': sha(directory / name)} for name in ['vectors.npy', 'items.json']}
        meta = {'schema': 'frozen-vector-cache-v1', 'role': 'documents', 'shape': list(base.shape), 'dtype': 'float32',
                'encoder_id': encoder_id, 'representation_id': arm + ':' + files['vectors.npy']['sha256'], 'files': files}
        if arm == 'enriched':
            if files == expected['files'] and list(base.shape) == expected['shape'] and encoder_id == expected['encoder_id']:
                meta = expected
            else:
                meta['representation_family'] = 'complete-context-v1'
        write_json(directory / 'manifest.json', meta)
        cache_paths[arm] = directory / 'manifest.json'
    return cache_paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-vectors', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--changed-passages', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True, help='paired encoder output directory containing shards/')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-numerical-drift', action='store_true', help='explicitly keep a re-encoded cache with a different numerical hash')
    args = p.parse_args()
    spec = read(HERE / 'configs/corpus.json')
    if args.output.exists():
        raise ValueError('cache output exists; choose a new directory')
    if sha(args.base_vectors) != spec['export_sha256'] or sha(args.metadata) != spec['source_metadata_sha256']:
        raise ValueError('imported vectors/metadata do not match the measured corpus')
    with args.metadata.open('rb') as stream:
        metadata = pickle.load(stream)  # Only after checking the registered public artifact hash.
    names = [canonical_name(r) for r in metadata['data']]
    digest = hashlib.sha256(('\n'.join(names) + '\n').encode()).hexdigest()
    if len(names) != spec['rows'] or len(names) != len(set(names)) or digest != spec['ordered_names_sha256']:
        raise ValueError('corpus row order differs')
    pairs = prepare_pairs(args.changed_passages)
    base = np.load(args.base_vectors, mmap_mode='r', allow_pickle=False)
    if base.shape != (len(names), spec['dimension']) or base.dtype != np.float32:
        raise ValueError('base vector layout differs')
    def get_pair(row):
        return shard_values(args.pairs / 'shards' / f"{row['row']:08d}.npz",
            {'row': row['row'], 'name': row['name'], 'original_sha256': row['original_sha256'],
             'enriched_sha256': row['candidate_sha256']}, ['original', 'enriched'], spec['dimension'])
    # Validate every pair before creating large outputs.
    for row in pairs:
        get_pair(row)
    expected_path = HERE / 'reference/enriched_documents_manifest.json'
    expected = read(expected_path)
    paths = assemble(base, names, pairs, get_pair, args.output, expected['encoder_id'], expected)
    exact = sha(paths['enriched']) == sha(expected_path)
    report = {'status': 'passed' if exact or args.allow_numerical_drift else 'numerical_drift_requires_opt_in',
              'rows': len(names), 'reencoded_rows_per_arm': len(pairs), 'unchanged_imported_rows': len(names) - len(pairs),
              'enriched_cache_exactly_matches_historical_manifest': exact,
              'allow_new_corpus_required_for_published_head': not exact,
              'caches': {k: {'path': str(v), 'sha256': sha(v)} for k, v in paths.items()}, 'model_calls': 0}
    write(args.output / 'build_report.json', report)
    print(json.dumps(report, indent=2))
    if not exact and not args.allow_numerical_drift:
        raise ValueError('fresh BF16 vectors differ numerically; outputs retained with new provenance. Review build_report.json; use explicit --allow-reencoded-cache in benchmark.py to evaluate these caches.')


if __name__ == '__main__':
    main()
