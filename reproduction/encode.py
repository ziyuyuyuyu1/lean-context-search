"""Build the measured BF16 query vectors or paired original/enriched row vectors.

One model, one GPU, batch size one. Completed row shards support explicit resume.
The paired document job encodes only the 3,222 changed rows; other corpus vectors
are imported unchanged from the original index, as in the reported experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import HERE, ROOT, read, rows, sha, write
from lean_context_search.cache import Cache, normalize, write_cache
from lean_context_search.encoder import validate_encoder_contract
from lean_context_search.runtime_assets import validate_runtime_assets


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_encoder(path, contract, device):
    validate_encoder_contract(contract)
    validate_runtime_assets(path, 'embedding', contract)
    for name, digest in contract['model_file_hashes'].items():
        if sha(Path(path) / name) != digest:
            raise ValueError('model checksum mismatch: ' + name)
    import torch
    from sentence_transformers import SentenceTransformer
    if not device.startswith('cuda') or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('the measured encoding path requires a BF16 CUDA device')
    model = SentenceTransformer(str(path), device=device, local_files_only=True,
                               model_kwargs={'torch_dtype': torch.bfloat16})
    model.eval()
    if model.max_seq_length != contract['embedding_limit_tokens'] or {str(p.dtype) for p in model.parameters()} != {'torch.bfloat16'}:
        raise ValueError('model precision/token limit mismatch')
    return model


def encode_one(model, text, contract):
    tokens = model.tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
    actual = model.tokenize([text])['input_ids'][0].tolist()
    if tokens != actual or len(tokens) > contract['embedding_limit_tokens']:
        raise ValueError('runtime tokenization differs or passage overflows')
    vector = np.asarray(model.encode([text], batch_size=1, convert_to_numpy=True,
                                    show_progress_bar=False)[0], dtype=np.float32)
    if vector.shape != (contract['dimension'],):
        raise ValueError('encoder output dimension mismatch')
    normalize(vector[None])  # finite/nonzero check
    return vector / np.linalg.norm(vector), len(tokens)


def prepare_queries(records, split, contract):
    if not records or len({r['query_id'] for r in records}) != len(records):
        raise ValueError('empty/duplicate query IDs')
    if split not in {'train', 'development', 'benchmark'}:
        raise ValueError('unsupported reproduction query split')
    prepared = []
    for index, row in enumerate(records):
        text = row['query_text']
        if not isinstance(text, str) or not text.strip() or row.get('split', split) != split:
            raise ValueError('empty query or split mismatch')
        digest = text_hash(text)
        if row.get('query_text_sha256', digest) != digest:
            raise ValueError('query text checksum mismatch')
        formatted = contract['query_format'].format(instruction=contract['query_instruction'], text=text)
        prepared.append({'index': index, 'query_id': row['query_id'], 'text_sha256': digest,
                         'split': split, 'formatted': formatted})
    return prepared


def prepare_pairs(path):
    spec = read(HERE / 'configs/corpus.json')
    if sha(path) != spec['changed_passages_sha256']:
        raise ValueError('changed passage registry differs from the measured corpus')
    records = rows(path)
    if len(records) != spec['changed_rows'] or [r['row'] for r in records] != sorted({r['row'] for r in records}):
        raise ValueError('paired row count/order mismatch')
    for row in records:
        if text_hash(row['original_text']) != row['original_sha256'] or text_hash(row['enriched_text']) != row['candidate_sha256']:
            raise ValueError('paired passage checksum mismatch')
    return records


def shard_values(path, expected, vector_keys, dimension):
    with np.load(path, allow_pickle=False) as data:
        for key, value in expected.items():
            if data[key].item() != value:
                raise ValueError('resume shard input mismatch: ' + str(path))
        result = {}
        for key in vector_keys:
            v = data[key]
            if v.dtype != np.float32 or v.shape != (dimension,) or not np.isfinite(v).all() or abs(np.linalg.norm(v) - 1) > 1e-4:
                raise ValueError('resume shard vector mismatch: ' + str(path))
            result[key] = v.copy()
        return result


def run(args):
    contract = read(ROOT / 'configs/encoder_contract.json')
    validate_encoder_contract(contract)
    if args.max_new < 0:
        raise ValueError('max-new must be nonnegative')
    input_path = args.records if args.kind == 'queries' else args.changed_passages
    records = prepare_queries(rows(input_path), args.split, contract) if args.kind == 'queries' else prepare_pairs(input_path)
    lock = {'kind': args.kind, 'input_sha256': sha(input_path), 'rows': len(records),
            'encoder_contract_sha256': sha(ROOT / 'configs/encoder_contract.json'),
            'split': args.split if args.kind == 'queries' else None, 'batch_size': 1}
    if args.check_only:
        print(json.dumps({'status': 'passed', 'input_lock': lock, 'model_calls': 0}, indent=2))
        return
    if args.output.exists():
        if not args.resume or read(args.output / 'input_lock.json') != lock:
            raise ValueError('output exists; --resume requires identical inputs')
    else:
        if args.resume:
            raise ValueError('resume output does not exist')
        args.output.mkdir(parents=True)
        (args.output / 'shards').mkdir()
        write(args.output / 'input_lock.json', lock)
    pending = []
    for row in records:
        key = row['index'] if args.kind == 'queries' else row['row']
        path = args.output / 'shards' / f'{key:08d}.npz'
        expected = ({'query_id': row['query_id'], 'text_sha256': row['text_sha256']}
                    if args.kind == 'queries' else {'row': row['row'], 'name': row['name'],
                    'original_sha256': row['original_sha256'], 'enriched_sha256': row['candidate_sha256']})
        keys = ['vector'] if args.kind == 'queries' else ['original', 'enriched']
        if path.exists():
            shard_values(path, expected, keys, contract['dimension'])
        else:
            pending.append((row, path, expected))
    selected = pending[:args.max_new] if args.max_new else pending
    if selected:
        model = load_encoder(args.model, contract, args.device)
        for number, (row, path, expected) in enumerate(selected, 1):
            if args.kind == 'queries':
                v, count = encode_one(model, row['formatted'], contract)
                values = {'vector': v, 'tokens': count}
            else:
                original, n_original = encode_one(model, row['original_text'], contract)
                enriched, n_enriched = encode_one(model, row['enriched_text'], contract)
                if (n_original, n_enriched) != (row['original_tokens'], row['complete_tokens']):
                    raise ValueError('rendered/runtime passage token count mismatch')
                values = {'original': original, 'enriched': enriched,
                          'original_tokens': n_original, 'enriched_tokens': n_enriched}
            temporary = path.with_suffix('.part')
            with temporary.open('xb') as stream:
                np.savez_compressed(stream, **expected, **values)
            temporary.rename(path)
            print(json.dumps({'new_rows_completed': number, 'remaining_this_invocation': len(selected) - number}), flush=True)
        del model
    complete = len(selected) == len(pending)
    if complete and args.kind == 'queries':
        vectors = [shard_values(args.output / 'shards' / f"{r['index']:08d}.npz",
                   {'query_id': r['query_id'], 'text_sha256': r['text_sha256']}, ['vector'], contract['dimension'])['vector'] for r in records]
        items = [{k: r[k] for k in ['query_id', 'text_sha256', 'split']} for r in records]
        representation = read(HERE / 'reference/enriched_documents_manifest.json')['representation_id']
        if (args.output / 'cache').exists():
            cached = Cache(args.output / 'cache/manifest.json', 'queries')
            if (cached.items != items or cached.meta['encoder_id'] != contract['source_contract_sha256'] or
                    cached.meta['representation_id'] != representation or
                    not np.array_equal(cached.vectors, np.stack(vectors))):
                raise ValueError('completed query cache differs from its input shards')
        else:
            write_cache(args.output / 'cache', np.stack(vectors), items, 'queries',
                        contract['source_contract_sha256'], representation)
    receipt = {**lock, 'status': 'complete' if complete else 'partial',
               'rows_completed': len(records) - len(pending) + len(selected), 'new_rows_encoded': len(selected),
               'shards': {p.name: sha(p) for p in sorted((args.output / 'shards').glob('*.npz'))}}
    write(args.output / 'encoding_report.json', receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind', required=True)
    for kind in ['queries', 'pairs']:
        command = sub.add_parser(kind)
        command.add_argument('--model', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--resume', action='store_true')
        command.add_argument('--max-new', type=int, default=0, help='0 means all remaining rows')
        command.add_argument('--check-only', action='store_true', help='validate text inputs without loading a model')
        if kind == 'queries':
            command.add_argument('--records', type=Path, required=True)
            command.add_argument('--split', choices=['train', 'development', 'benchmark'], required=True)
        else:
            command.add_argument('--changed-passages', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
