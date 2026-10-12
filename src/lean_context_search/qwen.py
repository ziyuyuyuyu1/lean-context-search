"""Optional local Qwen scoring; imports Torch/Transformers only on live calls."""
import hashlib
import json
import math
import pickle
from importlib.resources import files
from pathlib import Path

import numpy as np

from .heads import load_distribution_head, qwen_scores
from .qwen_adapter import PREFIX, SUFFIX, prepare_inputs, extract_pair_features
from .runtime_assets import validate_runtime_assets


ORIGINAL_FEATURE_ID = 'f82112f61b7bd554e14dfde8ed8e586f8073508628c2bf1771f346918fc23984'
ENRICHED_FEATURE_ID = 'db4d3cb620852e93be55f1875217800a9d4ca3d90857208456746373f4def601'
_CONTRACT_FILES = {ORIGINAL_FEATURE_ID: 'qwen_feature_contract.json',
                   ENRICHED_FEATURE_ID: 'qwen_enriched_feature_contract.json'}


def default_feature_contract():
    return json.loads(files('lean_context_search').joinpath(_CONTRACT_FILES[ENRICHED_FEATURE_ID]).read_text())


class VerifiedPassages(tuple):
    """Immutable passage texts returned by the bound local-asset loader."""
    def __new__(cls, passages, feature_contract_id, changed_rows=()):
        value = super().__new__(cls, passages)
        value.feature_contract_id = feature_contract_id
        value.changed_rows = frozenset(changed_rows)
        return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_feature_contract(contract, require_full=False):
    """Check the frozen feature definition before reading a cache or model.

    ID-only metadata remains usable for already extracted, row-bound features.
    A full caller-supplied contract must agree with the installed release.
    The legacy pipeline_sha256 describes the original pipeline's provenance;
    it is not a checksum of this portable implementation.
    """
    if contract is None:
        contract = default_feature_contract()
    if not isinstance(contract, dict):
        raise ValueError('invalid Qwen feature contract')
    name = _CONTRACT_FILES.get(contract.get('feature_contract_id'))
    if name is None:
        raise ValueError('Qwen feature contract digest mismatch; unregistered contract')
    canonical = json.loads(files('lean_context_search').joinpath(name).read_text())
    if not require_full and contract.keys() == {'feature_contract_id'}:
        if contract['feature_contract_id'] != canonical['feature_contract_id']:
            raise ValueError('saved Qwen feature contract mismatch')
        return canonical
    if contract.keys() != canonical.keys():
        raise ValueError('Qwen feature contract fields differ from validated release')
    fc = contract.get('feature_contract')
    prompt = contract.get('prompt')
    if not isinstance(fc, dict) or not isinstance(prompt, dict):
        raise ValueError('invalid Qwen feature contract fields')
    digest = hashlib.sha256(json.dumps(fc, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != contract.get('feature_contract_id') or digest != canonical['feature_contract_id']:
        raise ValueError('Qwen feature contract digest mismatch')
    if fc != canonical['feature_contract'] or prompt != canonical['prompt']:
        raise ValueError('Qwen feature settings or prompt differ from validated release')
    if contract.get('passage_inputs') != canonical.get('passage_inputs'):
        raise ValueError('Qwen passage input binding differs from validated release')
    for name, value in [('instruction', prompt['instruction']), ('prefix', PREFIX), ('suffix', SUFFIX)]:
        if hashlib.sha256(value.encode()).hexdigest() != fc[f'{name}_sha256']:
            raise ValueError(f'Qwen actual {name} hash mismatch')
    if prompt['prefix'] != PREFIX or prompt['suffix'] != SUFFIX:
        raise ValueError('Qwen template differs from validated adapter')
    if fc['max_length'] - prompt['prefix_tokens'] - prompt['suffix_tokens'] != fc['body_max_length']:
        raise ValueError('Qwen declared body length mismatch')
    return canonical


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{label} must be a positive integer')


def _score_settings(mode, beta, return_k, microbatch=None):
    if mode not in {'trained', 'margin', 'stock'}:
        raise ValueError('invalid Qwen scoring mode')
    _positive_integer(return_k, 'return_k')
    if microbatch is not None:
        _positive_integer(microbatch, 'microbatch')
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta):
        raise ValueError('beta must be finite')
    if mode == 'trained' and beta <= 0:
        raise ValueError('trained mode requires an explicit positive beta (historical value: 1.0)')
    if mode != 'trained' and beta != 0:
        raise ValueError('margin and stock modes require beta=0; no residual head is applied')


def _validate_pools(pools):
    if not isinstance(pools, list) or not pools:
        raise ValueError('require nonempty Qwen candidate pools')
    query_ids = set()
    for pool in pools:
        qid = pool.get('query_id') if isinstance(pool, dict) else None
        if not isinstance(qid, str) or not qid or qid in query_ids:
            raise ValueError('duplicate or invalid Qwen pool query ID')
        query_ids.add(qid)
        digest = pool.get('query_text_sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid Qwen pool query text hash')
        candidates = pool.get('results')
        if not isinstance(candidates, list) or not candidates:
            raise ValueError('empty or invalid Qwen candidate pool')
        row_ids = set()
        for candidate in candidates:
            row = candidate.get('row_id') if isinstance(candidate, dict) else None
            if isinstance(row, bool) or not isinstance(row, int) or row < 0 or row in row_ids:
                raise ValueError('duplicate or invalid Qwen candidate row ID')
            row_ids.add(row)
    return query_ids


def validate_score_request(pools, contract=None, mode='margin', beta=0.0,
                           return_k=100, microbatch=None, require_full=False):
    """Cheap public preflight, also used before CLI passage/feature reads."""
    _score_settings(mode, beta, return_k, microbatch)
    pool_ids = _validate_pools(pools)
    validated = validate_feature_contract(contract, require_full=require_full)
    if validated['feature_contract_id'] == ENRICHED_FEATURE_ID and mode != 'margin':
        raise ValueError('enriched passages support measured margin mode; stock/trained require the historical original contract')
    if validated['feature_contract_id'] == ENRICHED_FEATURE_ID and microbatch is not None and microbatch > 4:
        raise ValueError('enriched margin microbatch must not exceed the measured maximum of 4')
    return pool_ids, validated


def _trusted_passage_list(path, expected):
    if sha(path) != expected:
        raise ValueError('reranker passages checksum mismatch for the selected feature contract')
    with Path(path).open('rb') as stream:
        passages = pickle.load(stream)  # Only trusted local corpus files.
    if not isinstance(passages, list) or not all(isinstance(x, str) for x in passages):
        raise ValueError('invalid trusted passage list')
    return passages


def verify_passages(path, contract=None, enriched_passages_path=None):
    """Load the exact measured pickle, or original pickle plus bound JSONL overlay.

    The overlay is the existing changed_passages.jsonl, not a caller-generated
    conversion. Its file hash, row registry and before/after text hashes are
    additional transport validation; the historical scientific IDs stay intact.
    """
    contract = validate_feature_contract(contract, require_full=True)
    identifier = contract['feature_contract_id']
    if identifier == ORIGINAL_FEATURE_ID:
        if enriched_passages_path is not None:
            raise ValueError('historical original-passage mode does not accept an enriched overlay')
        passages = _trusted_passage_list(path, contract['feature_contract']['original_passages_sha256'])
        return VerifiedPassages(passages, identifier)
    binding = contract['passage_inputs']
    changed_rows = set(binding['changed_row_ids'])
    if enriched_passages_path is None:
        passages = _trusted_passage_list(path, binding['enriched_passages_sha256'])
    else:
        if sha(enriched_passages_path) != binding['changed_passages_sha256']:
            raise ValueError('enriched passage overlay checksum mismatch')
        passages = _trusted_passage_list(path, binding['original_passages_sha256'])
        seen = set()
        with Path(enriched_passages_path).open(encoding='utf-8') as stream:
            for line in stream:
                entry = json.loads(line)
                row = entry.get('row')
                if isinstance(row, bool) or not isinstance(row, int) or row not in changed_rows or row in seen or not 0 <= row < len(passages):
                    raise ValueError('duplicate or unregistered enriched passage row')
                before, after = entry.get('original_text'), entry.get('enriched_text')
                if not isinstance(before, str) or not isinstance(after, str) or before == after or entry.get('changed') is not True:
                    raise ValueError('invalid changed passage overlay entry')
                if passages[row] != before or hashlib.sha256(before.encode()).hexdigest() != entry.get('original_sha256'):
                    raise ValueError('enriched overlay original passage mismatch')
                if hashlib.sha256(after.encode()).hexdigest() != entry.get('candidate_sha256'):
                    raise ValueError('enriched overlay candidate passage mismatch')
                passages[row] = after
                seen.add(row)
        if seen != changed_rows:
            raise ValueError('enriched passage overlay row coverage mismatch')
    if len(passages) != binding['row_count']:
        raise ValueError('measured enriched passage count mismatch')
    return VerifiedPassages(passages, identifier, changed_rows)


def _head_request(mode, head_dir):
    if mode != 'trained' and head_dir is not None:
        raise ValueError('a reranker head is only accepted in explicit trained mode')
    if mode == 'trained' and head_dir is None:
        raise ValueError('trained mode requires an explicit reranker head')


def _bound_passages(passages, contract):
    identifier = contract['feature_contract_id']
    if identifier == ENRICHED_FEATURE_ID:
        if not isinstance(passages, VerifiedPassages) or passages.feature_contract_id != identifier or \
           passages.changed_rows != frozenset(contract['passage_inputs']['changed_row_ids']) or \
           len(passages) != contract['passage_inputs']['row_count']:
            raise ValueError('enriched scoring requires bound passages returned by verify_passages')
    elif isinstance(passages, VerifiedPassages) and passages.feature_contract_id != identifier:
        raise ValueError('passage corpus belongs to a different feature contract')


def _text_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def score_saved_features(pools, features, head_dir=None, contract=None, passages=None, mode='margin', beta=0.0, return_k=100):
    """Row-bound JSONL: efficient_margin; hidden only for trained, stock_probability only for stock."""
    _, contract = validate_score_request(pools, contract, mode, beta, return_k)
    _head_request(mode, head_dir)
    _bound_passages(passages, contract)
    if mode == 'trained':
        delta, cfg = load_distribution_head(head_dir, 'joint_pair_linear_residual')
        if cfg['feature_contract_id'] != contract['feature_contract_id']:
            raise ValueError('saved Qwen feature/head contract mismatch')
    groups = {}
    for row in features:
        qid, row_id = row['query_id'], row['row_id']
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
            raise ValueError('invalid saved Qwen feature row ID')
        key = (qid, row_id)
        origin = row.get('feature_contract_id')
        registered = origin == contract['feature_contract_id'] or \
            (contract['feature_contract_id'] == ENRICHED_FEATURE_ID and origin == ORIGINAL_FEATURE_ID)
        if key in groups or not registered:
            raise ValueError('duplicate or mismatched pair features')
        if not _text_digest(row.get('query_text_sha256')) or not _text_digest(row.get('passage_sha256')):
            raise ValueError('pair feature missing text identity')
        groups[key] = row
    result = []
    used = set()
    for pool in pools:
        qid = pool['query_id']
        pairs = []
        for candidate in pool['results']:
            key = (qid, candidate['row_id'])
            if key not in groups or key in used:
                raise ValueError('missing or repeated Qwen pair feature')
            used.add(key)
            feature = groups[key]
            if contract['feature_contract_id'] == ENRICHED_FEATURE_ID and \
               feature['feature_contract_id'] == ORIGINAL_FEATURE_ID and candidate['row_id'] in passages.changed_rows:
                raise ValueError('original feature origin cannot be reused for a changed enriched passage')
            if not 0 <= candidate['row_id'] < len(passages):
                raise ValueError('pair row outside selected passage list')
            actual_passage_hash = hashlib.sha256(passages[candidate['row_id']].encode()).hexdigest()
            if feature['passage_sha256'] != actual_passage_hash:
                raise ValueError('cached feature/passage mismatch')
            if feature['query_text_sha256'] != pool['query_text_sha256']:
                raise ValueError('query text changed since feature extraction')
            if 'passage_sha256' in candidate and candidate['passage_sha256'] != feature['passage_sha256']:
                raise ValueError('candidate passage changed since feature extraction')
            pairs.append((candidate, feature))
        if mode == 'stock':
            values = np.asarray([f['stock_probability'] for _,f in pairs], dtype=np.float64)
        elif mode == 'margin':
            values = np.asarray([f['efficient_margin'] for _,f in pairs], dtype=np.float64)
        else:
            hidden = np.asarray([f['hidden'] for _,f in pairs], dtype=np.float64)
            values = qwen_scores(hidden, [f['efficient_margin'] for _,f in pairs], delta, beta)
        if values.shape != (len(pairs),) or not np.isfinite(values).all():
            raise ValueError('nonfinite Qwen score')
        order = sorted(range(len(pairs)), key=lambda i: (-values[i], pairs[i][0]['row_id']))[:return_k]
        result.append({'query_id':qid,'query_text_sha256':pool['query_text_sha256'],
                       'candidate_count':len(pairs),'mode':f'qwen_{mode}',
                       'feature_contract_id':contract['feature_contract_id'], 'reranker_head_applied':mode == 'trained',
                       'feature_origin_counts':{origin:sum(f['feature_contract_id'] == origin for _,f in pairs)
                                                for origin in sorted({f['feature_contract_id'] for _,f in pairs})},
                       'results':[{'row_id':pairs[i][0]['row_id'], 'name':pairs[i][0].get('name'),
                                   'score':float(values[i])} for i in order]})
    if len(used) != len(groups):
        raise ValueError('extra or stale feature pairs outside supplied pools')
    return result


def plan_enriched_batches(lengths, microbatch=4, padded_token_cap=4096, max_length=8192):
    """Preserve source order, at most four pairs/4096 padded tokens; long singles."""
    _positive_integer(microbatch, 'microbatch')
    if microbatch > 4:
        raise ValueError('enriched margin microbatch must not exceed 4')
    batches = []
    start = 0
    while start < len(lengths):
        longest = 0
        stop = start
        while stop < len(lengths) and stop-start < microbatch:
            length = lengths[stop]
            if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= max_length:
                raise ValueError('invalid token count in enriched batch plan')
            trial = max(longest, length)
            if stop > start and trial*(stop-start+1) > padded_token_cap:
                break
            longest = trial
            stop += 1
        batches.append((start, stop))
        start = stop
    return batches


def _enriched_token_lengths(tokenizer, pairs, contract):
    prompt, fc = contract['prompt'], contract['feature_contract']
    lengths = []
    for start in range(0, len(pairs), 128):
        bodies = [f"<Instruct>: {prompt['instruction']}\n<Query>: {p[2]}\n<Document>: {p[3]}" for p in pairs[start:start+128]]
        encoded = tokenizer(bodies, padding=False, truncation='longest_first',
                            return_attention_mask=False, add_special_tokens=False, max_length=fc['body_max_length'])
        lengths.extend(len(tokens)+prompt['prefix_tokens']+prompt['suffix_tokens'] for tokens in encoded['input_ids'])
    if len(lengths) != len(pairs):
        raise ValueError('tokenizer changed enriched pair coverage')
    return lengths


def score_live(pools, query_records, passages_path, model_path, head_dir=None, contract=None,
               mode='margin', beta=0.0, microbatch=4, return_k=100, device='cuda:0', enriched_passages_path=None):
    """One decoder forward in trained/margin mode; stock runs its historical full LM path."""
    pool_ids, contract = validate_score_request(pools, contract, mode, beta, return_k,
                                                microbatch, require_full=True)
    _head_request(mode, head_dir)
    if not isinstance(query_records, list):
        raise ValueError('invalid live query records')
    query_by_id = {}
    for record in query_records:
        qid = record.get('query_id') if isinstance(record, dict) else None
        if not isinstance(qid, str) or not qid or qid in query_by_id:
            raise ValueError('duplicate or invalid live query ID')
        if not isinstance(record.get('query_text'), str):
            raise ValueError('missing live query text')
        query_by_id[qid] = record
    if set(query_by_id) != pool_ids:
        raise ValueError('missing or extra live query records')
    for pool in pools:
        q = query_by_id[pool['query_id']]
        if hashlib.sha256(q['query_text'].encode()).hexdigest() != pool['query_text_sha256']:
            raise ValueError('live query text hash mismatch')
    if enriched_passages_path is None:
        passages = verify_passages(passages_path, contract)
    else:
        passages = verify_passages(passages_path, contract, enriched_passages_path)
    _bound_passages(passages, contract)
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise ValueError('model must be an existing local directory; no download is attempted')
    validate_runtime_assets(model_path, 'qwen', contract)
    fc = contract['feature_contract']
    if sha(model_path / 'tokenizer.json') != fc['tokenizer_sha256']:
        raise ValueError('Qwen tokenizer checksum mismatch')
    if sha(model_path / 'model.safetensors.index.json') != fc['model_index_sha256']:
        raise ValueError('Qwen model index checksum mismatch')
    for name, expected in fc['model_shards_sha256'].items():
        if sha(model_path / name) != expected:
            raise ValueError(f'Qwen model shard mismatch: {name}')
    pairs = []
    for pool in pools:
        qid = pool['query_id']
        q = query_by_id[qid]
        for candidate in pool['results']:
            row = candidate['row_id']
            if not 0 <= row < len(passages):
                raise ValueError('passage row out of bounds')
            if 'passage_sha256' in candidate and candidate['passage_sha256'] != hashlib.sha256(passages[row].encode()).hexdigest():
                raise ValueError('live candidate passage hash mismatch')
            pairs.append((qid,row,q['query_text'],passages[row]))
    if mode == 'trained':
        delta, cfg = load_distribution_head(head_dir, 'joint_pair_linear_residual')
        if cfg['feature_contract_id'] != contract['feature_contract_id']:
            raise ValueError('head/feature contract mismatch')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not device.startswith('cuda'):
        raise ValueError('measured BF16 reranking requires a local CUDA device')
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side=fc['padding'],local_files_only=True)
    yes, no = fc['yes_id'], fc['no_id']
    if tokenizer.convert_tokens_to_ids('yes') != yes or tokenizer.convert_tokens_to_ids('no') != no:
        raise ValueError('Qwen decision-token IDs changed')
    prompt = contract['prompt']
    if tokenizer.padding_side != fc['padding'] or \
       len(tokenizer.encode(PREFIX, add_special_tokens=False)) != prompt['prefix_tokens'] or \
       len(tokenizer.encode(SUFFIX, add_special_tokens=False)) != prompt['suffix_tokens']:
        raise ValueError('Qwen tokenizer template lengths or padding changed')
    if fc['max_length'] - prompt['prefix_tokens'] - prompt['suffix_tokens'] != fc['body_max_length']:
        raise ValueError('Qwen tokenizer body length changed')
    if contract['feature_contract_id'] == ENRICHED_FEATURE_ID:
        batches = plan_enriched_batches(_enriched_token_lengths(tokenizer, pairs, contract), microbatch)
    else:
        batches = [(start, min(start+microbatch, len(pairs))) for start in range(0, len(pairs), microbatch)]
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.bfloat16,
                                                 attn_implementation='sdpa',local_files_only=True,
                                                 low_cpu_mem_usage=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    features = []
    for start, stop in batches:
        batch = pairs[start:stop]
        inputs = prepare_inputs(tokenizer,[x[2] for x in batch],[x[3] for x in batch],
                                contract['prompt']['instruction'],device,fc['max_length'])
        if mode == 'stock':
            with torch.inference_mode():
                logits = model(**inputs).logits[:,-1,:]
                values = torch.log_softmax(logits[:,[no,yes]],dim=1)[:,1].exp().float().cpu().numpy()
        else:
            output = extract_pair_features(model,inputs,yes,no)
            values = output['base_logits'] if mode == 'margin' else qwen_scores(output['hidden'],output['base_logits'],delta,beta)
        for pair,value in zip(batch,values):
            if not math.isfinite(float(value)):
                raise ValueError('nonfinite live Qwen score')
            features.append((pair[0],pair[1],float(value)))
    del model
    scored = {(qid,row):value for qid,row,value in features}
    result=[]
    for pool in pools:
        qid=pool['query_id']
        candidates=pool['results']
        order=sorted(candidates,key=lambda c:(-scored[(qid,c['row_id'])],c['row_id']))[:return_k]
        result.append({'query_id':qid,'query_text_sha256':pool['query_text_sha256'],
                       'candidate_count':len(candidates),'mode':f'qwen_{mode}',
                       'feature_contract_id':contract['feature_contract_id'], 'reranker_head_applied':mode == 'trained',
                       'results':[{'row_id':c['row_id'],'name':c.get('name'),
                                   'score':scored[(qid,c['row_id'])]} for c in order]})
    return result
