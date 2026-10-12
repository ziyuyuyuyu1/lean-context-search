"""Opt-in local SentenceTransformer query encoding under the pinned BF16 contract."""
import hashlib
import json
from importlib.resources import files
from pathlib import Path

import numpy as np

from .cache import normalize
from .runtime_assets import validate_runtime_assets


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


_CORPUS_FIELDS = {'candidate_mathlib_commit', 'corpus_source_commit'}


def validate_encoder_contract(contract, allow_new_corpus=False):
    """Bind the historical cache ID to this release's measured runtime.

    The source ID is historical provenance, not a digest of the portable JSON.
    Only corpus provenance may vary under the explicit new-corpus opt-in.
    """
    canonical = json.loads(files('lean_context_search').joinpath('encoder_contract.json').read_text())
    if not isinstance(contract, dict) or contract.keys() != canonical.keys():
        raise ValueError('encoder contract fields differ from the validated release')
    for field, expected in canonical.items():
        if field in _CORPUS_FIELDS and allow_new_corpus:
            continue
        if contract[field] != expected:
            raise ValueError(f'encoder contract {field} differs from the validated release')
    return canonical['source_contract_sha256']


def encode_local(text, model_path, contract_path, device='cuda:0', allow_new_corpus=False):
    """No import or download occurs unless this function is called."""
    contract = json.loads(Path(contract_path).read_text())
    validate_encoder_contract(contract, allow_new_corpus)
    if not isinstance(text, str) or not text:
        raise ValueError('require a nonempty text query')
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise ValueError('supply a local pinned encoder directory and its contract')
    validate_runtime_assets(model_path, 'embedding', contract)
    for filename, expected in contract['model_file_hashes'].items():
        path = model_path / filename
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f'encoder file differs from pinned contract: {filename}')
    import torch
    from sentence_transformers import SentenceTransformer
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('the measured BF16 fresh-query path requires a local CUDA device')
    # Load directly in BF16: loading FP32 on a 24 GB GPU first can OOM before
    # a later dtype conversion, even though the measured BF16 path fits.
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True,
                                model_kwargs={'torch_dtype': torch.bfloat16})
    prompt = contract['query_format'].format(instruction=contract['query_instruction'], text=text)
    vector = np.asarray(model.encode([prompt], convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    if vector.shape != (1, contract['dimension']):
        raise ValueError('encoder output dimension mismatch')
    return normalize(vector)[0]
