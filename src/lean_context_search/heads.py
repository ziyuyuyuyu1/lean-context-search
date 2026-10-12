"""Safe loaders and CPU scoring for the two distinct frozen heads."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .cache import EPS, AffineHead


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_distribution_head(directory, kind):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text())
    if config.get('schema') != 'lean-context-head-distribution-v1' or config.get('kind') != kind:
        raise ValueError('wrong head distribution kind')
    if digest(directory / 'head.npz') != config['weights_sha256']:
        raise ValueError('head weight checksum mismatch')
    with np.load(directory / 'head.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, shape in config['npz_arrays'].items():
        value = arrays.get(key)
        if value is None or list(value.shape) != shape or str(value.dtype) != config['dtype']:
            raise ValueError(f'head array {key} shape/dtype mismatch')
        if not np.isfinite(value).all():
            raise ValueError('nonfinite head array')
    if kind == 'shared_affine_cosine_head':
        if set(arrays) != {'weights', 'bias'} or config['epsilon'] != EPS:
            raise ValueError('invalid embedding head archive')
        return AffineHead(arrays['weights'], arrays['bias']), config
    if set(arrays) != {'delta', 'metadata_json'}:
        raise ValueError('unexpected Qwen head archive fields')
    metadata = json.loads(str(arrays['metadata_json'].item()))
    if metadata.get('feature_contract_id') != config['feature_contract_id']:
        raise ValueError('Qwen feature contract mismatch')
    return arrays['delta'], config


def rms_features(hidden):
    x = np.asarray(hidden, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 4096 or not len(x) or not np.isfinite(x).all():
        raise ValueError('expected finite [pairs,4096] final hidden states')
    rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True))
    if np.any(rms < EPS) or not np.isfinite(rms).all():
        raise ValueError('zero or invalid pair hidden RMS')
    return x / rms


def qwen_scores(hidden, efficient_margin, delta, beta=1.0):
    x = rms_features(hidden)
    margin = np.asarray(efficient_margin, dtype=np.float64)
    w = np.asarray(delta, dtype=np.float64)
    if margin.shape != (len(x),) or w.shape != (4096,) or not np.isfinite(margin).all():
        raise ValueError('Qwen score shape/nonfinite mismatch')
    if not np.isfinite(beta):
        raise ValueError('nonfinite beta')
    return margin + beta * (x @ w)
