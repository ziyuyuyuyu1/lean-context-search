"""Pre-construction checks for the measured local model metadata.

The packaged manifest supplements historical cache/feature contracts. It is
not supplied by a caller and does not reinterpret their existing identifiers.
"""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
import json
from pathlib import Path, PurePosixPath


_SCHEMA = 'lean-context-frozen-runtime-assets-v1'
_PROFILES = {'embedding', 'qwen'}
_PROFILE_KEYS = {'scientific_id', 'loader_versions', 'required_sha256',
                 'required_absent', 'previously_checked_files'}
_LOADERS = {'embedding': {'sentence-transformers', 'transformers', 'tokenizers'},
            'qwen': {'transformers', 'tokenizers'}}


def _safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
        raise ValueError(f'unsafe runtime asset path: {name!r}')
    path = PurePosixPath(name)
    if path.is_absolute() or str(path) != name or any(part in {'.', '..'} for part in name.split('/')):
        raise ValueError(f'unsafe runtime asset path: {name!r}')
    return Path(*path.parts)


def _manifest_profile(profile):
    if profile not in _PROFILES:
        raise ValueError(f'unknown runtime asset profile: {profile}')
    manifest = json.loads(files('lean_context_search').joinpath('runtime_assets.json').read_text())
    if not isinstance(manifest, dict) or manifest.get('schema') != _SCHEMA or \
       set(manifest) != {'schema', 'meaning', 'profiles'} or \
       not isinstance(manifest.get('meaning'), str) or \
       not isinstance(manifest.get('profiles'), dict) or set(manifest['profiles']) != _PROFILES:
        raise ValueError('invalid installed runtime asset manifest schema')
    data = manifest['profiles'][profile]
    if not isinstance(data, dict) or set(data) != _PROFILE_KEYS or \
       not isinstance(data['required_sha256'], dict) or \
       not isinstance(data['required_absent'], list) or \
       not isinstance(data['previously_checked_files'], list) or \
       not isinstance(data['loader_versions'], dict) or \
       set(data['loader_versions']) != _LOADERS[profile] or \
       any(not isinstance(v, str) or not v for v in data['loader_versions'].values()) or \
       not isinstance(data['scientific_id'], str) or len(data['scientific_id']) != 64 or \
       any(c not in '0123456789abcdef' for c in data['scientific_id']):
        raise ValueError('invalid installed runtime asset profile')
    required = data['required_sha256']
    absent = data['required_absent']
    previous = data['previously_checked_files']
    for name, digest in required.items():
        _safe_relative(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError(f'invalid installed runtime asset digest: {name}')
    for name in absent + previous:
        _safe_relative(name)
    if len(absent) != len(set(absent)) or len(previous) != len(set(previous)) or \
       set(required) & set(absent) or set(previous) & set(absent):
        raise ValueError('conflicting installed runtime asset paths')
    return data


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_runtime_assets(model_path, profile, scientific_contract):
    """Hash metadata on every live call, before any model/tokenizer constructor."""
    data = _manifest_profile(profile)
    if profile == 'embedding':
        identifier = scientific_contract.get('source_contract_sha256')
        previously_checked = sorted(scientific_contract['model_file_hashes'])
    else:
        identifier = scientific_contract.get('feature_contract_id')
        feature = scientific_contract['feature_contract']
        previously_checked = sorted(['tokenizer.json', 'model.safetensors.index.json',
                                     *feature['model_shards_sha256']])
    runtime_identifier = identifier
    if profile == 'qwen' and identifier == 'db4d3cb620852e93be55f1875217800a9d4ca3d90857208456746373f4def601':
        # Both exact registered contracts use the same measured backbone and
        # loader metadata. Only passage provenance differs. The old manifest
        # binding and old feature ID are preserved, rather than relabeled.
        canonical = json.loads(files('lean_context_search').joinpath('qwen_enriched_feature_contract.json').read_text())
        original = json.loads(files('lean_context_search').joinpath('qwen_feature_contract.json').read_text())
        common_original = {key: value for key, value in original['feature_contract'].items()
                           if key != 'original_passages_sha256'}
        common_enriched = {key: value for key, value in feature.items()
                           if key not in {'passage_corpus_sha256', 'passage_source'}}
        if scientific_contract != canonical or common_original != common_enriched or \
           original['prompt'] != canonical['prompt']:
            raise ValueError('qwen enriched runtime/scientific contract binding mismatch')
        runtime_identifier = original['feature_contract_id']
    if runtime_identifier != data['scientific_id'] or previously_checked != sorted(data['previously_checked_files']):
        raise ValueError(f'{profile} runtime manifest/scientific contract binding mismatch')
    for package, expected in data['loader_versions'].items():
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise ValueError(f'{profile} measured loader not installed: {package}') from exc
        if installed != expected:
            raise ValueError(f'{profile} measured loader version differs: {package} {installed} != {expected}')
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError(f'{profile} model must be an existing local directory')
    resolved_root = root.resolve()
    for name, expected in data['required_sha256'].items():
        path = root / _safe_relative(name)
        if not path.is_file() or not path.resolve().is_relative_to(resolved_root):
            raise ValueError(f'{profile} runtime asset missing or escaping: {name}')
        try:
            observed = _sha(path)
        except OSError as exc:
            raise ValueError(f'{profile} runtime asset unreadable: {name}') from exc
        if observed != expected:
            raise ValueError(f'{profile} runtime asset checksum mismatch: {name}')
    for name in data['required_absent']:
        path = root / _safe_relative(name)
        if path.exists() or path.is_symlink():
            raise ValueError(f'{profile} unexpected runtime override: {name}')
    return {'profile': profile, 'scientific_id': identifier,
            'metadata_files_checked': len(data['required_sha256']),
            'absent_overrides_checked': len(data['required_absent'])}
