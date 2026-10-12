"""Small shared helpers for the explicit, opt-in reproduction scripts."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_rows(path, values):
    with Path(path).open('x') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def safe_path(root, relative):
    root = Path(root).resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or '..' in candidate.parts:
        raise ValueError('expected a contained relative asset path')
    path = root / candidate
    if not path.resolve().is_relative_to(root):
        raise ValueError('asset path escapes its root')
    return path


def verify_data():
    manifest = read(HERE / 'data_manifest.json')
    if manifest.get('schema') != 'embedding-reproduction-data-v1':
        raise ValueError('unsupported reproduction data manifest')
    for name, info in manifest['files'].items():
        path = safe_path(HERE, name)
        if not path.is_file() or sha(path) != info['sha256'] or path.stat().st_size != info['bytes']:
            raise ValueError('reproduction input checksum mismatch: ' + name)
    return manifest


def check_policy(policy):
    if policy.get('schema') != 'data-exposure-v1':
        raise ValueError('unsupported family policy')
    mapping, splits = policy['declaration_to_family'], policy['family_split']
    protected = policy['protected_families']
    if any(f not in splits for f in mapping.values()) or any(f not in splits for f in protected):
        raise ValueError('unmapped family')
    if any(splits[f] == 'train' for f in protected):
        raise ValueError('protected family assigned to training')


def check_training_record(policy, record):
    if record.get('split') != 'train' or record.get('record_kind') != 'relevance':
        raise ValueError('only training relevance records are accepted')
    if record.get('query_provenance') not in {'generated_from_train_spec', 'independent_train_query'}:
        raise ValueError('unapproved training-query provenance')
    if not isinstance(record.get('query_text'), str) or not record['query_text'].strip():
        raise ValueError('missing training query')
    for key in ['source_names', 'positive_names', 'negative_names', 'masked_alternative_names']:
        names = record.get(key)
        if not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError('invalid declaration-name list: ' + key)
        for name in names:
            family = policy['declaration_to_family'].get(name)
            if family is None or family in policy['protected_families'] or policy['family_split'].get(family) != 'train':
                raise ValueError('nontraining declaration: ' + name)
    if not record['positive_names']:
        raise ValueError('missing positive')
