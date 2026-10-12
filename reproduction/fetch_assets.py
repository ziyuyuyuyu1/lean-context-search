"""Fetch explicitly selected, hash-pinned public assets; no authentication or inference."""
import argparse
from pathlib import Path
import shutil
import urllib.request

from common import HERE, read, safe_path, sha, write


def fetch(asset, output, verify_only=False):
    output = Path(output)
    if not verify_only:
        output.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for entry in asset['files']:
        target = safe_path(output, entry['path'])
        if target.exists():
            if target.stat().st_size != entry['bytes'] or sha(target) != entry['sha256']:
                raise ValueError('existing asset differs; preserve it and use a fresh directory: ' + str(target))
            continue
        if verify_only:
            raise ValueError('missing asset: ' + str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.part')
        if temporary.exists():
            raise ValueError('unfinished download retained; move/remove it before retrying: ' + str(temporary))
        request = urllib.request.Request(entry['url'], headers={'User-Agent': 'LeanContextSearch-Reproduction/1'})
        print('Downloading ' + entry['path'], flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('xb') as stream:
            shutil.copyfileobj(response, stream, length=8 << 20)
        if temporary.stat().st_size != entry['bytes'] or sha(temporary) != entry['sha256']:
            raise ValueError('download checksum mismatch; failed .part retained: ' + entry['path'])
        temporary.rename(target)
        downloaded += 1
    return {'status': 'passed', 'repository': asset['repository'], 'revision': asset['revision'],
            'files_verified': len(asset['files']), 'downloaded_files': downloaded,
            'total_bytes': asset['total_bytes'], 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', choices=['corpus', 'embedding', 'reranker'], required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plan', action='store_true', help='show pinned files and size without downloading')
    parser.add_argument('--verify-only', action='store_true', help='check existing files; no network or writes')
    args = parser.parse_args()
    asset = read(HERE / 'configs/assets.json')['assets'][args.asset]
    import json
    if args.plan:
        print(json.dumps(asset, indent=2))
        return
    if args.output is None:
        parser.error('--output is required')
    report = fetch(asset, args.output, args.verify_only)
    if not args.verify_only:
        write(args.output / 'reproduction_download.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
