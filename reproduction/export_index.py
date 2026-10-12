"""Export the exact float32 dataset retained in the original cuVS CAGRA index.

Requires the separately installed matching cuVS 25.12 runtime. This optional
GPU phase performs no neural inference; release its process before encoding.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import HERE, read, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    spec = read(HERE / 'configs/corpus.json')
    if args.output.exists() or args.output.with_suffix('.partial.npy').exists():
        raise ValueError('output exists; preserve it and choose a new path')
    if sha(args.index) != spec['source_index_sha256']:
        raise ValueError('original index hash mismatch')
    import cuvs
    from cuvs.neighbors import cagra
    if cuvs.__version__ != spec['cuvs_version']:
        raise ValueError('use the recorded cuVS version: ' + spec['cuvs_version'])
    index = cagra.load(str(args.index))
    view = index.dataset
    if tuple(view.shape) != (spec['rows'], spec['dimension']) or np.dtype(view.dtype) != np.float32:
        raise ValueError('index dataset shape/dtype mismatch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.partial.npy')
    output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=view.shape)
    for start in range(0, len(output), 512):
        block = view.slice_rows(start, min(start + 512, len(output))).copy_to_host()
        if not np.isfinite(block).all() or np.any(np.linalg.norm(block, axis=1) == 0):
            raise ValueError('invalid imported vector')
        output[start:start + len(block)] = block
    output.flush()
    del output
    observed = sha(temporary)
    if observed != spec['export_sha256']:
        raise ValueError('export bytes differ from recorded reference; partial file retained')
    temporary.rename(args.output)
    report = {'status': 'passed', 'vectors_sha256': observed, 'rows': spec['rows'],
              'dimension': spec['dimension'], 'neural_model_calls': 0}
    write(args.output.with_suffix('.receipt.json'), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
