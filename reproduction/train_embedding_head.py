"""Reproduce the 100-update embedding head on the bundled frozen training vectors.

CPU-only. No encoder, reranker, network access or confirmation data is used.
Writes a new output directory; the distributed head is never overwritten.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

# Match the recorded single-thread numerical run and avoid CPU oversubscription.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
import numpy as np
from lean_context_search.cache import AffineHead, Cache, normalize, sha256, write_json
from common import HERE, ROOT, read, rows as read_records, verify_data
from training_core import prepare_records, loss_gradient, groups_for_epoch, train_diagnostics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, help='new checkpoint/log directory')
    ap.add_argument('--seed', type=int, choices=[20261001, 20261002, 20261003], default=20261001)
    ap.add_argument('--check-only', action='store_true', help='verify all training inputs without initializing or updating a head')
    cli = ap.parse_args()
    if not cli.check_only and cli.output is None:
        ap.error('--output is required for training')
    data_manifest = verify_data()
    config = read(HERE / 'configs/embedding_head.json')
    args = argparse.Namespace(**{**config, 'seed': cli.seed,
        'policy': str(HERE / 'data/family_policy.json'), 'records': str(HERE / 'data/train.jsonl'),
        'documents': str(HERE / 'data/documents/manifest.json'),
        'queries': str(HERE / 'data/queries/manifest.json'),
        'output': str(cli.output) if cli.output else '', 'check_only': cli.check_only})
    if args.updates <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or not 0 < args.alpha <= 1:
        raise ValueError("Invalid optimizer budget/configuration")
    output = Path(args.output)
    started = time.monotonic()
    policy = json.loads(Path(args.policy).read_text())
    records = read_records(args.records)
    docs, queries = Cache(args.documents, "documents"), Cache(args.queries, "queries")
    prepared = prepare_records(policy, records, docs, queries)
    families = {policy["declaration_to_family"][r["source_names"][0]] for r in records}
    if args.check_only:
        print(json.dumps({'status': 'passed', 'queries': len(records), 'families': len(families),
                          'gradient_documents': len(docs.items), 'training_updates': 0,
                          'confirmation_content_read': False}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    if len(records) != len(prepared):
        raise ValueError("Prepared record mismatch")
    head = AffineHead.initialize(docs.vectors.shape[1], args.head_dim, args.seed)
    initial_w, initial_b = head.weights.copy(), head.bias.copy()
    params = [head.weights, head.bias]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    rng = np.random.default_rng(args.seed)
    checkpoints = set(args.checkpoint_updates) | {0, args.updates}
    if any(x < 0 or x > args.updates for x in checkpoints):
        raise ValueError("Checkpoint outside update budget")
    history = []
    order = np.empty(0, dtype=np.int64)
    cursor = 0
    epoch = 0
    source_meta = {"kind": "relevance_only_shared_affine_cosine",
                   "config": vars(args), "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                   "record_count": len(records), "family_count": len(families),
                   "document_cache_sha256": data_manifest["source"]["full_documents_manifest_sha256"],
                   "training_document_cache_sha256": docs.digest, "query_cache_sha256": queries.digest,
                   "encoder_id": docs.meta["encoder_id"],
                   "representation_id": docs.meta["representation_id"],
                   "policy_sha256": sha256(args.policy), "records_sha256": sha256(args.records),
                   "code_sha256": {"train_embedding_head.py": sha256(__file__),
                                   "training_core.py": sha256(HERE / "training_core.py"),
                                   "common.py": sha256(HERE / "common.py")},
                   "selection_policy": "one statement and one task per family each epoch; rotate variants; shuffled family pairs",
                   "updates_per_full_epoch": int(np.ceil(2 * len(families) / args.batch_size))}

    def save(step):
        diag = train_diagnostics(head, prepared, docs, queries, args.alpha, args.temperature)
        changes = {"weight_delta_frobenius": float(np.linalg.norm(head.weights - initial_w)),
                   "bias_delta_l2": float(np.linalg.norm(head.bias - initial_b)),
                   "weight_frobenius": float(np.linalg.norm(head.weights)),
                   "bias_l2": float(np.linalg.norm(head.bias)),
                   "weights_finite": bool(np.isfinite(head.weights).all()),
                   "bias_finite": bool(np.isfinite(head.bias).all())}
        record = {"update": step, "epoch": epoch, "cursor_in_epoch": cursor,
                  "elapsed_seconds": time.monotonic() - started, **diag, **changes}
        head.save(output / f"checkpoint_{step:03d}", {**source_meta, "update": step,
                  "diagnostics": record, "training_history": history})
        write_json(output / f"diagnostics_{step:03d}.json", record)
        print(json.dumps({"checkpoint": record}), flush=True)

    save(0)
    for step in range(1, args.updates + 1):
        if cursor >= len(order):
            order, family_count = groups_for_epoch(records, policy, rng)
            if family_count != len(families):
                raise ValueError("Family count drift")
            cursor = 0
            epoch += 1
        batch = order[cursor:cursor + args.batch_size]
        cursor += len(batch)
        gw, gb = np.zeros_like(head.weights), np.zeros_like(head.bias)
        losses = []
        for index in batch:
            qi, rows, positive_count = prepared[int(index)]
            q = normalize(np.asarray(queries.vectors[qi:qi + 1]))
            d = normalize(np.asarray(docs.vectors[rows]))
            loss, w, b = loss_gradient(head, q, d, positive_count, args.alpha, args.temperature)
            losses.append(loss)
            gw += w / len(batch)
            gb += b / len(batch)
        grad_norm = float(np.sqrt(np.sum(gw.astype(np.float64) ** 2) +
                                  np.sum(gb.astype(np.float64) ** 2)))
        if not np.isfinite(grad_norm):
            raise ValueError("Nonfinite gradient")
        gradients = [gw / max(1, grad_norm), gb / max(1, grad_norm)]
        for i, (parameter, gradient) in enumerate(zip(params, gradients)):
            first[i] = .9 * first[i] + .1 * gradient
            second[i] = .999 * second[i] + .001 * gradient * gradient
            update = (first[i] / (1 - .9 ** step)) / (np.sqrt(second[i] / (1 - .999 ** step)) + 1e-8)
            parameter *= 1 - args.learning_rate * args.weight_decay
            parameter -= args.learning_rate * update
            if not np.isfinite(parameter).all():
                raise ValueError("Nonfinite parameters")
        history.append({"update": step, "epoch": epoch, "batch_size": len(batch),
                        "mean_online_nll": float(np.mean(losses)),
                        "unclipped_gradient_l2": grad_norm})
        if step in checkpoints:
            save(step)
    write_json(output / "training_history.json", history)
    write_json(output / "run_manifest.json", {**source_meta, "completed_updates": args.updates,
               "completed_epochs": epoch, "elapsed_seconds": time.monotonic() - started,
               "checkpoints": sorted(checkpoints), "final_head": str(output / f"checkpoint_{args.updates:03d}")})

    reference, _ = AffineHead.load(ROOT / 'heads/e0')
    difference = max(float(np.max(np.abs(head.weights - reference.weights))),
                     float(np.max(np.abs(head.bias - reference.bias))))
    write_json(output / 'reference_comparison.json', {
        'reference_head_sha256': data_manifest['source']['original_head_sha256'],
        'seed_matches_reference': args.seed == 20261001,
        'maximum_absolute_parameter_difference': difference,
        'arrays_identical': bool(np.array_equal(head.weights, reference.weights) and np.array_equal(head.bias, reference.bias)),
        'note': 'Only seed 20261001 is the distributed head; NumPy/BLAS differences can change floating-point rounding.'})
    print(json.dumps({'output': str(output), 'updates': args.updates,
                      'max_parameter_difference_from_published_head': difference}, indent=2))


if __name__ == '__main__':
    main()
