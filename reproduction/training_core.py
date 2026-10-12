"""Original loss, sampling and diagnostics, with portable package imports.

These function bodies retain the numerical operations used by the measured run.
"""
import numpy as np
from lean_context_search.cache import normalize, normalization_backward, text_hash
from common import check_policy, check_training_record

def prepare_records(policy, records, documents, queries):
    check_policy(policy)
    documents.compatible_queries(queries)
    if not records or any(item["split"] != "train" for item in queries.items):
        raise ValueError("training requires nonempty records and a train-only query cache")
    prepared, seen = [], set()
    for record in records:
        check_training_record(policy, record)
        if record["record_kind"] != "relevance":
            raise ValueError("this trainer accepts relevance records only")
        if (record.get("judgment_status") != "accepted" or
                not isinstance(record.get("judgment_provenance"), str) or
                not record["judgment_provenance"].strip() or not record["source_names"]):
            raise ValueError("missing accepted independent relevance judgment/source")
        qid = record.get("query_id")
        if qid in seen or qid not in queries.row:
            raise ValueError("duplicate or uncached training query")
        seen.add(qid)
        qi = queries.row[qid]
        if queries.items[qi]["text_sha256"] != text_hash(record["query_text"]):
            raise ValueError("query text/cache hash mismatch")
        positives, negatives = record["positive_names"], record["negative_names"]
        masked = set(record["masked_alternative_names"])
        if (not negatives or len(set(positives)) != len(positives) or
                len(set(negatives)) != len(negatives) or set(positives) & set(negatives) or
                masked & (set(positives) | set(negatives))):
            raise ValueError("invalid candidate/positive/negative/masked overlap")
        names = positives + negatives
        if any(name not in documents.row for name in names):
            raise ValueError("training answer missing from C document cache")
        prepared.append((qi, np.array([documents.row[n] for n in names]), len(positives)))
    if seen != set(queries.row):
        raise ValueError("train cache includes queries outside this training arm")
    return prepared

def loss_gradient(head, q, docs, positive_count, alpha, temperature):
    """Any-accepted-answer softmax NLL and analytic gradients for one query."""
    if not 0 < alpha <= 1 or not temperature > 0 or not 0 < positive_count < len(docs):
        raise ValueError("invalid objective configuration")
    raw_q = q @ head.weights.T + head.bias
    raw_d = docs @ head.weights.T + head.bias
    uq, ud = normalize(raw_q, allow_zero=True), normalize(raw_d, allow_zero=True)
    logits = ((1 - alpha) * (docs @ q[0]) + alpha * (ud @ uq[0])) / temperature
    logits = logits - logits.max()
    probabilities = np.exp(logits)
    denominator = probabilities.sum()
    numerator = probabilities[:positive_count].sum()
    if numerator <= 0 or not np.isfinite(denominator):
        raise ValueError("invalid softmax; check temperature/inputs")
    loss = float(np.log(denominator) - np.log(numerator))
    delta = probabilities / denominator
    delta[:positive_count] -= probabilities[:positive_count] / numerator
    gq = (delta @ ud).reshape(1, -1) * (alpha / temperature)
    gd = delta[:, None] * uq * (alpha / temperature)
    rq = normalization_backward(raw_q, uq, gq)
    rd = normalization_backward(raw_d, ud, gd)
    gw = rq.T @ q + rd.T @ docs
    gb = rq[0] + rd.sum(axis=0)
    if not np.isfinite(loss) or not np.isfinite(gw).all() or not np.isfinite(gb).all():
        raise ValueError("nonfinite loss/gradient")
    return loss, gw, gb

def groups_for_epoch(records, policy, rng):
    """Equal two-example weight per family; rotate variants within each style."""
    groups = {}
    for index, record in enumerate(records):
        fam = policy["declaration_to_family"][record["source_names"][0]]
        style = record.get("mixed_style")
        if style not in {"statement", "task"}:
            raise ValueError("mixed_style must be statement or task")
        groups.setdefault(fam, {}).setdefault(style, []).append(index)
    if any(set(styles) != {"statement", "task"} for styles in groups.values()):
        raise ValueError("Every owner family requires both styles")
    pairs = []
    for fam in sorted(groups):
        styles = groups[fam]
        pairs.append([int(rng.choice(styles["statement"])),
                      int(rng.choice(styles["task"]))])
    rng.shuffle(pairs)
    return np.asarray([i for pair in pairs for i in pair], dtype=np.int64), len(groups)

def train_diagnostics(head, prepared, docs, queries, alpha, temperature):
    losses, margins, mean_margins, raw_norms = [], [], [], []
    for qi, rows, positive_count in prepared:
        q = normalize(np.asarray(queries.vectors[qi:qi + 1]))
        d = normalize(np.asarray(docs.vectors[rows]))
        raw_q = q @ head.weights.T + head.bias
        raw_d = d @ head.weights.T + head.bias
        u_q = normalize(raw_q, allow_zero=True)
        u_d = normalize(raw_d, allow_zero=True)
        scores = (1 - alpha) * (d @ q[0]) + alpha * (u_d @ u_q[0])
        logits = scores / temperature
        logits -= logits.max()
        exp = np.exp(logits)
        losses.append(float(np.log(exp.sum()) - np.log(exp[:positive_count].sum())))
        margins.append(float(scores[:positive_count].max() - scores[positive_count:].max()))
        mean_margins.append(float(scores[:positive_count].mean() - scores[positive_count:].mean()))
        raw_norms.extend(np.linalg.norm(raw_d, axis=1).tolist())
        raw_norms.append(float(np.linalg.norm(raw_q)))
    return {"mean_train_nll": float(np.mean(losses)),
            "mean_positive_vs_hardest_negative_margin": float(np.mean(margins)),
            "positive_beats_hardest_negative_fraction": float(np.mean(np.asarray(margins) > 0)),
            "mean_positive_vs_mean_negative_margin": float(np.mean(mean_margins)),
            "sample_raw_projection_norm_min": float(np.min(raw_norms)),
            "sample_raw_projection_norm_max": float(np.max(raw_norms)),
            "sample_raw_projection_norm_count": len(raw_norms)}
