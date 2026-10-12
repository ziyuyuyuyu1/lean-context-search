"""Minimal Qwen3 feature adapter; real Torch/model execution is unrun locally.

The server must validate against its pinned original scorer before caching.
This module loads no model on import and performs no generation.
"""


PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on '
          'the Query and the Instruct provided. Note that the answer can only be "yes" or '
          '"no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def prepare_inputs(tokenizer, queries, passages, instruction, device, max_length=8192):
    if tokenizer.padding_side != "left" or len(queries) != len(passages) or not queries:
        raise ValueError("Require left padding and equally sized nonempty query/passage batches")
    if not instruction or any(not isinstance(s, str) for s in list(queries) + list(passages)):
        raise ValueError("Require instruction and textual query/passage inputs")
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    body_length = max_length - len(prefix) - len(suffix)
    if body_length <= 0:
        raise ValueError("Prompt template exceeds context limit")
    pairs = [f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}"
             for q, d in zip(queries, passages, strict=True)]
    encoded = tokenizer(pairs, padding=False, truncation="longest_first",
                        return_attention_mask=False, add_special_tokens=False, max_length=body_length)
    for i, tokens in enumerate(encoded["input_ids"]):
        encoded["input_ids"][i] = prefix + tokens + suffix
    inputs = tokenizer.pad(encoded, padding=True, return_tensors="pt", max_length=max_length)
    return {key: value.to(device) for key, value in inputs.items()}


def extract_pair_features(model, inputs, yes_id, no_id):
    import torch

    if model.training or yes_id == no_id or min(yes_id, no_id) < 0:
        raise ValueError("Require eval model and distinct valid yes/no token IDs")
    if "attention_mask" not in inputs or not bool(torch.all(inputs["attention_mask"][:, -1] == 1)):
        raise ValueError("Final token must be nonpadding for every pair")
    backbone = getattr(model, model.base_model_prefix)
    with torch.inference_mode():
        # Use the decoder directly, avoiding all-layer hidden-state retention and
        # sequence-by-vocabulary logits. The decoder includes its final norm.
        hidden = backbone(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
        # The existing LM head is applied only at this final decision position.
        logits = model.get_output_embeddings()(hidden)
        if max(yes_id, no_id) >= logits.shape[-1]:
            raise ValueError("Token IDs exceed vocabulary")
        two_logits = logits[:, [no_id, yes_id]]
        original_probabilities = torch.log_softmax(two_logits, dim=1)[:, 1].exp()
        base_margins = two_logits[:, 1].float() - two_logits[:, 0].float()
        if not bool(torch.isfinite(hidden).all() and torch.isfinite(base_margins).all()):
            raise ValueError("Nonfinite model outputs")
        return {"hidden": hidden.float().cpu().numpy(),
                "base_logits": base_margins.cpu().numpy(),
                "original_probabilities": original_probabilities.float().cpu().numpy()}
