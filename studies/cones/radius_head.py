"""Matched flat/hyperbolic radius head and signed declared-projection cone loss."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

RHO_MIN = 0.4
RHO_MAX = 2.0
RHO0 = 1.0
K = 0.1
CONE_MARGIN = 0.02


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized_input(x):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    if torch.any(~torch.isfinite(norm)) or torch.any(norm <= 1e-8):
        raise ArithmeticError("Nonfinite or zero frozen input")
    return x / norm


class RadiusHead(nn.Module):
    def __init__(self, weight, bias, *, dtype=torch.float32, device="cpu"):
        super().__init__()
        w = torch.as_tensor(np.array(weight, copy=True), dtype=dtype, device=device)
        b = torch.as_tensor(np.array(bias, copy=True), dtype=dtype, device=device)
        if w.shape != (128, 4096) or b.shape != (128,):
            raise ValueError("Expected successful 4096-to-128 ordinary head")
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)
        self.radius_weight = nn.Parameter(torch.zeros(4096, dtype=dtype, device=device))
        initial_bias = math.log((RHO0 - RHO_MIN) / (RHO_MAX - RHO0))
        self.radius_bias = nn.Parameter(torch.tensor(initial_bias, dtype=dtype, device=device))

    def forward(self, e):
        e = normalized_input(e)
        raw = torch.nn.functional.linear(e, self.weight, self.bias)
        norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        if torch.any(~torch.isfinite(norm)) or torch.any(norm <= 1e-7):
            raise ArithmeticError("Coincident or zero direction; no silent replacement")
        direction = raw / norm
        logit = torch.nn.functional.linear(e, self.radius_weight[None, :], self.radius_bias[None]).squeeze(-1)
        rho = RHO_MIN + (RHO_MAX - RHO_MIN) * torch.sigmoid(logit)
        if torch.any(~torch.isfinite(rho)) or torch.any(rho < RHO_MIN) or torch.any(rho > RHO_MAX):
            raise ArithmeticError("Invalid learned radius")
        return direction, rho, norm.squeeze(-1)

    def save(self, output, metadata):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        arrays = {name: parameter.detach().cpu().numpy().astype(np.float32)
                  for name, parameter in self.named_parameters()}
        np.savez(output / "parameters.npz", **arrays)
        meta = {"schema": "radius-direction-head-v1", "input_dim": 4096,
                "head_dim": 128, "radius_bounds": [RHO_MIN, RHO_MAX], "initial_radius": RHO0,
                "parameter_count": sum(p.numel() for p in self.parameters()),
                "parameters_sha256": sha(output / "parameters.npz"), **metadata}
        (output / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
        return meta

    @classmethod
    def load(cls, directory, *, dtype=torch.float32, device="cpu"):
        directory = Path(directory)
        meta = json.loads((directory / "manifest.json").read_text())
        if meta.get("schema") != "radius-direction-head-v1" or sha(directory / "parameters.npz") != meta["parameters_sha256"]:
            raise ValueError("Head manifest/hash mismatch")
        with np.load(directory / "parameters.npz", allow_pickle=False) as arrays:
            head = cls(arrays["weight"], arrays["bias"], dtype=dtype, device=device)
            for name, parameter in head.named_parameters():
                value = torch.as_tensor(arrays[name].copy(), dtype=dtype, device=device)
                if value.shape != parameter.shape:
                    raise ValueError(f"Parameter shape changed: {name}")
                parameter.data.copy_(value)
        return head, meta


def sigma(geometry):
    if geometry == "euclidean":
        return RHO0 ** 2
    if geometry == "hyperbolic":
        return math.sinh(RHO0) ** 2
    raise ValueError("Unknown geometry")


def learned_score(uq, ud, rq, rd, geometry):
    """Scores for q [B,k] and candidate documents [B,L,k]."""
    cosine = torch.sum(uq[:, None, :] * ud, dim=-1)
    if geometry == "euclidean":
        energy = 0.5 * (rq[:, None] - rd).square() + rq[:, None] * rd * (1 - cosine)
    elif geometry == "hyperbolic":
        energy = 2 * torch.sinh((rq[:, None] - rd) / 2).square()
        energy += torch.sinh(rq)[:, None] * torch.sinh(rd) * (1 - cosine)
    else:
        raise ValueError("Unknown geometry")
    return 1 - energy / sigma(geometry)


def mixed_relevance(uq, ud, rq, rd, frozen_cosine, geometry, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Invalid alpha")
    return (1 - alpha) * frozen_cosine + alpha * learned_score(uq, ud, rq, rd, geometry)


def _checked_cosine(cosine):
    if torch.any(~torch.isfinite(cosine)) or torch.any(torch.abs(cosine) > 1 + 1e-4):
        raise ArithmeticError("Invalid cone exterior-angle cosine")
    return torch.clamp(cosine, -1, 1)


def cone_gap(us, ur, rs, rr, geometry, *, k=K):
    """Parent/result inside child/source cone exactly when gap <= 0."""
    if torch.any(rs < RHO_MIN) or torch.any(rr < RHO_MIN) or not (0 < k < RHO_MIN / 2):
        raise ArithmeticError("Invalid cone radial domain")
    if geometry == "euclidean":
        source = rs[:, None] * us
        result = rr[:, None] * ur
        difference = result - source
        dist2 = difference.square().sum(-1)
        if torch.any(dist2 <= 1e-12):
            raise ArithmeticError("Distinct structural points numerically coincident")
        cosine = _checked_cosine((source * difference).sum(-1) / (rs * torch.sqrt(dist2)))
        aperture = torch.sqrt(1 - (2 * k / rs).square())
        return aperture - cosine
    if geometry == "hyperbolic":
        source = torch.tanh(rs / 2)[:, None] * us
        result = torch.tanh(rr / 2)[:, None] * ur
        a = source.square().sum(-1)
        b = result.square().sum(-1)
        t = (source * result).sum(-1)
        dist2 = (source - result).square().sum(-1)
        factor = 1 + a * b - 2 * t
        if torch.any(a <= 0) or torch.any(a >= 1) or torch.any(b <= 0) or torch.any(b >= 1):
            raise ArithmeticError("Outside Poincare ball")
        if torch.any(dist2 <= 1e-12) or torch.any(factor <= 1e-12):
            raise ArithmeticError("Distinct structural points numerically coincident")
        sin_aperture = k * (1 - a) / torch.sqrt(a)
        if torch.any(sin_aperture >= 1):
            raise ArithmeticError("Invalid hyperbolic cone aperture")
        cosine = _checked_cosine((t * (1 + a) - a * (1 + b)) / torch.sqrt(a * dist2 * factor))
        aperture = torch.sqrt(1 - sin_aperture.square())
        return aperture - cosine
    raise ValueError("Unknown geometry")


def signed_cone_loss(positive_gaps, negative_gaps, margin=CONE_MARGIN):
    if positive_gaps.numel() == 0 or negative_gaps.numel() == 0:
        raise ValueError("Both structural label classes required")
    return 0.5 * torch.relu(positive_gaps + margin).mean() + 0.5 * torch.relu(margin - negative_gaps).mean()


def document_features(u, rho, geometry):
    if geometry == "euclidean":
        z = rho[:, None] * u
        return torch.cat((z, z.square().sum(-1, keepdim=True)), dim=-1)
    if geometry == "hyperbolic":
        spatial = torch.sinh(rho)[:, None] * u
        return torch.cat((spatial, torch.cosh(rho)[:, None]), dim=-1)
    raise ValueError("Unknown geometry")


def query_features(u, rho, geometry, alpha):
    scale = sigma(geometry)
    if geometry == "euclidean":
        z = rho[:, None] * u
        features = torch.cat((alpha * z / scale,
                              torch.full((len(z), 1), -alpha / (2 * scale), dtype=z.dtype, device=z.device)), dim=-1)
        constant = alpha * (1 - z.square().sum(-1) / (2 * scale))
        return features, constant
    if geometry == "hyperbolic":
        features = torch.cat((alpha * torch.sinh(rho)[:, None] * u / scale,
                              -alpha * torch.cosh(rho)[:, None] / scale), dim=-1)
        constant = torch.full((len(u),), alpha * (1 + 1 / scale), dtype=u.dtype, device=u.device)
        return features, constant
    raise ValueError("Unknown geometry")
