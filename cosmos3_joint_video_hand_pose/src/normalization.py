from __future__ import annotations

import json
from pathlib import Path

import torch


class PiecewiseAsinhNormalizer:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload["method"] != "piecewise_asinh_rot":
            raise ValueError(f"unsupported normalizer method {payload['method']!r}")
        self.beta = float(payload.get("beta", 1.0))
        stats = payload["stats"]
        if "center" in stats or "scale" in stats:
            if "center" not in stats or "scale" not in stats:
                raise ValueError("normalizer stats must provide both center and scale")
            center = torch.tensor(stats["center"], dtype=torch.float32)
            scale = torch.tensor(stats["scale"], dtype=torch.float32)
        else:
            q01 = torch.tensor(stats["q01"], dtype=torch.float32)
            q99 = torch.tensor(stats["q99"], dtype=torch.float32)
            center = (q99 + q01) / 2
            scale = (q99 - q01) / 2
        if center.shape != (27,) or scale.shape != (27,):
            raise ValueError("EgoVerse pose normalizers must contain 27 channels")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or torch.any(scale < 0):
            raise ValueError("normalizer center/scale must be finite and scale must be non-negative")
        self.center = center
        # Legacy state statistics contain mathematically constant identity
        # channels whose tiny float64 q01/q99 gap rounds to zero in float32.
        self.scale = scale.clamp_min(1e-8)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        center = self.center.to(values)
        scale = self.scale.to(values)
        z = (values - center) / scale
        beta = torch.as_tensor(self.beta, device=values.device, dtype=values.dtype)
        tail = 1 + torch.asinh(beta * (z.abs() - 1)) / beta
        return torch.where(z.abs() <= 1, z, z.sign() * tail)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        center = self.center.to(values)
        scale = self.scale.to(values)
        beta = torch.as_tensor(self.beta, device=values.device, dtype=values.dtype)
        tail = 1 + torch.sinh(beta * (values.abs() - 1)) / beta
        z = torch.where(values.abs() <= 1, values, values.sign() * tail)
        return z * scale + center
