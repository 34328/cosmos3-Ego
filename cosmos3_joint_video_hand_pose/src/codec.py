from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class FrozenHandMLPAE15(nn.Module):
    """Frozen canonical MLP-AE-15 codec stored in the project artifacts."""

    def __init__(self, checkpoint: str | Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("architecture") != "60-64-SiLU-32-SiLU-15 / 15-32-SiLU-64-SiLU-60":
            raise ValueError(f"unexpected hand codec architecture in {checkpoint}")
        state = payload["state_dict"]
        self.encoder = nn.Sequential(nn.Linear(60, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 15))
        self.decoder = nn.Sequential(nn.Linear(15, 32), nn.SiLU(), nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 60))
        self.encoder.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")})
        self.decoder.load_state_dict({k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")})
        self.register_buffer("input_mean", state["mean"].float())
        self.register_buffer("input_std", state["std"].float())
        self.register_buffer("latent_mean", payload["latent_mean"].float())
        self.register_buffer("latent_std", payload["latent_std"].float())
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, wrist_local_non_wrist_points: torch.Tensor) -> torch.Tensor:
        flat = wrist_local_non_wrist_points.reshape(*wrist_local_non_wrist_points.shape[:-2], 60).float()
        raw = self.encoder((flat - self.input_mean) / self.input_std)
        return (raw - self.latent_mean) / self.latent_std

    @torch.no_grad()
    def decode(self, standardized_latent: torch.Tensor) -> torch.Tensor:
        raw = standardized_latent.float() * self.latent_std + self.latent_mean
        flat = self.decoder(raw) * self.input_std + self.input_mean
        return flat.reshape(*standardized_latent.shape[:-1], 20, 3)
