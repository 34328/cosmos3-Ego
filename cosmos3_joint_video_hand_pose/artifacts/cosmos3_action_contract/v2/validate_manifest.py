#!/usr/bin/env python3
"""Validate V2 artifact hashes and its explicit translation-scale contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if manifest.get("artifact_id") != "egoverse_cosmos3_action_contract_v2":
        errors.append("unexpected artifact_id")
    if not manifest.get("training_ready") or manifest.get("status") != "ready":
        errors.append("artifact is not training-ready")
    for name, entry in manifest["normalizers"].items():
        path = ROOT / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            errors.append(f"normalizer hash mismatch: {name}")
    future = json.loads((ROOT / manifest["normalizers"]["future_delta"]["path"]).read_text(encoding="utf-8"))
    # The immutable normalizer keeps this legacy JSON key for hash/checkpoint
    # compatibility; it is the contract used by overfit_v0.0.
    contract = future.get("v6_contract", {})
    if contract.get("translation_center") != "zero":
        errors.append("future translation must be zero-centered")
    if contract.get("translation_scale") != "train_only_std_from_actual_85k_sampler":
        errors.append("future translation scale method mismatch")
    if contract.get("rotation_channels") != "v1_q01_q99_center_scale_unchanged":
        errors.append("future rotation contract mismatch")
    report_entry = manifest["normalizers"]["future_delta"]["report"]
    report_path = ROOT / report_entry["path"]
    if not report_path.is_file() or sha256(report_path) != report_entry["sha256"]:
        errors.append("normalizer report hash mismatch")
    if errors:
        print("V2 action artifact validation FAILED:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("V2 action artifact integrity check passed.")
    print(f"manifest_sha256={sha256(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
