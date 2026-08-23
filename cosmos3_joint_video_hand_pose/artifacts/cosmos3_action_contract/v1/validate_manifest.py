#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(entry: dict[str, Any]) -> Path:
    return (ROOT / entry["path"]).resolve()


def check_hashed_file(label: str, entry: dict[str, Any], errors: list[str]) -> None:
    path = resolve(entry)
    if not path.is_file():
        errors.append(f"{label}: missing file: {path}")
        return
    actual = sha256(path)
    if actual != entry["sha256"]:
        errors.append(f"{label}: SHA256 mismatch: expected {entry['sha256']}, got {actual}")


def check_layout(manifest: dict[str, Any], errors: list[str]) -> None:
    contract = manifest["contract"]
    layout = contract["action_layout"]
    cursor = 0
    for block in layout:
        if block["start"] != cursor:
            errors.append(
                f"action_layout.{block['name']}: expected start {cursor}, got {block['start']}"
            )
        if block["end"] <= block["start"]:
            errors.append(f"action_layout.{block['name']}: invalid range")
        cursor = block["end"]
    if cursor != contract["model_action_dim"]:
        errors.append(
            f"action_layout ends at {cursor}, model_action_dim is {contract['model_action_dim']}"
        )
    padding = layout[-1]
    if padding["name"] != "model_padding" or padding["start"] != contract["raw_action_dim"]:
        errors.append("model_padding must begin at raw_action_dim")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-training-ready", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    check_layout(manifest, errors)

    check_hashed_file("codec.manifest", manifest["codec"]["manifest"], errors)
    check_hashed_file("codec.right", manifest["codec"]["right"], errors)
    check_hashed_file("codec.left", manifest["codec"]["left"], errors)
    check_hashed_file("source_data.episodes_manifest", manifest["source_data"]["episodes_manifest"], errors)
    check_hashed_file("source_data.segments_manifest", manifest["source_data"]["segments_manifest"], errors)
    check_hashed_file("legacy_analysis", manifest["legacy_analysis"], errors)

    for name in ("state", "future_delta"):
        entry = manifest["normalizers"][name]
        if entry["status"] == "ready":
            if not entry.get("path") or not entry.get("sha256"):
                errors.append(f"normalizers.{name}: ready entry requires path and sha256")
            else:
                check_hashed_file(f"normalizers.{name}", entry, errors)

    if args.require_training_ready:
        if not manifest.get("training_ready") or manifest.get("status") != "ready":
            errors.append("manifest is not marked training-ready")
        for name in ("state", "future_delta"):
            if manifest["normalizers"][name]["status"] != "ready":
                errors.append(f"normalizers.{name} is not ready")

    if errors:
        print("Action artifact validation FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Action artifact integrity check passed.")
    print(f"manifest_sha256={sha256(MANIFEST_PATH)}")
    print(f"status={manifest['status']}")
    print(f"training_ready={str(manifest['training_ready']).lower()}")
    if not manifest["training_ready"]:
        print("Action artifacts are not training-ready; do not start formal training with this manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
