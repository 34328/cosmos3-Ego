"""Build the shared, offline-capable AR video gallery from a render manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _media_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Media paths must be nonempty strings: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"Media paths must be relative to the output directory: {value!r}")
    return value


def _inline_json(value: dict) -> str:
    # A JSON script must not let sample descriptions terminate its script tag.
    return json.dumps(value, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def _validate_manifest(manifest: dict, output_dir: Path) -> set[str]:
    experiment = manifest["experiment"]
    for key in ("title", "version", "step", "sigma_small", "seed", "joint_steps", "checkpoint"):
        if key not in experiment:
            raise ValueError(f"Missing experiment.{key}")
    samples = manifest["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("The manifest must contain at least one rendered sample")
    ids = set()
    media = set()
    for sample in samples:
        for key in ("id", "split", "title", "sample_id", "start", "short_start_sec", "short_end_sec"):
            if key not in sample:
                raise ValueError(f"Missing sample.{key}")
        if sample["id"] in ids or sample["split"] not in ("train", "heldout"):
            raise ValueError(f"Invalid or duplicated sample: {sample['id']}")
        ids.add(sample["id"])
        if not 0 <= sample["short_start_sec"] < sample["short_end_sec"]:
            raise ValueError(f"Invalid short interval: {sample['id']}")
        for key in ("full_video", "short_video", "poster", "reference_video"):
            if key == "reference_video" and not sample.get(key):
                continue
            relative = _media_path(sample[key])
            media.add(relative)
            if not (output_dir / relative).is_file():
                raise FileNotFoundError(output_dir / relative)
    return media


def build_page(manifest_path: Path, output_dir: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = _validate_manifest(manifest, output_dir)
    versions = manifest.get("versions", [])
    version_ids = set()
    version_media = set()
    for version in versions:
        if not version.get("id") or not version.get("label") or version["id"] in version_ids:
            raise ValueError("Versions require distinct IDs and display labels")
        version_ids.add(version["id"])
        version_media.update(_validate_manifest(version, output_dir))
    if versions and version_media != allowed:
        raise ValueError("Version media must match the gallery's media allowlist")
    template = Path(__file__).with_name("index_template.html").read_text(encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output_dir / "index.html"
    result.write_text(template.replace("__MANIFEST_JSON__", _inline_json(manifest)), encoding="utf-8")
    return result


def compose_versions(manifest_paths: list[Path], output_dir: Path) -> Path:
    """Compose explicit render manifests; retain their actual per-version settings."""
    root = output_dir.resolve()
    versions, samples = [], []
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        version = manifest["experiment"]["version"]
        rows = []
        for original in manifest["samples"]:
            row = dict(original)
            for key in ("full_video", "short_video", "poster", "reference_video"):
                if key == "reference_video" and not row.get(key):
                    continue
                row[key] = str((path.parent / _media_path(row[key])).resolve().relative_to(root))
            rows.append(row)
            samples.append(dict(row, id=version + "_" + row["id"]))
        versions.append(dict(id=version, label=version.replace("ar_v", "V", 1),
                             experiment=manifest["experiment"], samples=rows))
    if not versions:
        raise ValueError("At least one version manifest is required")
    combined = dict(experiment=versions[-1]["experiment"], samples=samples,
                    versions=versions, default_version=versions[-1]["id"])
    # Validate before replacing the active gallery's media allowlist.
    allowed = _validate_manifest(combined, root)
    if len({version["id"] for version in versions}) != len(versions):
        raise ValueError("Pass one render manifest per model version")
    assert allowed == set().union(*(_validate_manifest(version, root) for version in versions))
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "manifest.json"
    destination.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def self_check() -> None:
    payload = {"title": "</script><script>alert(1)</script>"}
    encoded = _inline_json(payload)
    assert "</script>" not in encoded and json.loads(encoded) == payload
    assert _media_path("videos/train_01_short.mp4") == "videos/train_01_short.mp4"
    for unsafe in ("../elsewhere.mp4", "/absolute.mp4", "https://example.org/video.mp4"):
        try:
            _media_path(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(unsafe)
    print("Self-check passed: inline JSON escaping and relative media paths")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path)
    source.add_argument("--version-manifests", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return
    if args.output is None or (args.manifest is None and args.version_manifests is None):
        parser.error("--output and either --manifest or --version-manifests are required")
    if args.version_manifests is not None:
        args.manifest = compose_versions(args.version_manifests, args.output)
    print(build_page(args.manifest, args.output))


if __name__ == "__main__":
    main()
