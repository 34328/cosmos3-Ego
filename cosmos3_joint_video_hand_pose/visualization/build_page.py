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


def build_page(manifest_path: Path, output_dir: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiment = manifest["experiment"]
    for key in ("title", "version", "step", "sigma_small", "seed", "joint_steps", "checkpoint"):
        if key not in experiment:
            raise ValueError(f"Missing experiment.{key}")
    samples = manifest["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("The manifest must contain at least one rendered sample")
    ids = set()
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
            if not (output_dir / relative).is_file():
                raise FileNotFoundError(output_dir / relative)
    template = Path(__file__).with_name("index_template.html").read_text(encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = output_dir / "index.html"
    result.write_text(template.replace("__MANIFEST_JSON__", _inline_json(manifest)), encoding="utf-8")
    return result


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
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required")
    print(build_page(args.manifest, args.output))


if __name__ == "__main__":
    main()
