"""Prepare fixed IT2V inputs and render generated-vs-GT H.264 replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _instruction(prompt: str) -> str:
    try:
        actions = json.loads(prompt).get("actions", [])
        if actions:
            return str(actions[0].get("description", prompt))
    except (json.JSONDecodeError, AttributeError):
        pass
    return prompt


def _wrap(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines, current = [], ""
    for char in " ".join(text.split()):
        candidate = current + char
        if not current or font.getlength(candidate) <= width:
            current = candidate
        else:
            lines.append(current.rstrip())
            current = char.lstrip()
    if current:
        lines.append(current.rstrip())
    return lines or ["(empty instruction)"]


def prepare(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        prepared = {
            "name": payload["name"],
            "model_mode": "image2video",
            "prompt": payload["prompt"],
            "vision_path": payload["vision_path"],
            "fps": payload["fps"],
            "num_frames": payload["num_frames"],
            "seed": payload["seed"],
        }
        (output / path.name).write_text(json.dumps(prepared, indent=2) + "\n", encoding="utf-8")


def render(generated: Path, reference: Path, prompt: str, output: Path) -> None:
    gen = cv2.VideoCapture(str(generated))
    gt = cv2.VideoCapture(str(reference))
    if not gen.isOpened() or not gt.isOpened():
        raise ValueError("could not open generated or reference video")
    fps = float(gen.get(cv2.CAP_PROP_FPS)) or float(gt.get(cv2.CAP_PROP_FPS)) or 30.0
    font = _font(15)
    lines = _wrap(_instruction(prompt), font, 1240)
    banner_h = max(42, 12 + 20 * len(lines))
    if banner_h % 2:
        banner_h += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output), format="FFMPEG", mode="I", fps=fps, codec="libx264",
        pixelformat="yuv420p", macro_block_size=1, ffmpeg_log_level="error",
        output_params=["-crf", "18", "-preset", "medium", "-movflags", "+faststart"],
    )
    try:
        while True:
            ok_gen, gen_bgr = gen.read()
            ok_gt, gt_bgr = gt.read()
            if not ok_gen or not ok_gt:
                break
            gen_rgb = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2RGB)[:360, :640]
            gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)[:360, :640]
            canvas = Image.new("RGB", (1280, banner_h + 360), "black")
            canvas.paste(Image.fromarray(gen_rgb), (0, banner_h))
            canvas.paste(Image.fromarray(gt_rgb), (640, banner_h))
            draw = ImageDraw.Draw(canvas)
            for index, line in enumerate(lines):
                draw.text((12, 7 + 20 * index), line, fill="white", font=font)
            draw.rectangle((0, banner_h, 1280, banner_h + 27), fill=(5, 12, 18))
            draw.text((10, banner_h + 5), "Generated (IT2V)", fill="white", font=_font(14))
            draw.text((650, banner_h + 5), "Ground truth", fill="white", font=_font(14))
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        gen.release()
        gt.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    draw = sub.add_parser("render")
    draw.add_argument("--generated", type=Path, required=True)
    draw.add_argument("--reference", type=Path, required=True)
    draw.add_argument("--prompt", required=True)
    draw.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.output)
    else:
        render(args.generated, args.reference, args.prompt, args.output)


if __name__ == "__main__":
    main()
