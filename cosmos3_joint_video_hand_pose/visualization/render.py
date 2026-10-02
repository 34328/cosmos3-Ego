"""Render fixed action previews from rollout archives; no inference or metrics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from cosmos3_joint_video_hand_pose.src.ar_overlay import draw_hand, GT_COLOR
from cosmos3_joint_video_hand_pose.src.ar_v02_eval import (
    load_rollout, _normalizers, _codecs, _tensors, _rgb_chunks, _raw_gt_payload,
)
from cosmos3_joint_video_hand_pose.src.ar_v02_overlay import (
    render_joint_overlay, project_chunk_hands, transformed_intrinsics,
)


def prediction_panel(layout, meta, arrays):
    args = SimpleNamespace()
    state, future = _normalizers(meta, args)
    frames, timeline = render_joint_overlay(
        layout, *_tensors(arrays), gt_rgb=arrays["gt_rgb"],
        generated_rgb_chunks=_rgb_chunks(arrays), intrinsics=arrays["intrinsics"],
        gt_pixel_transform=arrays["gt_pixel_transform"],
        generated_pixel_transform=arrays["generated_pixel_transform"],
        state_normalizer=state, future_normalizer=future, hand_codecs=_codecs(args, meta),
        history=meta["history"], source_fps=meta["source_fps"],
        speed_factor=meta["speed_factor"], mode="real_time",
        source_offset=meta["source_offset"], raw_gt=_raw_gt_payload(meta, arrays),
    )
    width = arrays["gt_rgb"].shape[2]
    return np.ascontiguousarray(frames[:, :, width:]), timeline


def annotate(frames, title, timeline, *, prediction):
    # Same label band and native replay mapping as the accepted step1000 preview.
    for i, frame in enumerate(frames):
        frame[:56] = 0
        cv2.putText(frame, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .49, (255, 255, 255), 1, cv2.LINE_AA)
        held = " | held RGB" if prediction and timeline["held_background"][i] else ""
        label = (f"clip {i/30:.3f}s | source {int(timeline['source_indexes'][i])}"
                 f" | c{int(timeline['chunk_ids'][i])}{held}")
        cv2.putText(frame, label, (8, 43), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (255, 255, 255), 1, cv2.LINE_AA)


def clip_bounds(start, end, count):
    first, stop = round(float(start) * 30), round(float(end) * 30)
    if not 0 <= first < stop <= count:
        raise ValueError(f"short interval {start}:{end} exceeds {count}/30 seconds")
    return first, stop


def write_video(path, frames):
    if not path.exists():
        temporary = path.with_name(path.stem + ".partial.mp4")
        imageio.mimwrite(temporary, frames, format="FFMPEG", fps=30,
                        codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                        ffmpeg_log_level="error",
                        output_params=["-threads", "1", "-movflags", "+faststart"])
        temporary.replace(path)
    cap = cv2.VideoCapture(str(path))
    actual = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    expected = (len(frames), 30, frames.shape[2], frames.shape[1])
    if actual != expected:
        raise ValueError(f"invalid video {path}: {actual} != {expected}")


def load_sample(path, sample, experiment, history):
    layout, meta, arrays = load_rollout(path)
    expected = dict(sample_id=sample["sample_id"], source_offset=sample["start"],
                    seed=experiment["seed"], steps=experiment["joint_steps"],
                    sigma_small=experiment["sigma_small"], history=history,
                    checkpoint=experiment["checkpoint"], model_version=experiment["version"])
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError(f"archive recipe differs from manifest: {path}")
    if len(layout.boundaries) != 17 or arrays["gt_rgb"].shape[0] != 545:
        raise ValueError(f"expected complete 17-block source timeline: {path}")
    return layout, meta, arrays


def render_one(sample, experiment, output):
    began = time.monotonic()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output = Path(output)
    identity = sample["id"]
    if Path(identity).name != identity or identity in (".", ".."):
        raise ValueError(f"unsafe sample id: {identity}")
    receipt = output / "videos" / f"{identity}.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if all((output / result[key]).is_file() for key in ("full_video", "short_video", "poster")):
            print(json.dumps(dict(event="reused", id=identity)), flush=True)
            return result
    layout, meta, arrays = load_sample(sample["gt_archive"], sample, experiment, "gt")
    gt_prediction, timeline = prediction_panel(layout, meta, arrays)
    if timeline["output_fps"] != 30 or not np.array_equal(
            timeline["source_indexes"], arrays["raw_gt_source_indexes"]):
        raise ValueError("native renderer did not retain the dense 30fps source timeline")
    ground_truth = arrays["gt_rgb"].copy()
    intrinsic = transformed_intrinsics(arrays["intrinsics"], arrays["gt_pixel_transform"])
    uv, valid = project_chunk_hands(arrays["raw_gt_keypoints"], arrays["raw_gt_camera_poses"], intrinsic)
    for t, frame in enumerate(ground_truth):
        for side in range(2):
            draw_hand(frame, uv[t, side], valid[t, side], GT_COLOR, 1)
    del arrays
    layout, meta, arrays = load_sample(sample["generated_archive"], sample, experiment, "generated")
    generated, generated_timeline = prediction_panel(layout, meta, arrays)
    if not np.array_equal(timeline["source_indexes"], generated_timeline["source_indexes"]):
        raise ValueError("GT/generated replay timelines differ")
    del arrays
    version = experiment["version"]
    step = experiment["step"]
    sigma = experiment["sigma_small"]
    annotate(ground_truth, "GT RGB | true hands (green)", timeline, prediction=False)
    annotate(gt_prediction, f"{version} step{step} | GT history | sigma={sigma}", timeline, prediction=True)
    annotate(generated, f"{version} step{step} | generated history | sigma={sigma}", timeline, prediction=True)
    frames = np.concatenate((ground_truth, gt_prediction, generated), axis=2)
    del ground_truth, gt_prediction, generated
    first, stop = clip_bounds(sample["short_start_sec"], sample["short_end_sec"], len(frames))
    full = f"videos/{identity}_full.mp4"
    short = f"videos/{identity}_short.mp4"
    poster = f"posters/{identity}.jpg"
    write_video(output / full, frames)
    write_video(output / short, frames[first:stop])
    if not (output / poster).exists():
        image = cv2.cvtColor(frames[(first + stop) // 2], cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output / poster), image):
            raise OSError(f"could not write {poster}")
    result = dict(sample, full_video=full, short_video=short, poster=poster,
                  full_duration=len(frames) / 30, short_duration=(stop - first) / 30)
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(event="rendered", id=identity, seconds=time.monotonic() - began)), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        assert clip_bounds(2, 6, 545) == (60, 180)
        try:
            clip_bounds(0, 19, 545)
        except ValueError:
            pass
        else:
            raise AssertionError("clip outside source timeline was accepted")
        print("short-clip boundary self-check passed")
        return
    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required")
    began = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    samples, experiment = manifest["samples"], manifest["experiment"]
    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("samples must be nonempty with unique IDs")
    for sample in samples:
        clip_bounds(sample["short_start_sec"], sample["short_end_sec"], 545)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    recipe = output / "render_input.json"
    if recipe.exists() and json.loads(recipe.read_text()) != manifest:
        raise ValueError(f"refusing to overwrite another render recipe: {output}")
    if not recipe.exists():
        if list((output / "videos").glob("*.mp4")):
            raise ValueError("existing videos have no matching recipe; choose a new output")
        recipe.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for name in ("videos", "posters"):
        (output / name).mkdir(exist_ok=True)
    workers = min(args.workers, len(samples))
    if workers < 1:
        raise ValueError("workers must be positive")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(render_one, sample, experiment, str(output)) for sample in samples]
        rows = [future.result() for future in futures]
    rendered = dict(manifest, samples=rows)
    destination = output / "manifest.json"
    destination.write_text(json.dumps(rendered, ensure_ascii=False, indent=2) + "\n")
    from cosmos3_joint_video_hand_pose.visualization.build_page import main as build_page
    build_page(["--manifest", str(destination), "--output", str(output)])
    receipt = dict(status="ok", samples=len(rows), videos=len(rows) * 2, workers=workers,
                   seconds=time.monotonic() - began, gpu_used=False, manifest=str(args.manifest))
    (output / "render_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(event="complete", **receipt)), flush=True)


if __name__ == "__main__":
    main()
