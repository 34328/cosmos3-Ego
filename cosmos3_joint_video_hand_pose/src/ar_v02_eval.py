"""V0.2 fixed-length offline sampling, CPU evaluation and synchronized overlay.

Run from the repository root with LD_LIBRARY_PATH='' PYTHONPATH=.:packages/cosmos3.
Use /home/lzh/miniconda3/envs/cosmos3/bin/python -m
cosmos3_joint_video_hand_pose.src.ar_v02_eval <command> --help.

Commands:
  sample   --ckpt /absolute/DCP/model --episodes-manifest heldout.csv
           --segments-manifest segments.csv --split heldout
           --eval-windows eval_windows.json --output /new/output/directory
           --history gt pred_history oracle generated
  evaluate --input /output/0000_gt.npz --output /output/metrics.json
  overlay  --input /output/0000_gt.npz --output /output/overlay.mp4
           --mode real_time

sample requires an explicitly allocated GPU and a trained joint_chunk_cond_v1
checkpoint containing its training contract and both type embeddings. Use
one process (torchrun --nproc_per_node=1). All modes use exactly 30 steps.
--no-cache uses the complete causal prefix ending at the current chunk, with
15-history-chunk attention; recomputing only the retained window is not equivalent.

This is NOT a bounded streaming control API: full-clip payloads, output buffers,
layout and per-step selection/remapping grow with clip length. A bounded KV ring
alone does not establish bounded end-to-end latency. Do not use these timings as
48-chunk streaming acceptance. GPU kernels/full Nano CLI execution require separate
validation; CPU offline evaluate/overlay can run with CUDA_VISIBLE_DEVICES=''.

Archives use schema ar_v02_rollout_v1 and allow_pickle=False: metadata (JSON),
predicted_action [states+future,64], gt_future [N,64], boundary_states [N/8,64],
and optional gt_rgb [N+1,H,W,3] + generated_rgb [sum(1+4C),H,W,3] uint8,
generated_offsets [chunks+1], intrinsics [3,3], gt_pixel_transform and
generated_pixel_transform [3,3]. Metadata binds normalizer files by SHA256,
layout, history, source offset/FPS, seed, sample id, conditions and call reports.
Each RGB block includes its U; never decode the concatenated latent payload.
Replay displays the initial U once and excludes later U frames from the timeline.
Left: green GT and red prediction with GT camera. Right: red prediction only,
with predicted camera. Non-finite decoded poses fail explicitly as data errors;
points behind the camera are masked. No detector-based consistency is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .ar_chunk_state import ChunkCameraStateNormalizer
from .ar_v02_layout import JointChunkLayout, LAYOUT_VERSION
from .ar_v02_evaluation import evaluate_joint_actions
from .ar_v02_inference import HISTORY_MODES, JOINT_STEPS
from .ar_v02_overlay import render_joint_overlay, save_joint_overlay
from .normalization import PiecewiseAsinhNormalizer

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "ar_v02_rollout_v1"
CODECS = ROOT / "cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v2_4/option_b_mlp15"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _new_output(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_file(path, value):
    path = _new_output(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def save_rollout(
    path,
    *,
    layout,
    predicted_action,
    gt_future,
    boundary_states,
    metadata,
    gt_rgb=None,
    generated_rgb_chunks=None,
    intrinsics=None,
    gt_pixel_transform=None,
    generated_pixel_transform=None,
):
    """Producer shared by sample CLI and callers using JointARSampler directly."""
    if not layout.chunk_state_conditioning:
        raise ValueError("complete V0.2 requires every S_k")
    meta = dict(
        metadata,
        schema=SCHEMA,
        layout_version=LAYOUT_VERSION,
        num_frames=layout.num_frames,
        chunk_size=layout.chunk_size,
        vision_tokens=layout.vision_tokens,
        bounded_control_api=False,
        execution_scope="fixed_length_offline",
        steps=JOINT_STEPS,
    )
    arrays = dict(
        predicted_action=_array(predicted_action), gt_future=_array(gt_future), boundary_states=_array(boundary_states)
    )
    if gt_rgb is not None:
        if generated_rgb_chunks is None or len(generated_rgb_chunks) != len(layout.boundaries):
            raise ValueError("RGB export requires one complete [U,V] decoded block per chunk")
        arrays.update(
            gt_rgb=_array(gt_rgb),
            generated_rgb=np.concatenate(generated_rgb_chunks),
            generated_offsets=np.cumsum([0] + [len(x) for x in generated_rgb_chunks]),
            intrinsics=_array(intrinsics),
            gt_pixel_transform=_array(gt_pixel_transform),
            generated_pixel_transform=_array(generated_pixel_transform),
        )
    _validate(meta, arrays)
    path = _new_output(path)
    with path.open("xb") as handle:
        np.savez_compressed(handle, metadata=np.asarray(json.dumps(meta, allow_nan=False)), **arrays)
    return path


def _validate(meta, arrays):
    if meta.get("schema") != SCHEMA or meta.get("layout_version") != LAYOUT_VERSION:
        raise ValueError("requires versioned V0.2 rollout; legacy V0.1 NPZ cannot be silently converted")
    if meta.get("history") not in HISTORY_MODES or meta.get("steps") != 30:
        raise ValueError("invalid history mode or sampling budget")
    for key in ("num_frames", "chunk_size", "vision_tokens", "source_offset", "seed"):
        if type(meta.get(key)) is not int:
            raise ValueError(f"metadata {key} must be an integer")
    if meta["source_offset"] < 0 or meta["chunk_size"] not in (1, 2, 3, 4):
        raise ValueError("invalid source offset or chunk size")
    if not meta.get("sample_id") or not meta.get("episode_id"):
        raise ValueError("sample_id and episode_id are required")
    for key in ("source_fps", "speed_factor"):
        if not isinstance(meta.get(key), (int, float)) or not np.isfinite(meta[key]) or meta[key] <= 0:
            raise ValueError(f"invalid {key}")
    layout = JointChunkLayout(meta["num_frames"], meta["vision_tokens"], meta["chunk_size"])
    n = (layout.num_frames - 1) * 8
    for name, shape in (
        ("predicted_action", (layout.num_action_rows, 64)),
        ("gt_future", (n, 64)),
        ("boundary_states", (layout.num_frames - 1, 64)),
    ):
        value = arrays.get(name)
        if value is None or value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating point {shape}")
        if np.count_nonzero(value[:, 57:]):
            raise ValueError(f"{name} padding must be zero")
    if np.count_nonzero(arrays["boundary_states"][:, :9]):
        raise ValueError("boundary camera slots must be zero")
    for name in ("state_normalizer", "future_normalizer"):
        artifact = meta.get(name, {})
        digest = artifact.get("sha256", "")
        if not artifact.get("path") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"missing normalizer identity: {name}")
    rgb_names = {
        "gt_rgb",
        "generated_rgb",
        "generated_offsets",
        "intrinsics",
        "gt_pixel_transform",
        "generated_pixel_transform",
    }
    if rgb_names & arrays.keys():
        if not rgb_names <= arrays.keys():
            raise ValueError("incomplete RGB/projection payload")
        gt, pred = arrays["gt_rgb"], arrays["generated_rgb"]
        expected = np.cumsum([0] + [1 + b.action_count // 2 for b in layout.boundaries])
        if (
            gt.ndim != 4
            or gt.shape[0] != n + 1
            or gt.shape[-1] != 3
            or gt.dtype != np.uint8
            or pred.shape != (expected[-1], *gt.shape[1:])
            or pred.dtype != np.uint8
            or not np.array_equal(arrays["generated_offsets"], expected)
        ):
            raise ValueError("RGB payload does not cover dense GT and per-chunk sampled [U,V]")
        from .ar_v02_overlay import transformed_intrinsics

        for transform in ("gt_pixel_transform", "generated_pixel_transform"):
            transformed_intrinsics(arrays["intrinsics"], arrays[transform])
    return layout


def load_rollout(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"].item()))
        arrays = {key: archive[key] for key in archive.files if key != "metadata"}
    return _validate(meta, arrays), meta, arrays


def _normalizers(meta, args):
    paths = {}
    for name in ("state_normalizer", "future_normalizer"):
        override = getattr(args, name, None)
        paths[name] = Path(override or meta[name]["path"])
        if _hash(paths[name]) != meta[name]["sha256"]:
            raise ValueError(f"{name} SHA256 differs from the rollout")
    return ChunkCameraStateNormalizer(paths["state_normalizer"]), PiecewiseAsinhNormalizer(paths["future_normalizer"])


def _codecs(args):
    from .codec import FrozenHandMLPAE15

    return tuple(FrozenHandMLPAE15(getattr(args, f"{side}_codec")).eval() for side in ("right", "left"))


def _tensors(arrays):
    return [torch.from_numpy(arrays[name]).float() for name in ("predicted_action", "gt_future", "boundary_states")]


def _rgb_chunks(arrays):
    offsets = arrays["generated_offsets"]
    return [arrays["generated_rgb"][a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def evaluate_archive(path, args):
    layout, meta, arrays = load_rollout(path)
    state, future = _normalizers(meta, args)
    codecs = None if args.rigid_only else _codecs(args)
    metrics = evaluate_joint_actions(
        layout,
        *_tensors(arrays),
        state_normalizer=state,
        future_normalizer=future,
        history=meta["history"],
        hand_codecs=codecs,
    )
    if "gt_rgb" in arrays:
        rect = meta.get("valid_image_rect", [0, 0, arrays["gt_rgb"].shape[2], arrays["gt_rgb"].shape[1]])
        x, y, w, h = rect
        if (
            any(type(v) is not int for v in rect)
            or min(x, y) < 0
            or min(w, h) <= 0
            or x + w > arrays["gt_rgb"].shape[2]
            or y + h > arrays["gt_rgb"].shape[1]
        ):
            raise ValueError("invalid valid_image_rect")
        for b, rgb, record in zip(layout.boundaries, _rgb_chunks(arrays), metrics["chunks"]):
            gt = arrays["gt_rgb"][b.source_start + 2 : b.source_stop + 1 : 2, y : y + h, x : x + w].astype(np.float64)
            pred = rgb[1:, y : y + h, x : x + w].astype(np.float64)
            mse = float(np.square(pred - gt).mean())
            record.update(
                video_mse_uint8=mse,
                video_psnr_db=10 * np.log10(255**2 / mse) if mse else None,
                video_exact_match=mse == 0,
                video_sampled_future_frames=len(gt),
            )
    return dict(
        input=str(Path(path).resolve()),
        metadata=meta,
        metrics=metrics,
        hand_metric_scope="all_finite_tracked_coordinates",
        image_skeleton_detector_metrics_available=False,
    )


def overlay_archive(path, output, args):
    layout, meta, arrays = load_rollout(path)
    if "gt_rgb" not in arrays:
        raise ValueError("overlay requires dense GT RGB and independently decoded generated chunks in the archive")
    state, future = _normalizers(meta, args)
    output = _new_output(output)
    if output.with_suffix(".json").exists():
        raise ValueError("overlay metadata output already exists")
    frames, timeline = render_joint_overlay(
        layout,
        *_tensors(arrays),
        gt_rgb=arrays["gt_rgb"],
        generated_rgb_chunks=_rgb_chunks(arrays),
        intrinsics=arrays["intrinsics"],
        gt_pixel_transform=arrays["gt_pixel_transform"],
        generated_pixel_transform=arrays["generated_pixel_transform"],
        state_normalizer=state,
        future_normalizer=future,
        hand_codecs=_codecs(args),
        history=meta["history"],
        source_fps=meta["source_fps"],
        speed_factor=meta["speed_factor"],
        mode=args.mode,
        source_offset=meta["source_offset"],
    )
    timeline.update(sample_id=meta["sample_id"], episode_id=meta["episode_id"], bounded_control_api=False)
    save_joint_overlay(output, frames, timeline)
    return dict(
        video=str(output), timeline=str(output.with_suffix(".json")), frames=len(frames), fps=timeline["output_fps"]
    )


def _load_bound_model(config, checkpoint, dataset_cfg):
    """Inference loading must not reuse official warm-start skip patterns."""
    import torch.distributed.checkpoint as dcp
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.utils.generator.model_loader import _load_model
    from .ar_v02_contract import ARTrainingContract

    state_path = Path(dataset_cfg.chunk_state_normalizer)
    digest = _hash(dataset_cfg.valid_windows_manifest)
    ChunkCameraStateNormalizer(state_path, expected_manifest_sha256=digest)
    contract = ARTrainingContract(
        state_normalizer=state_path, action_normalizer=dataset_cfg.future_normalizer, manifest_sha256=digest
    )
    reader = dcp.FileSystemReader(str(checkpoint))
    metadata = reader.read_metadata().state_dict_metadata
    binding = {"net.ar_training_contract._extra_state": contract.get_extra_state()}
    dcp.load(binding, storage_reader=reader, no_dist=True)
    contract.set_extra_state(binding["net.ar_training_contract._extra_state"])
    model = instantiate(config.model).cuda()
    model.on_train_start()
    required = [f"net.{name}" for name, _ in model.net.named_parameters()]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"checkpoint missing model weights (including learned types): {missing[:8]}")
    model.net.add_module("ar_training_contract", contract)
    _load_model(model, checkpoint_path=str(checkpoint), credential_path=None, keys_to_skip_loading=[])
    model.eval()
    return model


def sample(args):
    import os

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("offline V0.2 sampler requires one process and one allocated GPU")
    if not torch.cuda.is_available():
        raise ValueError("sample requires an allocated CUDA GPU; evaluate/overlay are CPU commands")
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils import distributed, misc
    from cosmos_framework.utils.lazy_config import instantiate
    from . import config as _registered_config
    from .ar_inference import _training_layout_batch
    from .ar_v02_inference import JointARSampler
    from .ar_v02_overlay import decode_video_chunks
    from .dataset import decode_rgb_video
    import zarr

    output = _new_output(args.output)
    output.mkdir()
    distributed.init()
    try:
        config = load_experiment_from_toml(
            args.toml,
            [
                "job.wandb_mode=disabled",
                "model.config.parallelism.data_parallel_shard_degree=1",
                "model.config.parallelism.data_parallel_replicate_degree=1",
                "model.config.parallelism.context_parallel_shard_degree=1",
                "model.config.parallelism.enable_inference_mode=true",
                "model.config.activation_checkpointing.mode=none",
            ],
        )
        config.validate()
        config.freeze()
        ds_cfg = config.dataloader_train.dataloader.datasets.egoverse.dataset
        dataset = instantiate(
            ds_cfg,
            iterable_shuffle=False,
            random_window=False,
            cfg_dropout_rate=0.0,
            episodes_manifest=str(args.episodes_manifest),
            segments_manifest=str(args.segments_manifest),
            split=args.split,
        )
        raw = dataset.dataset
        if not raw.chunk_camera_mode:
            raise ValueError("V0.2 dataset and frozen chunk-camera state statistics required")
        items = json.loads(Path(args.eval_windows).read_text())
        if not isinstance(items, list) or not items:
            raise ValueError("eval-windows must be a nonempty frozen JSON list")
        by_id = {
            f"{r['episode_hash']}:{r['span_index']}:{r['start_idx']}:{r['end_idx']}": i for i, r in enumerate(raw.rows)
        }
        selected = items if args.limit is None else items[: args.limit]
        if not selected:
            raise ValueError("empty evaluation selection")
        # Validate the entire requested selection before loading expensive model weights.
        for item in selected:
            if set(("sample_id", "start", "frames", "seed")) - item.keys() or item["sample_id"] not in by_id:
                raise ValueError("frozen eval item is absent from the selected dataset/split")
            row = raw.rows[by_id[item["sample_id"]]]
            if item["frames"] != row["_clip_frames"] or item["start"] not in row["_valid_starts"]:
                raise ValueError("eval window disagrees with audited frame count/valid starts")
            if any(type(item[k]) is not int for k in ("start", "frames", "seed")):
                raise ValueError("frozen window indexes and seed must be integers")
        model = _load_bound_model(config, args.ckpt, ds_cfg)
        saved = []
        for number, item in enumerate(selected):
            index = by_id[item["sample_id"]]
            raw_item = raw.get_item_at_window(index, window_start=item["start"])
            transformed = dataset.get_item_at_window(index, window_start=item["start"])
            batch = misc.to(_training_layout_batch(transformed), device="cuda")
            episode_id = raw.rows[index]["episode_hash"]
            episode = raw.episodes[episode_id]
            fps = float(episode["fps"])
            sampler = JointARSampler(
                model,
                batch,
                state_normalizer=raw.chunk_state_normalizer,
                future_normalizer=raw.action_builder.future_normalizer,
                chunk_size=args.chunk_size,
                source_fps=fps,
            )
            group = zarr.open_group(episode["abs_zarr_path"], mode="r")
            dense_indexes = raw_item["action_source_frame_indices"].numpy()
            gt_rgb = decode_rgb_video(group["images.front_1"][dense_indexes]).permute(1, 2, 3, 0).numpy()
            for history in args.history:
                video, action = sampler.sample(
                    history=history, seed=item["seed"], use_cache=not args.no_cache, verify_cache=args.verify_cache
                )
                rgb = decode_video_chunks(model, sampler.layout, video)
                _, gt_future, _ = sampler.layout.unpack_action(sampler.gt_action)
                meta = dict(
                    sample_id=item["sample_id"],
                    episode_id=episode_id,
                    history=history,
                    seed=item["seed"],
                    source_offset=item["start"],
                    source_fps=fps,
                    speed_factor=raw.speed_factor,
                    valid_image_rect=[0, 0, 640, 360],
                    checkpoint=str(Path(args.ckpt).resolve()),
                    eval_windows_sha256=_hash(args.eval_windows),
                    selected_windows=len(selected),
                    frozen_windows=len(items),
                    state_normalizer=dict(
                        path=str(Path(ds_cfg.chunk_state_normalizer).resolve()),
                        sha256=_hash(ds_cfg.chunk_state_normalizer),
                    ),
                    future_normalizer=dict(
                        path=str(Path(ds_cfg.future_normalizer).resolve()), sha256=_hash(ds_cfg.future_normalizer)
                    ),
                    condition_reports=[
                        dict(
                            record,
                            episode_boundary_source_index=record["boundary_source_index"] + item["start"],
                            episode_boundary_time=(record["boundary_source_index"] + item["start"]) / fps,
                        )
                        for record in sampler.condition_reports
                    ],
                    chunk_reports=sampler.chunk_reports,
                    text_prefill_seconds=sampler.cache_prefill_seconds,
                    timing_scope="sampler_only_excludes_initial_GT_VAE_and_offline_RGB_export",
                )
                # Native 640x360 images receive bottom-only reflection padding to 368;
                # image origin and focal lengths do not change, so both transforms are identity.
                archive = save_rollout(
                    output / f"{number:04d}_{history}.npz",
                    layout=sampler.layout,
                    predicted_action=action,
                    gt_future=gt_future,
                    boundary_states=sampler.gt_states,
                    metadata=meta,
                    gt_rgb=gt_rgb,
                    generated_rgb_chunks=rgb,
                    intrinsics=np.asarray(group.attrs["intrinsics"]["front_1"]),
                    gt_pixel_transform=np.eye(3),
                    generated_pixel_transform=np.eye(3),
                )
                saved.append(str(archive))
                print(json.dumps(dict(archive=str(archive), history=history, sample_id=item["sample_id"])), flush=True)
            del sampler, batch
        _json_file(output / "run.json", dict(schema=SCHEMA, bounded_control_api=False, archives=saved))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("sample", help="fixed-length CUDA inference; not a bounded streaming benchmark")
    run.add_argument("--ckpt", type=Path, required=True, help="trained V0.2 DCP model directory, containing .metadata")
    run.add_argument("--toml", type=Path, default=ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_2.toml")
    for name in ("episodes-manifest", "segments-manifest", "eval-windows", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--split", required=True)
    run.add_argument("--chunk-size", type=int, choices=(1, 2, 3, 4), default=4)
    run.add_argument("--history", nargs="+", choices=HISTORY_MODES, default=["gt"])
    run.add_argument("--limit", type=int, help="explicit subset of the frozen list, recorded in metadata")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument(
        "--verify-cache",
        action="store_true",
        help="same-input flow comparison only; not independent rollout acceptance",
    )
    evaluate = commands.add_parser("evaluate", help="CPU action metrics and sampled-video PSNR")
    evaluate.add_argument("--input", type=Path, nargs="+", required=True)
    evaluate.add_argument("--rigid-only", action="store_true", help="omit hand-codec metrics explicitly")
    overlay = commands.add_parser("overlay", help="CPU 30Hz dual-panel overlay and timestamp JSON")
    overlay.add_argument("--input", type=Path, required=True)
    overlay.add_argument("--mode", choices=("real_time", "model_time"), default="real_time")
    for sub in (evaluate, overlay):
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--state-normalizer", type=Path, help="relocated file; hash must match archive")
        sub.add_argument("--future-normalizer", type=Path, help="relocated file; hash must match archive")
        for side in ("right", "left"):
            sub.add_argument(f"--{side}-codec", type=Path, default=CODECS / f"{side}_mlp15_primary.pt")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        torch.set_num_threads(1)
        if args.command == "sample":
            if args.limit is not None and args.limit < 1:
                raise ValueError("--limit must be positive")
            if args.no_cache and args.verify_cache:
                raise ValueError("--verify-cache requires cache")
            sample(args)
        elif args.command == "evaluate":
            report = dict(
                schema=SCHEMA, bounded_control_api=False, samples=[evaluate_archive(p, args) for p in args.input]
            )
            _json_file(args.output, report)
            print(json.dumps(dict(report=str(args.output), samples=len(report["samples"]))))
        else:
            print(json.dumps(overlay_archive(args.input, args.output, args)))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(f"V0.2 evaluation error: {exc}") from exc


if __name__ == "__main__":
    main()
