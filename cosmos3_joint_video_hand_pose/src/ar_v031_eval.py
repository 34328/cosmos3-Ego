"""V0.3.1 offline joint sampling with joint history refresh; shared V0.2 archive/metrics.

V0.3 exposes no registration or metadata-identity injection point. This module keeps only its sample
orchestration adapted for V0.3.1, and reuses its loader, contract, archive, metric
and overlay helpers. It never patches V0.2 globals or bypasses the V0.3.1 model.
Run with LD_LIBRARY_PATH='' PYTHONPATH=.:packages/cosmos3 and the cosmos3 Python.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .ar_v02_eval import (
    ROOT, SCHEMA, _array, _hash, _inference_hand_codecs, _json_file,
    _load_bound_model, _new_output, evaluate_archive, overlay_archive, save_rollout,
)
from .ar_v02_inference import HISTORY_MODES, JOINT_STEPS
from .ar_v031_eval_contract import (
    MODEL_TARGET, MODEL_VERSION, PREFIX_LOSS_DENOMINATOR, PREFIX_LOSS_MASK_SCOPE,
    validate_checkpoint_snapshot, validate_inference_config,
)


def sample(args):
    import os

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("offline V0.3.1 sampler requires one process and one allocated GPU")
    if not torch.cuda.is_available():
        raise ValueError("sample requires an allocated CUDA GPU; evaluate/overlay are CPU commands")
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils import distributed, misc
    from cosmos_framework.utils.lazy_config import instantiate
    from . import ar_v031_config as _registered_config
    from .ar_inference import _training_layout_batch
    from .ar_v03_inference import DiffusionForcingJointARSampler
    from .ar_v02_overlay import decode_video_chunks
    from .dataset import decode_rgb_video
    import zarr

    snapshot_config, provenance = validate_checkpoint_snapshot(args.training_snapshot, args.ckpt)
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
        provenance.update(validate_inference_config(config, snapshot_config))
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
            fixed_windows_manifest=None,
        )
        raw = dataset.dataset
        if not raw.chunk_camera_mode:
            raise ValueError("V0.3.1 dataset and frozen chunk-camera state statistics required")
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
        from .ar_v031_model import EgoVerseARV031Model
        if (not isinstance(model, EgoVerseARV031Model)
                or model.config.mask_prefix_loss is not True):
            raise ValueError("V0.3.1 sampling requires the registered EgoVerseARV031Model")
        video_shift = getattr(args, "video_shift", 5.0)
        video_guidance = getattr(args, "video_guidance", 1.0)
        from .ar_inference import flow_sigmas
        sampler_type, guidance_kwargs = DiffusionForcingJointARSampler, {}
        if video_guidance != 1:
            raise ValueError("V0.3.1 uses the unchanged video_guidance=1 recipe")
        saved = []
        for number, item in enumerate(selected):
            index = by_id[item["sample_id"]]
            raw_item = raw.get_item_at_window(index, window_start=item["start"])
            transformed = dataset.get_item_at_window(index, window_start=item["start"])
            batch = misc.to(_training_layout_batch(transformed), device="cuda")
            episode_id = raw.rows[index]["episode_hash"]
            episode = raw.episodes[episode_id]
            fps = float(episode["fps"])
            sampler = sampler_type(
                model,
                batch,
                state_normalizer=raw.chunk_state_normalizer,
                future_normalizer=(raw.future_normalizer if getattr(raw, "fixed_camera_mode", False)
                                   else raw.action_builder.future_normalizer),
                hand_codecs=_inference_hand_codecs(raw),
                chunk_size=args.chunk_size,
                source_fps=fps,
                **guidance_kwargs,
            )
            group = zarr.open_group(episode["abs_zarr_path"], mode="r")
            dense_indexes = raw_item["action_source_frame_indices"].numpy()
            from .action import pose_matrices
            # Preserve original annotations; never reconstruct GT from AE latents.
            raw_gt = dict(
                keypoints=_array(raw_item["ar_source_keypoints_world"]),
                camera_poses=pose_matrices(_array(raw_item["ar_source_poses"])[:, 0]),
                source_indexes=dense_indexes,
                coordinate_frame="world", units="metres", hand_order=["right", "left"],
            )
            gt_rgb = decode_rgb_video(group["images.front_1"][dense_indexes]).permute(1, 2, 3, 0).numpy()
            for history in args.history:
                video, action = sampler.sample(
                    history=history, seed=item["seed"], use_cache=not args.no_cache, verify_cache=args.verify_cache,
                    video_schedule=flow_sigmas(JOINT_STEPS, video_shift),
                    sigma_small=args.sigma_small,
                )
                rgb = decode_video_chunks(model, sampler.layout, video)
                _, gt_future, _ = sampler.layout.unpack_action(sampler.gt_action)
                meta = dict(
                    model_version=MODEL_VERSION,
                    model_target=MODEL_TARGET,
                    mask_prefix_loss=True,
                    prefix_loss_denominator=PREFIX_LOSS_DENOMINATOR,
                    prefix_loss_mask_scope=PREFIX_LOSS_MASK_SCOPE,
                    training_snapshot_sha256=provenance["training_snapshot_sha256"],
                    checkpoint_metadata_sha256=provenance["checkpoint_metadata_sha256"],
                    recipe_reference_snapshot_sha256=provenance["recipe_reference_snapshot_sha256"],
                    history_video_sigma=args.sigma_small,
                    history_action_sigma=args.sigma_small,
                    action_representation=sampler.action_adapter.representation,
                    sample_id=item["sample_id"],
                    episode_id=episode_id,
                    history=history,
                    seed=item["seed"],
                    video_shift=video_shift, action_shift=5.0,
                    sigma_small=args.sigma_small,
                    video_guidance=video_guidance, action_guidance=1.0,
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
                if getattr(raw, "fixed_hand_codecs", None) is not None:
                    meta["hand_codecs"] = {
                        side: dict(path=str(codec.checkpoint_path), sha256=codec.checkpoint_sha256)
                        for side, codec in zip(("right", "left"), raw.fixed_hand_codecs)
                    }
                # Native 640x360 images receive bottom-only reflection padding to 368;
                # image origin and focal lengths do not change, so both transforms are identity.
                archive = save_rollout(
                    output / f"{number:04d}_{history}.npz",
                    layout=sampler.layout,
                    predicted_action=action,
                    gt_future=gt_future,
                    boundary_states=sampler.gt_states,
                    metadata=meta,
                    raw_gt=raw_gt,
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
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("sample", help="V0.3.1 fixed-length CUDA inference with joint history refresh")
    run.add_argument("--ckpt", type=Path, required=True, help="trained V0.3.1 DCP model directory")
    run.add_argument("--toml", type=Path, default=Path(__file__).resolve().parents[1] / "configs/ar_v0_3_1.toml")
    run.add_argument("--training-snapshot", type=Path, required=True,
                     help="this actual V0.3.1 run official config.yaml; recipe and complete save are checked")
    for name in ("episodes-manifest", "segments-manifest", "eval-windows", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--split", required=True)
    run.add_argument("--chunk-size", type=int, choices=(1, 2, 3, 4), default=4)
    run.add_argument("--history", nargs="+", choices=HISTORY_MODES, default=["gt"])
    run.add_argument("--limit", type=int, help="explicit subset of frozen list; recorded in metadata")
    run.add_argument("--sigma-small", type=float, default=0.02,
                     help="joint video/action sigma when writing completed chunks into history KV")
    run.add_argument("--video-shift", type=float, default=5.0)
    run.add_argument("--no-cache", action="store_true", help="clean reference only; requires --sigma-small 0")
    run.add_argument("--verify-cache", action="store_true", help="same-input clean reference; requires --sigma-small 0")
    evaluate = commands.add_parser("evaluate", help="shared CPU action metrics and sampled-video PSNR")
    evaluate.add_argument("--input", type=Path, nargs="+", required=True)
    evaluate.add_argument("--rigid-only", action="store_true")
    overlay = commands.add_parser("overlay", help="shared CPU 30Hz dual-panel overlay")
    overlay.add_argument("--input", type=Path, required=True)
    overlay.add_argument("--mode", choices=("real_time", "model_time"), default="real_time")
    for sub in (evaluate, overlay):
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--state-normalizer", type=Path)
        sub.add_argument("--future-normalizer", type=Path)
        for side in ("right", "left"):
            sub.add_argument(f"--{side}-codec", type=Path)
    return root


def validate_sample_args(args):
    if not np.isfinite(args.sigma_small) or not 0 <= args.sigma_small <= 1:
        raise ValueError("sigma_small must be finite and in [0,1]")
    if not np.isfinite(args.video_shift) or args.video_shift <= 0:
        raise ValueError("video shift must be positive and finite")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.no_cache and args.verify_cache:
        raise ValueError("--verify-cache requires cache")
    if args.sigma_small and (args.no_cache or args.verify_cache):
        raise ValueError("nonzero sigma_small requires persistent cache without clean-reference verification")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        torch.set_num_threads(1)
        if args.command == "sample":
            validate_sample_args(args)
            sample(args)
        elif args.command == "evaluate":
            report = dict(schema=SCHEMA, model_version=MODEL_VERSION, bounded_control_api=False,
                          samples=[evaluate_archive(p, args) for p in args.input])
            _json_file(args.output, report)
            print(json.dumps(dict(report=str(args.output), samples=len(report["samples"]))))
        else:
            print(json.dumps(overlay_archive(args.input, args.output, args)))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise SystemExit(f"V0.3.1 evaluation error: {exc}") from exc


if __name__ == "__main__":
    main()
