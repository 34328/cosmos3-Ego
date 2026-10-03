"""Pure RGB/text EgoVerse IT2V clips, using the native Cosmos video contract.

Full-segment mode retains every source frame and pads only the final VAE group.
The explicit window mode retains older preview/crop compatibility. No action is read.
"""
from __future__ import annotations

import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path
import random
from collections import Counter

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, get_worker_info


DEFAULT_FRAME_TIERS = (97, 81, 65, 49, 33, 17)
FULL_SEGMENT_TOKEN_FORMULA = 'caption_tokens + 3 + 240 * (1 + ceil((true_frames - 1) / 4))'
RETENTION_PERCENTAGES = (100, 90, 80, 70, 60, 50)


def full_segment_geometry(true_frames):
    """Retain every source frame; append at most three for the VAE's 4N+1 grid."""
    if int(true_frames) < 2:
        raise ValueError("IT2V segment needs a condition and at least one real future frame")
    latent_frames = 1 + (int(true_frames) - 1 + 3) // 4
    padded_frames = 1 + 4 * (latent_frames - 1)
    return padded_frames, latent_frames, padded_frames - int(true_frames)


def format_video_caption(text, true_frames, fps, append_video_metadata=True, duration_seconds=None):
    caption = text.strip().rstrip('.') + '.'
    if append_video_metadata:
        duration = true_frames/fps if duration_seconds is None else duration_seconds
        caption += f" The video is {duration:.1f} seconds long and is of {fps:.0f} FPS. This video is of 368x640 resolution."
    return caption


def uniform_retention_indices(start, end, retained_frames):
    """Deterministic nearest-grid samples, including both original endpoints."""
    total = int(end) - int(start)
    count = int(retained_frames)
    if not 2 <= count <= total:
        raise ValueError('Uniform retention needs 2 <= retained_frames <= segment frames')
    return int(start) + (np.arange(count, dtype=np.int64)*(total-1)+(count-1)//2)//(count-1)


def choose_retention_plan(true_frames, fps, text, count_caption_tokens, budget):
    """Choose the largest approved ratio fitting the strict native packing budget."""
    for percentage in RETENTION_PERCENTAGES:
        count = max(2, int(true_frames)*percentage//100)
        effective_fps = (count-1)/(true_frames-1)*fps
        caption = format_video_caption(text,count,effective_fps,duration_seconds=true_frames/fps)
        text_tokens = int(count_caption_tokens(caption))
        padded,latent,padding = full_segment_geometry(count)
        cost = video_packing_tokens(text_tokens,padded)
        plan = dict(orig_true_frames=int(true_frames),true_frames=count,
            retention_ratio=percentage/100,actual_retention_ratio=count/true_frames,
            effective_fps=effective_fps,text_tokens=text_tokens,padded_frames=padded,
            latent_frames=latent,temporal_padding=padding,packing_tokens=cost)
        if cost < budget:
            return dict(plan,excluded=False)
    return dict(plan,excluded=True,exclusion_reason='exceeds_token_budget_at_50_percent')


def video_packing_tokens(text_tokens, padded_frames):
    """Exact native PackingDataLoader cost: caption + three markers + 240/latent."""
    if (int(padded_frames) - 1) % 4:
        raise ValueError("Token cost requires VAE-aligned frames")
    return int(text_tokens) + 3 + 240 * (1 + (int(padded_frames) - 1) // 4)


def tokenizer_source_path(config_or_processor):
    """Native lazy composition may already instantiate the tokenizer processor."""
    tokenizer = getattr(config_or_processor, 'tokenizer', None)
    path = getattr(tokenizer, 'name_or_path', None)
    if not path and hasattr(config_or_processor, 'get'):
        path = config_or_processor.get('pretrained_model_name')
    if not path or not Path(path).is_dir():
        raise ValueError('Cannot verify statistics: tokenizer must expose its actual local source directory')
    return str(Path(path).resolve())


def validate_full_segment_budget(receipt_path, split, manifest_paths, budget, tokenizer_path=None,
                                 long_segment_policy='error', return_records=False):
    """Fail before iteration if the verified complete manifest exceeds the budget."""
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt.get('token_formula') != FULL_SEGMENT_TOKEN_FORMULA or receipt.get('frame_stride') != 1:
        raise ValueError('Full-segment statistics use a different token contract')
    if receipt.get('long_segment_policy','error') != long_segment_policy:
        raise ValueError('Full-segment statistics long-segment policy differs')
    if long_segment_policy == 'uniform_retention' and (
            receipt.get('policy_token_budget') != int(budget) or
            receipt.get('retention_percentages') != list(RETENTION_PERCENTAGES)):
        raise ValueError('Uniform-retention statistics budget/ratio ladder differs')
    expected = {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in manifest_paths}
    if receipt['manifest_summaries'][split]['manifest_sha256'] != expected:
        raise ValueError('Full-segment statistics manifest hashes differ')
    if tokenizer_path is not None:
        actual = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in Path(tokenizer_path).iterdir() if p.is_file()}
        if actual != receipt['tokenizer_files_sha256']:
            raise ValueError('Full-segment statistics tokenizer hashes differ')
    records = Path(receipt['records'])
    if hashlib.sha256(records.read_bytes()).hexdigest() != receipt['records_sha256']:
        raise ValueError('Full-segment statistics records hash differs')
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    rows = [r for r in rows if r['split'] == split]
    if len(rows) != receipt['manifest_summaries'][split]['segments_total']:
        raise ValueError('Full-segment statistics do not cover every segment')
    included = [r for r in rows if not r.get('excluded',False)]
    oversized = [r for r in included if r['packing_tokens'] >= int(budget)]
    if oversized:
        hours = sum(r['duration_seconds'] for r in oversized) / 3600
        maximum = max(r['packing_tokens'] for r in rows)
        raise ValueError(f'Full {split} manifest exceeds strict token budget {budget}: '
                         f'{len(oversized)} segments / {hours:.6f} hours; max={maximum}. '
                         'No segments may be dropped; choose an explicit supported budget.')
    result = {'segments':len(included), 'segments_total':len(rows), 'excluded':len(rows)-len(included),
              'max_tokens':max(r['packing_tokens'] for r in included), 'budget':int(budget)}
    return (result,rows,receipt['records_sha256']) if return_records else result


def select_clip_frames(source_frames, frame_stride=1, tiers=DEFAULT_FRAME_TIERS):
    if frame_stride < 1 or any(t < 17 or (t - 1) % 16 for t in tiers):
        raise ValueError("IT2V tiers require 1+16N RGB frames, N>=1, and positive stride")
    return next((t for t in sorted(set(tiers), reverse=True)
                 if 1 + frame_stride * (t - 1) <= source_frames), None)


def _seed(seed, epoch, index):
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{index}".encode()).digest()[:8], "little")


def decode_rgb_video(encoded_frames):
    frames = []
    for value in encoded_frames:
        while isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("Expected JPEG bytes in images.front_1")
        with Image.open(BytesIO(bytes(value))) as image:
            frames.append(torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()))
    video = torch.stack(frames).permute(3, 0, 1, 2).contiguous()
    if tuple(video.shape[-2:]) != (360, 640):
        raise ValueError(f"Expected native 640x360 RGB, got {tuple(video.shape[-2:])}")
    return F.pad(video, (0, 0, 0, 8), mode="reflect")


class CosmosCaptionTokenizer:
    """Same caption tokenizer as the official SFTDataset; no action formatter."""

    def __init__(self, tokenizer_config):
        from cosmos_framework.utils.lazy_config import instantiate
        from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
        self.tokenizer, _ = add_special_tokens(instantiate(tokenizer_config).tokenizer)

    def __call__(self, caption):
        from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
        ids = tokenize_caption(caption, self.tokenizer, is_video=True, use_system_prompt=False)
        if len(ids) > 1024:
            raise ValueError("Caption exceeds 1024 tokens; do not silently truncate labels")
        return torch.tensor(ids, dtype=torch.long)


class EgoVerseIT2VDataset(Dataset):
    """Explicit complete segments, or legacy reproducible window crops.

    ``end_idx`` is exclusive. Manifest split is authoritative: missing episodes,
    invalid bounds or blank text fail loudly. Only window mode excludes clips
    below its minimum tier. Full segments are never silently filtered or cropped.
    """

    def __init__(self, episodes_manifest, segments_manifest, *, split="train",
                 frame_stride=1, clip_frame_tiers=DEFAULT_FRAME_TIERS, seed=42,
                 random_window=True, tokenizer_config=None, caption_tokenizer=None,
                 cfg_dropout_rate=0.1, append_video_metadata=True,
                 sample_mode="window", max_sequence_length=None, segment_statistics_path=None,
                 long_segment_policy='error'):
        if split not in ("train", "test"):
            raise ValueError("Use the original train or test split")
        if sample_mode not in ("window", "full_segment"):
            raise ValueError("sample_mode must be window or full_segment")
        if sample_mode == "full_segment" and frame_stride != 1:
            raise ValueError("Full segments require every original frame: frame_stride=1")
        if long_segment_policy not in ('error','uniform_retention'):
            raise ValueError('Unknown long_segment_policy')
        if long_segment_policy == 'uniform_retention' and (
                sample_mode != 'full_segment' or segment_statistics_path is None or max_sequence_length is None):
            raise ValueError('Uniform retention requires full_segment, token budget and verified statistics receipt')
        self.long_segment_policy = long_segment_policy
        select_clip_frames(0, frame_stride, clip_frame_tiers)
        self.sample_mode, self.max_sequence_length = sample_mode, max_sequence_length
        if not 0 <= cfg_dropout_rate <= 1:
            raise ValueError("cfg_dropout_rate must be in [0,1]")
        self.seed, self.frame_stride = int(seed), int(frame_stride)
        self.random_window = bool(random_window)
        self.cfg_dropout_rate = float(cfg_dropout_rate)
        self.append_video_metadata = bool(append_video_metadata)
        self._tokenizer_config, self._caption_tokenizer = tokenizer_config, caption_tokenizer
        self.epoch = 0
        paths = [Path(episodes_manifest).resolve(), Path(segments_manifest).resolve()]
        with paths[0].open(newline="", encoding="utf-8") as f:
            all_episodes = list(csv.DictReader(f))
        if len({r["episode_hash"] for r in all_episodes}) != len(all_episodes):
            raise ValueError("Duplicate episode hashes can leak train/test splits")
        self.episodes = {r["episode_hash"]: r for r in all_episodes if r["split"] == split}
        with paths[1].open(newline="", encoding="utf-8") as f:
            segments = [r for r in csv.DictReader(f) if r["split"] == split]
        self.rows, self.excluded = [], []
        seen = set()
        for row in segments:
            if row["episode_hash"] not in self.episodes:
                raise ValueError("Segment is missing its episode in the same split")
            ep = self.episodes[row["episode_hash"]]
            start, end = int(row["start_idx"]), int(row["end_idx"])
            sid = f"{row['episode_hash']}:{row['span_index']}:{start}:{end}"
            if sid in seen:
                raise ValueError(f"Duplicate segment {sid}")
            seen.add(sid)
            if not (0 <= start < end <= int(ep["total_frames"])) or float(ep["fps"]) <= 0:
                raise ValueError(f"Invalid segment bounds/fps: {sid}")
            caption = row["text_normalized"].strip()
            if not caption:
                raise ValueError(f"Empty segment caption: {sid}")
            if self.sample_mode == "full_segment":
                full_segment_geometry(end - start)
                frames = end - start
            else:
                frames = select_clip_frames(end - start, self.frame_stride, clip_frame_tiers)
            item = dict(row, sample_id=sid, _clip_frames=frames)
            if frames is None:
                self.excluded.append(dict(item, reason="too_short_for_minimum_tier"))
            else:
                self.rows.append(item)
        if not self.rows:
            raise ValueError(f"No eligible {split} segments")
        def hours(rows):
            return sum((int(r["end_idx"])-int(r["start_idx"]))/float(self.episodes[r["episode_hash"]]["fps"])
                       for r in rows) / 3600
        self.manifest_summary = {
            "split": split, "episodes": len(self.episodes), "segments_total": len(segments),
            "segments_eligible": len(self.rows), "segments_excluded_short": len(self.excluded),
            "episode_hours": sum(int(e["total_frames"])/float(e["fps"]) for e in self.episodes.values())/3600,
            "caption_hours_total": hours(segments), "caption_hours_eligible": hours(self.rows),
            "caption_hours_excluded_short": hours(self.excluded),
            "clip_tier_counts": dict(sorted(Counter(r["_clip_frames"] for r in self.rows).items())),
            "sampled_clip_hours_per_epoch": sum(r["_clip_frames"]*self.frame_stride/float(self.episodes[r["episode_hash"]]["fps"]) for r in self.rows)/3600,
            "frame_stride": self.frame_stride, "speed_factor": 1.0,
            "sample_mode": self.sample_mode,
            "manifest_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        }
        contract = dict(sample_mode=sample_mode, frame_stride=self.frame_stride,
                        long_segment_policy=long_segment_policy,
                        manifest_sha256=self.manifest_summary['manifest_sha256'], seed=self.seed,
                        cfg_dropout_rate=self.cfg_dropout_rate,
                        append_video_metadata=self.append_video_metadata)
        if sample_mode == 'window':
            contract.update(clip_frame_tiers=list(clip_frame_tiers), random_window=self.random_window)
        if segment_statistics_path is not None:
            if sample_mode != 'full_segment' or max_sequence_length is None:
                raise ValueError('Statistics receipt requires full_segment and an explicit token budget')
            tokenizer_path = tokenizer_source_path(tokenizer_config)
            preflight, records, records_sha = validate_full_segment_budget(
                segment_statistics_path, split, paths, max_sequence_length, tokenizer_path,
                long_segment_policy=long_segment_policy, return_records=True)
            self.manifest_summary['budget_preflight'] = preflight
            if long_segment_policy == 'uniform_retention':
                contract.update(retention_percentages=RETENTION_PERCENTAGES,
                                policy_token_budget=int(max_sequence_length),records_sha256=records_sha)
                plans = {r['sample_id']:r for r in records}
                if set(plans) != {r['sample_id'] for r in self.rows}:
                    raise ValueError('Retention records must map each original segment exactly once')
                retained = []
                for original_index,row in enumerate(self.rows):
                    plan = plans[row['sample_id']]
                    if plan['orig_true_frames'] != int(row['end_idx'])-int(row['start_idx']):
                        raise ValueError('Retention plan has wrong original segment length')
                    row.update(_manifest_row_index=original_index, _retention_plan=plan)
                    if plan['excluded']:
                        self.excluded.append(dict(row,reason=plan['exclusion_reason']))
                    else:
                        row['_clip_frames'] = plan['true_frames']
                        retained.append(row)
                self.rows = retained
                if not self.rows:
                    raise ValueError('Uniform-retention policy excludes every segment')
                self.manifest_summary.update(long_segment_policy=long_segment_policy,
                    retention_ratio_counts=dict(Counter(r['_retention_plan']['retention_ratio'] for r in self.rows)),
                    segments_eligible=len(self.rows),segments_excluded_budget=len(self.excluded),
                    caption_hours_eligible=hours(self.rows),caption_hours_excluded_budget=hours(self.excluded),
                    sampled_clip_hours_per_epoch=hours(self.rows),
                    clip_tier_counts=dict(sorted(Counter(r['_clip_frames'] for r in self.rows).items())))
        self.dataset_contract = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def get_shuffle_blocks(self):
        return [(i, 1) for i in range(len(self))]

    def window_indices(self, index, *, epoch=None, window_start=None):
        row = self.rows[index]
        start, end = int(row["start_idx"]), int(row["end_idx"])
        if self.sample_mode == "full_segment":
            if window_start is not None and int(window_start) != start:
                raise ValueError("Full segment cannot be cropped or shifted")
            if self.long_segment_policy == 'uniform_retention':
                return uniform_retention_indices(start,end,row['_clip_frames'])
            return np.arange(start, end, dtype=np.int64)
        span = 1 + self.frame_stride * (row["_clip_frames"] - 1)
        rng = random.Random(_seed(self.seed, self.epoch if epoch is None else epoch, index))
        if window_start is None:
            window_start = rng.randint(start, end-span) if self.random_window else start
        if not start <= window_start <= end-span:
            raise ValueError("Crop crosses the caption boundary")
        return window_start + np.arange(row["_clip_frames"], dtype=np.int64) * self.frame_stride

    def __getitem__(self, index):
        return self.get_item_at_window(index, epoch=self.epoch)

    def get_item_at_window(self, index, *, epoch=0, window_start=None, source_frame_indices=None):
        import zarr
        from cosmos_framework.data.generator.sequence_packing import SequencePlan
        row = self.rows[index]
        ep = self.episodes[row["episode_hash"]]
        if source_frame_indices is not None:
            saved_indices = np.asarray(torch.as_tensor(source_frame_indices).cpu(), dtype=np.int64).reshape(-1)
            if saved_indices.size == 0:
                raise ValueError("Empty checkpoint source window")
            if window_start is not None and int(window_start) != int(saved_indices[0]):
                raise ValueError("Checkpoint start and source indices disagree")
            window_start = int(saved_indices[0])
        indices = self.window_indices(index, epoch=epoch, window_start=window_start)
        if source_frame_indices is not None and not np.array_equal(indices, saved_indices):
            raise ValueError("Checkpoint source indices differ from the configured stride/tier")
        group = zarr.open_group(ep["abs_zarr_path"], mode="r")
        rgb = group["images.front_1"]
        if rgb.shape[0] != int(ep["total_frames"]) or indices[-1] >= rgb.shape[0]:
            raise ValueError("Zarr frame count differs from manifest")
        fps = float(ep["fps"]) / self.frame_stride
        original_frames = int(row['end_idx'])-int(row['start_idx'])
        original_duration = None
        if self.long_segment_policy == 'uniform_retention':
            fps = row['_retention_plan']['effective_fps']
            original_duration = original_frames/float(ep['fps'])
        caption = format_video_caption(row["text_normalized"], len(indices), fps, self.append_video_metadata,
                                       duration_seconds=original_duration)
        # Separate stream from crop selection, deterministic across worker layouts.
        rng = random.Random(_seed(self.seed + 1, epoch, index))
        if rng.random() < self.cfg_dropout_rate:
            caption = ""
        if self._caption_tokenizer is None:
            self._caption_tokenizer = CosmosCaptionTokenizer(self._tokenizer_config)
        ids = torch.as_tensor(self._caption_tokenizer(caption), dtype=torch.long)
        padded_frames, temporal_padding = len(indices), 0
        if self.sample_mode == "full_segment":
            padded_frames, _, temporal_padding = full_segment_geometry(len(indices))
        cost = video_packing_tokens(ids.numel(), padded_frames)
        if self.max_sequence_length is not None and cost >= self.max_sequence_length:
            raise ValueError(f"Sample {row['sample_id']} needs {cost} tokens, exceeding strict budget {self.max_sequence_length}; no frames were dropped")
        video = decode_rgb_video(rgb[indices])
        if temporal_padding:
            video = torch.cat((video, video[:, -1:].expand(-1, temporal_padding, -1, -1)), dim=1)
        result = {
            "__key__": row["sample_id"], "__url__": ep["abs_zarr_path"],
            "sample_id": row["sample_id"], "dataset_index": int(index),
            "video": video, "ai_caption": caption, "text_token_ids": ids,
            "conditioning_fps": fps, "fps": float(ep["fps"]), "num_multiplier": self.frame_stride,
            "n_orig_video_frames": int(ep["total_frames"]), "num_frames": padded_frames,
            "frame_start": int(indices[0]), "frame_end": int(indices[-1]),
            "window_start": int(indices[0]),
            "source_frame_indices": torch.from_numpy(indices.copy()),
            "padding_mask": torch.zeros((1,368,640), dtype=torch.float32),
            "image_size": torch.tensor([368,640,368,640], dtype=torch.float32),
            "sequence_plan": SequencePlan(has_text=True, has_vision=True,
                                          condition_frame_indexes_vision=[0]),
        }
        if self.sample_mode == "full_segment":
            result.update(sample_mode="full_segment", video_true_num_frames=len(indices),
                          video_temporal_padding=temporal_padding)
        if self.long_segment_policy == 'uniform_retention':
            plan = row['_retention_plan']
            result.update(orig_true_frames=original_frames,retention_ratio=plan['retention_ratio'],
                          actual_retention_ratio=plan['actual_retention_ratio'],effective_fps=fps,
                          original_duration_seconds=original_duration,
                          manifest_row_index=row['_manifest_row_index'],long_segment_policy=self.long_segment_policy)
        return result


class IT2VIterableDataset(IterableDataset):
    """RankPartitionedDataLoader-compatible disjoint stream with exact resume.

    Crop/CFG decisions are keyed by epoch/index, so restoring needs only the
    iterator position rather than capturing global Python/NumPy worker RNGs.
    """
    def __init__(self, dataset, seed=42):
        self._dataset, self.seed = dataset, int(seed)
        self.shard_world_size, self.shard_rank = 1, 0
        self._state = {}

    def __len__(self):
        return len(self._dataset)

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        contract = getattr(self._dataset, 'dataset_contract', None)
        if state and contract is not None and state.get('dataset_contract') != contract:
            raise ValueError('Cannot resume a different dataset contract (mode/manifests/sampling)')
        self._state = dict(state)

    def __iter__(self):
        worker = get_worker_info()
        nw, wid = (worker.num_workers, worker.id) if worker else (1, 0)
        shard, nshards = self.shard_rank*nw + wid, self.shard_world_size*nw
        if nshards > len(self):
            raise ValueError("More rank/worker shards than eligible segments")
        state = self._state
        if state and (state["shard"], state["nshards"]) != (shard, nshards):
            raise ValueError("Cannot resume with a different rank/worker topology")
        epoch, offset = state.get("epoch", 0), state.get("offset", 0)
        while True:
            order = torch.randperm(len(self), generator=torch.Generator().manual_seed(self.seed+epoch)).tolist()[shard::nshards]
            for pos in range(offset, len(order)):
                sample = self._dataset.get_item_at_window(order[pos], epoch=epoch)
                self._state = {"epoch": epoch + (pos+1 == len(order)),
                               "offset": (pos+1) % len(order), "shard": shard, "nshards": nshards}
                contract = getattr(self._dataset, 'dataset_contract', None)
                if contract is not None:
                    self._state['dataset_contract'] = contract
                yield sample
            epoch, offset = epoch+1, 0


def get_egoverse_it2v_dataset(*, episodes_manifest, segments_manifest, tokenizer_config,
                            split="train", seed=42, iterable_shuffle=True, **kwargs):
    dataset = EgoVerseIT2VDataset(episodes_manifest, segments_manifest, split=split,
                                 seed=seed, tokenizer_config=tokenizer_config, **kwargs)
    return IT2VIterableDataset(dataset, seed=seed) if iterable_shuffle else dataset
