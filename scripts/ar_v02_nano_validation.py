"""Numerical acceptance helpers; production modules are imported, never patched on disk."""

from __future__ import annotations
import copy
import dataclasses
import gc
import json
import time
from pathlib import Path
import torch
from cosmos3_joint_video_hand_pose.src.ar_v02_inference import JointARSampler, _schedule, _condition_encoding
from cosmos3_joint_video_hand_pose.src.ar_chunk_state import decode_chunk_camera_state, decode_chunk_camera_action
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION, VIDEO, STATE, JointChunkLayout
from cosmos3_joint_video_hand_pose.src.ar_v02_cache import JointKVCache
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence


def reuse_reference_masks():
    """Memoize only immutable masks in the numerical oracle (no activation/KV reuse)."""
    from cosmos3_joint_video_hand_pose.src.ar_v02_attention import JointTeacherForcingAttention

    original = JointTeacherForcingAttention.block_mask
    masks = {}

    def block_mask(self, *, noisy, text_pad_len, text_len):
        key = (
            repr(self.layouts),
            str(self.device),
            noisy,
            text_pad_len,
            text_len,
            tuple(self.text_sample_ids.tolist()) if self.text_sample_ids is not None else None,
        )
        if key not in masks:
            masks[key] = original(self, noisy=noisy, text_pad_len=text_pad_len, text_len=text_len)
        return masks[key]

    JointTeacherForcingAttention.block_mask = block_mask
    return masks


class Comparisons:
    """Write each local result before failing; no aggregate can hide an outlier."""

    def __init__(self, path):
        self.handle = Path(path).open("x")
        self.count = 0

    def close(self, got, ref, context, *, exact=False):
        if got.shape != ref.shape:
            raise AssertionError(f"{context}: shape {got.shape} != {ref.shape}")
        a, b = got.detach().float(), ref.detach().float()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        error = a - b
        rms = float(b.square().mean().sqrt()) if b.numel() else 0.0
        max_abs = float(error.abs().max()) if b.numel() else 0.0
        relative = float(error.norm() / b.norm().clamp_min(1e-6))
        element = bool(torch.all(error.abs() <= (0.0 if exact else 0.01 + 0.03 * b.abs())))
        passed = (
            finite and element and (max_abs == 0 if exact else (max_abs <= 0.01 if rms < 1e-6 else relative <= 0.01))
        )
        record = dict(
            context=context,
            shape=list(a.shape),
            got_dtype=str(got.dtype),
            reference_dtype=str(ref.dtype),
            finite=finite,
            rms=rms,
            max_abs=max_abs,
            relative_l2=relative,
            element_pass=element,
            passed=passed,
        )
        # Nonfinite values are represented explicitly, never written as invalid JSON NaN.
        import math

        for key in ("rms", "max_abs", "relative_l2"):
            if not math.isfinite(record[key]):
                record[key] = None
        self.handle.write(json.dumps(record, allow_nan=False) + "\n")
        self.handle.flush()
        self.count += 1
        if not passed:
            raise AssertionError(f"bf16 acceptance failed: {record}")

    def finish(self):
        self.handle.close()


def matrix(chunk_sizes=(1, 2, 3, 4), histories=("gt", "pred_history", "generated")):
    return [
        dict(c=c, tail=tail, history=h, schedule=s)
        for c in chunk_sizes
        for tail in ([0] if c == 1 else range(1, c))
        for h in histories
        for s in ("shift5", "independent")
    ]


def extend_latents(base, full_chunks, tail):
    """Deterministic repeated real encoded blocks; explicitly a latent-only numerical fixture."""
    out = copy.copy(base)
    c = base.chunk_size
    layout = JointChunkLayout(1 + full_chunks * c + tail, base.layout.vision_tokens, c)
    full = [b for b in base.layout.boundaries if b.latent_stop - b.latent_start == c]
    assert full, "real clip must contain at least one full chunk"
    videos, futures, states = [], [], []
    for i, b in enumerate(layout.boundaries):
        src = full[i % len(full)]
        length = b.latent_stop - b.latent_start
        indexes = base.layout.video_indexes(src.chunk_id)[: length + 1].to(base.gt_video.device)
        videos.append(base.gt_video[:, :, indexes])
        rows = (base.roles == ACTION) & (base.chunks == src.chunk_id)
        futures.append(base.gt_action[rows][: length * 8])
        states.append(base.gt_states[src.latent_start - 1 : src.latent_start - 1 + length])
    out.layout = layout
    out.gt_video = torch.cat(videos, 2)
    out.gt_states = torch.cat(states)
    future = torch.cat(futures)
    out.gt_action, _ = layout.assemble_action(
        future, out.gt_states, torch.ones(len(future), 2, device=future.device, dtype=torch.bool)
    )
    out.gen = dataclasses.replace(
        base.gen, x0_tokens_vision=[out.gt_video], x0_tokens_action=[out.gt_action], raw_state_action=None
    )
    out.roles, out.chunks, out.sources = layout.action_metadata(device=future.device)
    return out


def reference(sampler, video, action, boundary, vs, acs):
    """Call the existing complete-prefix API and retain its actual training clean-pass KV."""
    # Adapter methods close over their MemoryState. Collect completed references
    # before constructing another large full-prefix oracle on the same GPU.
    gc.collect()
    model = sampler.model
    old = model.build_memory_state
    captured = []

    def capture(*args, **kwargs):
        memory = old(*args, **kwargs)
        captured.append(memory)
        return memory

    model.build_memory_state = capture
    try:
        result = sampler.forward(
            video,
            action,
            first_noisy=boundary.latent_start,
            end=boundary.latent_stop,
            video_sigmas=vs,
            action_sigmas=acs,
        )
        assert len(captured) == 1 and hasattr(
            captured[0], "_clean_gen_kv"
        ), "reference must use training teacher-forcing memory"
        return result, captured[0]._clean_gen_kv
    finally:
        model.build_memory_state = old
        for memory in captured:
            # No backward follows a numerical inference reference. Drop only its
            # instance wrappers to break cycles; returned clean K/V remain alive.
            for name in ("init", "read_for_layer", "write_for_layer"):
                memory.__dict__.pop(name, None)


def compare_kv(check, cache, ref, context, *, current_only=False):
    live = cache.ids >= 0
    if current_only:
        live &= cache.chunks == cache.current_chunk
    ids = cache.ids[live]
    for layer, (actual, expected) in enumerate(zip(cache.kv, ref, strict=True)):
        assert expected is not None
        for name, a, b in zip(("K", "V"), actual, expected, strict=True):
            check.close(a[:, live], b[:, ids], f"{context}/layer={layer}/{name}")


def check_window(cache, chunk, *, refreshed):
    expected = set(range(max(1, chunk - 15), chunk + 1))
    actual = set(cache.chunks[cache.ids >= 0].tolist())
    assert actual == expected, (chunk, actual, expected)
    roles, chunks, _ = cache.layout.metadata(device=cache.device)
    wanted = (chunks >= max(1, chunk - 15)) & (chunks <= chunk)
    if not refreshed:
        wanted &= (chunks < chunk) | ((roles != VIDEO) & (roles != ACTION))
    assert torch.equal(cache.ids[cache.ids >= 0].sort().values, torch.where(wanted)[0])
    assert len(cache.ids[cache.ids >= 0]) == len(cache.ids[cache.ids >= 0].unique())


def initialize(sampler, seed):
    gen = torch.Generator(device=sampler.gt_video.device).manual_seed(seed)
    vr, _, _ = sampler.layout.video_metadata(device=sampler.gt_video.device)
    video = torch.zeros_like(sampler.gt_video)
    action = torch.zeros_like(sampler.gt_action)
    video[:, :, vr == VIDEO] = torch.randn(video[:, :, vr == VIDEO].shape, device=video.device, generator=gen)
    action[sampler.roles == ACTION] = torch.randn(
        action[sampler.roles == ACTION].shape, device=action.device, generator=gen
    )
    action[:, 57:] = 0
    return video, action


def condition(sampler, video, action, boundary, history, terminal, previous):
    u = sampler.layout.video_indexes(boundary.chunk_id, True).to(video.device)
    rows = (sampler.roles == STATE) & (sampler.chunks == boundary.chunk_id)
    if history != "generated" or boundary.chunk_id == 1:
        terminal = decode_chunk_camera_state(
            sampler.gt_states[boundary.latent_start - 1], sampler.state_normalizer, source_index=boundary.source_start
        )
        video[:, :, u] = sampler.gt_video[:, :, u]
    else:
        video[:, :, u] = sampler._next_condition_video(previous)
    action[rows, :57] = _condition_encoding(terminal, sampler.state_normalizer)
    action[rows, 57:] = 0
    return terminal


@torch.no_grad()
def run_case(sampler, spec, output, seed=42):
    """Separate cache/reference buffers evolve independently. Same-input checks are additional forwards."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    check = Comparisons(output / "comparisons.jsonl")
    model = sampler.model
    layout = sampler.layout
    device = sampler.gt_video.device
    sv = _schedule(None, device)
    sa = sv.clone() if spec["schedule"] == "shift5" else _schedule(torch.linspace(1, 0, 31).square(), device)
    cv, ca = initialize(sampler, seed)
    rv, ra = initialize(sampler, seed)  # independent RNG instance, never reset to the other branch
    cache = JointKVCache(
        layout,
        num_layers=model.net.num_hidden_layers,
        num_kv_heads=model.net.num_kv_heads,
        head_dim=model.net.head_dim,
        device=device,
        dtype=model.tensor_kwargs["dtype"],
    )
    sampler.cache = cache
    sampler._cache_phase = None
    cfg = model.config.diffusion_expert_config
    sampler._cache_template = pack_joint_sequence(
        layout=layout,
        gen_data_clean=sampler.gen,
        text_ids=sampler.text[0],
        special_tokens=model.llm_special_tokens,
        timesteps=torch.zeros(layout.num_video_frames),
        latent_patch_size=cfg.patch_spatial,
        condition_frames=(),
        base_fps=cfg.base_fps,
        reset_spatial=cfg.unified_3d_mrope_reset_spatial_ids,
        modality_margin=cfg.unified_3d_mrope_temporal_modality_margin,
        initial_temporal_offset=sampler.memory_info["initial_temporal_offset"],
    )
    assert not cache.text_ready and not (cache.ids >= 0).any(), "episode reset must clear cache"
    sampler._cache_forward(cv, ca, [], chunk=0, phase="text")
    ct = rt = cp = rp = None
    gr, gc, _ = layout.metadata()
    report = dict(
        spec=spec, seed=seed, steps=30, history_chunks=15, chunks=[], status="running", scope="numerical_not_latency"
    )
    try:
        for b in layout.boundaries:
            context = f"C={spec['c']}/tail={spec['tail']}/{spec['history']}/{spec['schedule']}/chunk={b.chunk_id}"
            started = time.monotonic()
            before = cache.forward_calls
            ct = condition(sampler, cv, ca, b, spec["history"], ct, cp)
            rt = condition(sampler, rv, ra, b, spec["history"], rt, rp)
            u = layout.video_indexes(b.chunk_id, True).to(device)
            vi = layout.video_indexes(b.chunk_id, False).to(device)
            sr = (sampler.roles == STATE) & (sampler.chunks == b.chunk_id)
            ar = (sampler.roles == ACTION) & (sampler.chunks == b.chunk_id)
            target = torch.where((gc == b.chunk_id) & ((gr == VIDEO) | (gr == ACTION)))[0]
            check.close(cv[:, :, u], rv[:, :, u], context + "/independent/condition_U")
            check.close(ca[sr, :57], ra[sr, :57], context + "/independent/condition_S")
            sampler._cache_forward(
                cv, ca, layout.condition_prefill_indexes(b.chunk_id), chunk=b.chunk_id, phase="condition"
            )
            check_window(cache, b.chunk_id, refreshed=False)
            if b.chunk_id <= spec.get("observed_history_chunks", 0):
                # This is given GT history, not a shortened diffusion prediction.
                # Every subsequently predicted chunk still runs all 30 steps.
                cv[:, :, vi] = rv[:, :, vi] = sampler.gt_video[:, :, vi]
                ca[ar] = ra[ar] = sampler.gt_action[ar]
                sampler._cache_forward(cv, ca, target, chunk=b.chunk_id, phase="refresh")
                check_window(cache, b.chunk_id, refreshed=True)
                report["chunks"].append(dict(chunk=b.chunk_id, conditioning_only=True, forward_calls=2))
                continue
            vs = torch.zeros(layout.num_video_frames, device=device)
            acs = torch.zeros(layout.num_action_rows, device=device)
            _, kv = reference(sampler, cv, ca, b, vs, acs)
            compare_kv(check, cache, kv, context + "/same_input/prefill", current_only=True)
            del kv
            # All live history and current U/S must be bitwise unchanged by each noisy call.
            saved = [tuple(x[:, cache.ids >= 0].clone() for x in pair) for pair in cache.kv]
            for step in range(30):
                vs[vi] = sv[step]
                acs[ar] = sa[step]
                out = sampler._cache_forward(
                    cv, ca, target, chunk=b.chunk_id, phase="noisy", video_sigma=sv[step], action_sigma=sa[step]
                )
                pv = out["preds_vision"][0].float().reshape_as(cv[:, :, vi])
                pa = out["preds_action"][0].float().reshape(-1, 64)
                for layer, (pair, snapshot) in enumerate(zip(cache.kv, saved, strict=True)):
                    for kind, a, z in zip(("K", "V"), pair, snapshot, strict=True):
                        if not torch.equal(a[:, cache.ids >= 0], z):
                            raise AssertionError(f"{context}/step={step}/noisy mutated layer {layer} {kind}")
                (fv, fa), kv = reference(sampler, cv, ca, b, vs, acs)
                del kv
                check.close(pv, fv[:, :, vi], context + f"/step={step}/same_input/video_flow")
                check.close(pa[:, :57], fa[ar[: len(fa)], :57], context + f"/step={step}/same_input/action_flow")
                nv = cv[:, :, vi] + (sv[step + 1] - sv[step]) * pv
                na = ca[ar] + (sa[step + 1] - sa[step]) * pa
                check.close(
                    nv,
                    cv[:, :, vi] + (sv[step + 1] - sv[step]) * fv[:, :, vi],
                    context + f"/step={step}/same_input/video_euler",
                )
                check.close(
                    na[:, :57],
                    ca[ar, :57] + (sa[step + 1] - sa[step]) * fa[ar[: len(fa)], :57],
                    context + f"/step={step}/same_input/action_euler",
                )
                (iv, ia), kv = reference(sampler, rv, ra, b, vs, acs)
                del kv
                rv[:, :, vi] += (sv[step + 1] - sv[step]) * iv[:, :, vi]
                ra[ar] += (sa[step + 1] - sa[step]) * ia[ar[: len(ia)]]
                ra[:, 57:] = 0
                cv[:, :, vi] = nv
                ca[ar] = na
                ca[:, 57:] = 0
                del out, fv, fa, iv, ia
            del saved
            check.close(cv[:, :, vi], rv[:, :, vi], context + "/independent/final_video")
            check.close(ca[ar, :57], ra[ar, :57], context + "/independent/final_action")
            # Save each prediction before GT-history replacement, for both independent branches.
            torch.save(
                dict(
                    cache_video=cv[:, :, vi].cpu(),
                    reference_video=rv[:, :, vi].cpu(),
                    cache_action=ca[ar].cpu(),
                    reference_action=ra[ar].cpu(),
                ),
                output / f"chunk_{b.chunk_id:03d}.pt",
            )
            if spec["history"] == "generated":
                ct = decode_chunk_camera_action(ct, ca[ar], sampler.future_normalizer).end_state
                rt = decode_chunk_camera_action(rt, ra[ar], sampler.future_normalizer).end_state
                block = layout.video_indexes(b.chunk_id).to(device)
                cp = cv[:, :, block].clone()
                rp = rv[:, :, block].clone()
            if spec["history"] == "gt":
                cv[:, :, vi] = sampler.gt_video[:, :, vi]
                rv[:, :, vi] = sampler.gt_video[:, :, vi]
                ca[ar] = sampler.gt_action[ar]
                ra[ar] = sampler.gt_action[ar]
            sampler._cache_forward(cv, ca, target, chunk=b.chunk_id, phase="refresh")
            check_window(cache, b.chunk_id, refreshed=True)
            vs.zero_()
            acs.zero_()
            _, kv = reference(sampler, cv, ca, b, vs, acs)
            compare_kv(check, cache, kv, context + "/same_input/refresh", current_only=True)
            del kv
            _, kv = reference(sampler, rv, ra, b, vs, acs)
            compare_kv(check, cache, kv, context + "/independent/retained")
            del kv
            assert cache.forward_calls - before == 32
            record = dict(
                chunk=b.chunk_id,
                seconds=time.monotonic() - started,
                forward_calls=32,
                retained_chunks=sorted(set(cache.chunks[cache.ids >= 0].tolist())),
                comparisons=check.count,
                peak_allocated=torch.cuda.max_memory_allocated(),
            )
            report["chunks"].append(record)
            print(json.dumps(dict(case=spec, **record)), flush=True)
        report["status"] = "passed"
    finally:
        check.finish()
        report.update(comparisons=check.count)
        (output / "case.json").write_text(json.dumps(report, indent=2) + "\n")
        del sampler.cache, sampler._cache_template
    return report


def prepare_real_sampler(model, ds_cfg, chunk_size, root):
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.utils import misc
    from cosmos3_joint_video_hand_pose.src.ar_inference import _training_layout_batch

    frozen = Path(root) / "outputs/joint_video_hand_pose/ar_v0_2/data_v2/eval_windows.json"
    item = json.loads(frozen.read_text())[0]
    heldout = Path(root) / "outputs/joint_video_hand_pose/ar/eval/heldout_manifest"
    dataset = instantiate(
        ds_cfg,
        iterable_shuffle=False,
        random_window=False,
        cfg_dropout_rate=0.0,
        split="heldout",
        episodes_manifest=str(heldout / "episodes.csv"),
        segments_manifest=str(heldout / "segments.csv"),
    )
    raw = dataset.dataset
    ids = {f"{r['episode_hash']}:{r['span_index']}:{r['start_idx']}:{r['end_idx']}": i for i, r in enumerate(raw.rows)}
    index = ids[item["sample_id"]]
    row = raw.rows[index]
    assert item["frames"] == row["_clip_frames"] and item["start"] in row["_valid_starts"]
    batch = misc.to(
        _training_layout_batch(dataset.get_item_at_window(index, window_start=item["start"])), device="cuda"
    )
    sampler = JointARSampler(
        model,
        batch,
        state_normalizer=raw.chunk_state_normalizer,
        future_normalizer=raw.action_builder.future_normalizer,
        chunk_size=chunk_size,
        source_fps=float(raw.episodes[row["episode_hash"]]["fps"]),
    )
    return sampler, item
