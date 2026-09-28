"""Small full-Nano precision diagnostic, not rollout/latency acceptance."""

import argparse
import json
from pathlib import Path
import torch
from check_ar_v02_nano_cache import load_model, ROOT, DEFAULT_CKPT, fingerprint
from ar_v02_nano_validation import prepare_real_sampler, extend_latents, initialize, condition, reference
from cosmos3_joint_video_hand_pose.src.ar_v02_cache import JointKVCache, StreamingJointAttention
from cosmos3_joint_video_hand_pose.src.ar_v02_attention import JointTeacherForcingAttention
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence


def errors(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return dict(
        max_abs=d.max().item() if d.numel() else 0.0,
        relative_l2=((a - b).norm() / b.norm().clamp_min(1e-6)).item(),
        strict_bf16=bool((d <= 0.01 + 0.03 * b.abs()).all()),
        strict_fp32=bool((d <= 1e-5 + 1e-5 * b.abs()).all()),
    )


def fp32_text_attention(q, k, v, offsets):
    from cosmos_framework.data.generator.sequence_packing.runtime import get_causal_seq
    from torch.nn.attention import sdpa_kernel, SDPBackend

    q, k, v = [get_causal_seq(x)[0] for x in (q, k, v)]
    out = torch.zeros_like(q)
    for lo, hi in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        with sdpa_kernel(SDPBackend.MATH):
            result = torch.nn.functional.scaled_dot_product_attention(
                q[lo:hi].transpose(0, 1)[None],
                k[lo:hi].transpose(0, 1)[None],
                v[lo:hi].transpose(0, 1)[None],
                is_causal=True,
                enable_gqa=q.shape[1] != k.shape[1],
            )
        out[lo:hi] = result[0].transpose(0, 1)
    return out.flatten(-2, -1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["bf16", "fp32", "attention_fp32", "fixed_tile", "canonical", "ordered"], default="bf16"
    )
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--toml", type=Path, default=ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_2.toml")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", mode=args.mode, source=fingerprint(), scope="first_chunk_precision_diagnostic")
    try:
        torch.set_num_threads(1)
        model, ds = load_model(args)
        model.compact_noisy_training = False
        base, _ = prepare_real_sampler(model, ds, 4, ROOT)
        sampler = extend_latents(base, 1, 1)
        if args.mode == "fp32":
            model.net.float()
            model.precision = torch.float32
            model.tensor_kwargs["dtype"] = torch.float32
            # NATTEN only accepts low precision; the text-only math fallback is diagnostic.
            import cosmos_framework.model.generator.mot.causal_attention as ca

            ca._three_way_text_self_attention = fp32_text_attention
        if args.mode in ("attention_fp32", "fixed_tile", "canonical", "ordered"):
            import cosmos_framework.model.generator.mot.flex_attention as flex

            original = flex._COMPILED_FLEX_ATTENTION

            def precise(q, k, v, **kw):
                if args.mode == "ordered":
                    from torch.nn.attention.flex_attention import BlockMask

                    bm = kw["block_mask"]
                    dense = bm.to_dense()
                    counts = dense.sum(-1).to(torch.int32)
                    indexes = torch.argsort(dense.to(torch.int32), dim=-1, descending=True, stable=True).to(torch.int32)
                    kw["block_mask"] = BlockMask.from_kv_blocks(
                        counts, indexes, BLOCK_SIZE=bm.BLOCK_SIZE, mask_mod=bm.mask_mod, seq_lengths=bm.seq_lengths
                    )
                    return original(q, k, v, **kw)
                if args.mode == "canonical":
                    from torch.nn.attention import sdpa_kernel, SDPBackend

                    qi = torch.arange(q.shape[2], device=q.device)[:, None]
                    ki = torch.arange(k.shape[2], device=k.device)[None, :]
                    allowed = kw["block_mask"].mask_mod(0, 0, qi, ki)
                    patterns, groups = torch.unique(allowed, dim=0, return_inverse=True)
                    out = torch.zeros_like(q)
                    for i, pattern in enumerate(patterns):
                        rows = torch.where(groups == i)[0]
                        keys = torch.where(pattern)[0]
                        if not len(keys):
                            continue
                        with sdpa_kernel(SDPBackend.MATH):
                            out[:, :, rows] = torch.nn.functional.scaled_dot_product_attention(
                                q[:, :, rows], k[:, :, keys], v[:, :, keys], enable_gqa=q.shape[1] != k.shape[1]
                            )
                    return out
                if args.mode == "fixed_tile":
                    return original(q, k, v, **kw, kernel_options={"BLOCK_M": 64, "BLOCK_N": 64})
                return original(q.float(), k.float(), v.float(), **kw).to(q.dtype)

            flex._COMPILED_FLEX_ATTENTION = precise
        # Record first two GEN attention calls per branch. Text prefill has no GEN rows.
        captures = {}
        counts = {}
        for cls, label in [(StreamingJointAttention, "cache"), (JointTeacherForcingAttention, "reference")]:
            old = cls.__call__

            def wrapper(self, q, k, v, tk, tv, mem, _old=old, _label=label):
                out = _old(self, q, k, v, tk, tv, mem)
                if q.numel():
                    n = counts.get(_label, 0)
                    counts[_label] = n + 1
                    if n < 2:
                        values = [x.reshape(-1, x.shape[-2], x.shape[-1])[:241].float().cpu() for x in (q, k, v, out)]
                        text_len = int(mem.und_kv_offsets[-1])
                        values += [x[:, :text_len].float().cpu() for x in (tk, tv)]
                        captures[(_label, n)] = values
                return out

            cls.__call__ = wrapper
        layout = sampler.layout
        dev = sampler.gt_video.device
        cache = JointKVCache(
            layout,
            num_layers=model.net.num_hidden_layers,
            num_kv_heads=model.net.num_kv_heads,
            head_dim=model.net.head_dim,
            device=dev,
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
        video, action = initialize(sampler, 42)
        b = layout.boundaries[0]
        condition(sampler, video, action, b, "gt", None, None)
        sampler._cache_forward(video, action, [], chunk=0, phase="text")
        sampler._cache_forward(video, action, layout.condition_prefill_indexes(1), chunk=1, phase="condition")
        vs = torch.zeros(layout.num_video_frames, device=dev)
        acs = torch.zeros(layout.num_action_rows, device=dev)
        _, kv = reference(sampler, video, action, b, vs, acs)
        live = cache.ids >= 0
        ids = cache.ids[live]
        report["layers"] = [
            dict(layer=i, **{name: errors(a[:, live], z[:, ids]) for name, a, z in zip(("K", "V"), pair, ref)})
            for i, (pair, ref) in enumerate(zip(cache.kv, kv))
        ]
        report["attention"] = [
            dict(
                layer=i,
                **{
                    name: errors(a, z)
                    for name, a, z in zip(
                        ("Q", "K", "V", "out", "text_K", "text_V"), captures["cache", i], captures["reference", i]
                    )
                }
            )
            for i in range(2)
        ]
        report["status"] = "complete"
    except BaseException as e:
        report.update(status="failed", error=repr(e))
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
