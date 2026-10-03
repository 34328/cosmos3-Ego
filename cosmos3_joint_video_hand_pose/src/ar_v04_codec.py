"""V0.4 continuous VAE semantics carried in the existing joint token layout.

U rows are aliases of continuous latent boundaries, not fresh image encodes.
Only RGB is sliced for the existing renderer, after one native full decode.
"""

import torch

VIDEO_LATENT_FORMAT = "continuous_vae_joint_v1"


def require_video_format(value):
    if value != VIDEO_LATENT_FORMAT:
        raise ValueError("V0.4 requires continuous_vae_joint_v1; legacy block-reset video is incompatible")


def gather_joint_latents(layout, continuous):
    """Duplicate boundary aliases without resetting the native temporal VAE."""
    if continuous.ndim != 5 or continuous.shape[2] != layout.num_frames:
        raise ValueError("continuous latent length does not match the joint layout")
    if not torch.isfinite(continuous).all():
        raise ValueError("nonfinite continuous video latents")
    return torch.cat([continuous[:, :, b.latent_start - 1:b.latent_stop]
                      for b in layout.boundaries], dim=2)


def continuous_video_latents(layout, packed, *, video_latent_format, history):
    """Remove U aliases; GT-conditioned targets need not equal their GT U rows."""
    require_video_format(video_latent_format)
    if history not in ("gt", "generated", "oracle", "pred_history"):
        raise ValueError("unknown joint history mode")
    if packed.ndim != 5 or packed.shape[2] != layout.num_video_frames:
        raise ValueError("packed video length does not match the joint layout")
    if not torch.isfinite(packed).all():
        raise ValueError("nonfinite packed video latents")
    parts = []
    for b in layout.boundaries:
        indexes = layout.video_indexes(b.chunk_id).to(packed.device)
        block = packed.index_select(2, indexes)
        if not parts:
            parts.append(block[:, :, :1])
        elif history in ("generated", "oracle") and not torch.equal(block[:, :, :1], parts[-1][:, :, -1:]):
            raise ValueError("continuous generated boundary U must equal previous final latent")
        parts.append(block[:, :, 1:])
    continuous = torch.cat(parts, dim=2)
    if continuous.shape[2] != layout.num_frames:
        raise ValueError("unexpected deduplicated video length")
    return continuous


@torch.no_grad()
def decode_continuous_video_chunks(model, layout, packed, *, video_latent_format, history):
    """Native full-sequence decode, then overlapping RGB views for old replay math.

    In gt mode each target was conditioned on GT history. Those U rows are not
    inserted into the displayed predictions; this is still teacher-conditioned
    evaluation, not an autonomous rollout, despite the continuous RGB decode.
    """
    require_video_format(getattr(model.config, "video_latent_format", None))
    continuous = continuous_video_latents(layout, packed,
        video_latent_format=video_latent_format, history=history)
    decoded = model.decode(continuous.to(model.tensor_kwargs["dtype"]))
    expected_frames = 1 + 4 * (layout.num_frames - 1)
    if decoded.ndim != 5 or decoded.shape[0] != 1 or decoded.shape[2] != expected_frames:
        raise ValueError("native full video decode returned an incompatible temporal length")
    if not torch.isfinite(decoded).all():
        raise ValueError("nonfinite decoded video")
    rgb = ((decoded[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()
    # Source action rows are stride-1, video is stride-2. Keep this alignment.
    return [rgb[b.source_start // 2:b.source_stop // 2 + 1] for b in layout.boundaries]
