"""GT-history diagnostic: completed continuous slices change KV, not outputs."""
import pytest
import torch

from cosmos3_ar_it2v.inference import (
    _make_chunk_cache, cache_chunk_index, refresh_latents, rollout_chunks,
)


def reference(frames=19):
    return torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1)


def run(gt, **kwargs):
    calls, writes = [], []

    def denoise(noise, *, start):
        calls.append((start, noise.clone()))
        return torch.full_like(noise, -start)

    def refresh(value, *, start, sigma):
        writes.append((start, sigma, value.clone()))

    output = rollout_chunks(gt[:, :, :1], gt.shape[2], chunk_size=4, seed=42,
                            context_sigma=.02, denoise=denoise, refresh=refresh, **kwargs)
    return output, calls, writes


def test_modes_keep_target_noise_predictions_and_generated_default_unchanged():
    gt = reference()
    default, calls, writes = run(gt)
    explicit, explicit_calls, explicit_writes = run(gt, history_mode='generated')
    oracle, oracle_calls, oracle_writes = run(gt, history_mode='gt', gt_latents=gt)
    assert torch.equal(default, explicit) and torch.equal(default, oracle)
    assert [start for start, _ in calls] == [1, 5, 9, 13, 17]
    for (start, a), (_, b), (_, c) in zip(calls, explicit_calls, oracle_calls, strict=True):
        assert torch.equal(a, b) and torch.equal(a, c)
        expected = torch.randn(a.shape, generator=torch.Generator().manual_seed(42 + start))
        assert torch.equal(a, expected)
    assert [(start, sigma) for start, sigma, _ in oracle_writes] == [
        (0, 0), (1, .02), (5, .02), (9, .02), (13, .02)]
    assert torch.equal(oracle_writes[0][2], gt[:, :, :1])
    for (start, sigma, a), (_, _, b), (_, _, c) in zip(
            writes[1:], explicit_writes[1:], oracle_writes[1:], strict=True):
        assert torch.equal(a, b)
        assert torch.equal(a, refresh_latents(default[:, :, start:start + 4], sigma, seed=100042 + start))
        assert torch.equal(c, refresh_latents(gt[:, :, start:start + 4], sigma, seed=100042 + start))
        assert not torch.equal(c, oracle[:, :, start:start + 4])
    assert (oracle[:, :, 17:] == -17).all()  # Partial final block remains predicted.
    assert torch.equal(gt, reference())


def test_gt_refresh_reads_only_completed_blocks_and_native_cache_contains_no_prediction():
    from cosmos_framework.model.generator.utils.kv_cache import ARMemoryState

    gt = reference(31)
    gt[:, :, 29:] = float('nan')  # Target/future GT is never read or refreshed.
    cache = _make_chunk_cache(4, 16)
    completed, written, events = [], [], []

    def denoise(noise, *, start):
        memory = ARMemoryState(dual_kv_cache=[cache], frame_idx=cache_chunk_index(start),
            vision_token_shapes=[(noise.shape[2], 1, 1)], transfer_history_sink_tokens=0,
            transfer_history_max_tokens=12)
        actual = memory.read_for_layer(0).gen_k_hist.flatten()
        assert torch.equal(actual, torch.cat(written)[-12:])
        completed.append(start)
        events.append(('predict', start))
        return torch.full_like(noise, -start)

    def refresh(value, *, start, sigma):
        if start:
            assert completed[-1] == start
            expected = refresh_latents(gt[:, :, start:start + value.shape[2]], sigma,
                                       seed=100042 + start)
            assert torch.equal(value, expected)
        else:
            assert sigma == 0 and torch.equal(value, gt[:, :, :1])
        events.append(('write', start))
        written.append(value.flatten())
        keys = value.flatten().reshape(1, -1, 1, 1)
        cache.gen_cache.store_kv(keys, keys, frame_idx=cache_chunk_index(start))

    result = rollout_chunks(gt[:, :, :1], 31, chunk_size=4, seed=42, context_sigma=.02,
                            denoise=denoise, refresh=refresh, history_mode='gt', gt_latents=gt)
    assert torch.isfinite(result).all() and (result[:, :, 1:] < 0).all()
    assert result.shape[2] == 31 and (result[:, :, 29:] == -29).all()
    assert events[0] == ('write', 0) and events[-1] == ('predict', 29)
    for i in range(1, len(events) - 1, 2):
        assert events[i][0] == 'predict' and events[i + 1] == ('write', events[i][1])


def test_single_condition_frame():
    gt = reference(1)
    result, calls, writes = run(gt, history_mode='gt', gt_latents=gt)
    assert torch.equal(result, gt) and calls == []
    assert len(writes) == 1 and writes[0][:2] == (0, 0)


@pytest.mark.parametrize('mode', [None, 'unknown', 'teacher_forcing'])
def test_invalid_mode(mode):
    with pytest.raises(ValueError, match='history_mode'):
        run(reference(), history_mode=mode)


@pytest.mark.parametrize('gt', [None, reference(8), reference(19).double(), reference(19) + 1])
def test_invalid_gt_reference(gt):
    with pytest.raises(ValueError, match='GT history'):
        run(reference(), history_mode='gt', gt_latents=gt)


def test_generated_rejects_ambiguous_gt_argument():
    with pytest.raises(ValueError, match='only accepted'):
        run(reference(), gt_latents=reference())
