"""CPU orchestration checks; real Nano/cache equivalence needs the GPU gate."""
import pytest
import torch
from cosmos3_ar_it2v.inference import cache_chunk_index, chunk_ranges, refresh_latents, rollout_chunks


def test_chunk_partition():
    assert chunk_ranges(17) == [(0,1),(1,5),(5,9),(9,13),(13,17)]
    assert chunk_ranges(2) == [(0,1),(1,2)]
    assert chunk_ranges(16) == [(0,1),(1,5),(5,9),(9,13),(13,16)]
    with pytest.raises(ValueError):
        chunk_ranges(0)


def test_refresh_copy_and_independent_rng():
    x = torch.ones(1,2,4,2,2)
    state = torch.random.get_rng_state()
    y = refresh_latents(x,.02,seed=42)
    assert torch.equal(torch.random.get_rng_state(),state)
    assert torch.equal(x,torch.ones_like(x))
    assert torch.equal(y,refresh_latents(x,.02,seed=42))
    assert not torch.equal(y,x)
    assert torch.equal(refresh_latents(x,0,seed=7),x)
    assert refresh_latents(x,0,seed=7).data_ptr() != x.data_ptr()


@pytest.mark.parametrize("sigma",[-.1,1.1,float("nan")])
def test_invalid_sigma(sigma):
    with pytest.raises(ValueError):
        refresh_latents(torch.zeros(1),sigma,seed=0)


def test_whole_chunk_refresh_clean_first_frame_absolute_positions_and_output():
    first = torch.ones(1,2,1,2,2)
    records = []
    def denoise(noise, *, start):
        records.append(("denoise",start,noise.shape[2]))
        return torch.full_like(noise,float(start))
    def refresh(value, *, start, sigma):
        records.append(("refresh",start,value.shape[2],sigma,value.clone()))
    out = rollout_chunks(first,17,chunk_size=4,seed=42,context_sigma=.1,denoise=denoise,refresh=refresh)
    assert [(r[1],r[2]) for r in records if r[0]=="denoise"] == [(1,4),(5,4),(9,4),(13,4)]
    writes=[r for r in records if r[0]=="refresh"]
    assert [(r[1],r[2],r[3]) for r in writes] == [(0,1,0),(1,4,.1),(5,4,.1),(9,4,.1)]
    assert torch.equal(writes[0][4],first)
    assert not torch.equal(writes[1][4],out[:,:,1:5])
    for start,end in chunk_ranges(17)[1:]:
        assert torch.equal(out[:,:,start:end],torch.full_like(out[:,:,start:end],float(start)))


def test_sigma_does_not_change_output_in_history_agnostic_stub():
    first=torch.ones(1,1,1,1,1)
    def run(sigma):
        return rollout_chunks(first,9,chunk_size=4,seed=42,context_sigma=sigma,
                              denoise=lambda x,**kw:x,refresh=lambda *a,**kw:None)
    assert torch.equal(run(0),run(.1))


def test_cache_chunks_are_contiguous_while_rope_positions_are_absolute():
    assert [cache_chunk_index(i) for i in (0,1,5,9,13)] == [0,1,2,3,4]
    with pytest.raises(ValueError):
        cache_chunk_index(4)


def test_partial_final_chunk_keeps_every_latent():
    calls=[]
    first=torch.ones(1,2,1,2,2)
    def denoise(x, *, start):
        calls.append((start,x.shape[2]))
        return torch.full_like(x,start)
    out=rollout_chunks(first,19,chunk_size=4,seed=42,context_sigma=.02,
                       denoise=denoise,refresh=lambda *a,**kw:None)
    assert calls==[(1,4),(5,4),(9,4),(13,4),(17,2)]
    assert out.shape[2]==19 and (out[:,:,17:]==17).all()


def test_native_rope_full_video_and_chunk_packs_align_at_30fps():
    from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive as pack
    from cosmos_framework.data.generator.sequence_packing.modality import compute_text_split_length
    special={'eos_token_id':2,'start_of_generation':3,'end_of_generation':4}
    x=torch.zeros(1,2,19,2,2)
    kw=dict(action_latent=None,timestep=300.,fps_vision=[30.],fps_action=None,
        special_tokens=special,latent_patch_size=1,video_temporal_causal=True,
        temporal_compression_factor=4,enable_fps_modulation=True,base_fps=24.,
        unified_3d_mrope_temporal_modality_margin=2,force_action_tokens=False)
    full=pack(vision_latent=x,text_tokens=[7,8],frame_idx=0,condition_frame_indexes_vision=[0],**kw)
    expected=full.position_ids[:,full.vision.sequence_indexes]
    assert torch.allclose(expected[0,4::4]-expected[0,:-4:4],torch.full((18,),.8),atol=2e-6)
    for start,end in chunk_ranges(19):
        part=pack(vision_latent=x[:,:,start:end],text_tokens=[7,8] if start==0 else None,
            frame_idx=start,condition_frame_indexes_vision=[0] if start==0 else [],
            cached_text_offset=None if start==0 else compute_text_split_length(2,special),**kw)
        actual=part.position_ids[:,part.vision.sequence_indexes]
        assert torch.allclose(actual,expected[:,start*4:end*4],atol=2e-6)


@pytest.mark.parametrize('local_frames', [16, 4])
def test_native_chunk_cache_bounds_storage_and_matches_training_history(local_frames):
    from cosmos3_ar_it2v.inference import _make_chunk_cache
    from cosmos_framework.model.generator.utils.kv_cache import ARMemoryState

    chunk_size, patches_per_frame = 4, 2
    cache = _make_chunk_cache(chunk_size, local_frames)
    history_frames = local_frames - chunk_size
    assert cache.gen_cache.cache_size == (4 if local_frames == 16 else 2)
    # Cross the ring boundary many times, ending with a two-latent partial chunk.
    ranges = chunk_ranges(83, chunk_size)
    assert ranges[-1] == (81, 83)
    for start, end in ranges:
        index = cache_chunk_index(start, chunk_size)
        memory = ARMemoryState(dual_kv_cache=[cache], frame_idx=index,
            vision_token_shapes=[(end-start, 1, patches_per_frame)],
            transfer_history_sink_tokens=0,
            transfer_history_max_tokens=history_frames * patches_per_frame)
        value = memory.read_for_layer(0)
        expected = torch.arange(max(0, start-history_frames), start).repeat_interleave(patches_per_frame)
        if expected.numel():
            assert torch.equal(value.gen_k_hist.flatten(), expected.float())
            assert torch.equal(value.gen_v_hist.flatten(), -expected.float())
        else:
            # Native max_tokens=0 must return no history, not Python's [-0:] slice.
            assert value.gen_k_hist is None and value.gen_v_hist is None
        current = torch.arange(start, end, dtype=torch.float32).repeat_interleave(patches_per_frame)
        current = current.reshape(1, -1, 1, 1)
        cache.gen_cache.store_kv(current, -current, frame_idx=index)
        for storage in (cache.gen_cache.k_cache, cache.gen_cache.v_cache):
            assert len(storage) == cache.gen_cache.cache_size
            assert sum(t.numel() for t in storage if t is not None) <= cache.gen_cache.cache_size * chunk_size * patches_per_frame
