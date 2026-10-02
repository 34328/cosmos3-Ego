"""CPU orchestration checks; real Nano/cache equivalence needs the GPU gate."""
import pytest
import torch
from cosmos3_ar_it2v.inference import cache_chunk_index, chunk_ranges, refresh_latents, rollout_chunks


def test_chunk_partition():
    assert chunk_ranges(17) == [(0,1),(1,5),(5,9),(9,13),(13,17)]
    for n in (0,2,16):
        with pytest.raises(ValueError):
            chunk_ranges(n)


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
