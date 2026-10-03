"""Pure-video chunk causality, locality, packing isolation and history gradients."""
import torch
import torch.nn.functional as F
import pytest
from cosmos3_ar_it2v.attention import chunk_ids, causal_video_visibility, ChunkCausalAttention


def test_chunk_partition_and_complete_future_visibility():
    f=torch.arange(21)
    assert chunk_ids(f,4).tolist()==[0]+[1]*4+[2]*4+[3]*4+[4]*4+[5]*4
    m=causal_video_visibility(f[:,None],f[None,:])
    assert m[0].nonzero().flatten().tolist()==[0]
    assert m[1].nonzero().flatten().tolist()==list(range(5))
    assert torch.equal(m[1],m[4])
    assert m[17].nonzero().flatten().tolist()==list(range(5,21))
    assert not m[17,0]  # no permanent image sink


def test_packed_mask_isolates_samples_and_padding():
    a=ChunkCausalAttention([(5,1,1),(9,1,1)],[3,2],device='cpu')
    q=torch.arange(a.gen_pad)[:,None]
    k=torch.arange(128+a.gen_pad)[None,:]
    mask=a.mask_mod(128)(0,0,q,k)
    assert mask[1,:3].all() and not mask[1,3:128].any()
    assert not mask[:5,133:].any()
    assert not mask[5:14,128:133].any()
    assert not mask[14:].any()
    assert not mask[:,142:].any()


def test_future_loss_reaches_history_but_not_future_tokens():
    torch.manual_seed(7)
    x=torch.randn(1,1,13,8,requires_grad=True)
    f=torch.arange(13)
    m=causal_video_visibility(f[:,None],f[None,:])
    y=F.scaled_dot_product_attention(x,x,x,attn_mask=m)
    y[:,:,5:9].square().sum().backward()
    assert x.grad[:,:,1:5].abs().sum()>0
    assert x.grad[:,:,9:].abs().sum()==0


def test_block_mask_build_cpu():
    a=ChunkCausalAttention([(5,2,2)],[3],device='cpu')
    assert a.block_mask(128).shape[-2:] == (128,256)


def test_partial_tail_nominal_window_and_packed_isolation():
    a=ChunkCausalAttention([(19,1,1),(2,1,1)],[3,2],device='cpu')
    mask=a.mask_mod(128)(0,0,torch.arange(a.gen_pad)[:,None],torch.arange(128+a.gen_pad)[None,:])
    # Partial C4 at 17:19 keeps 12 history frames, matching inference cache.
    assert mask[17,128:].nonzero().flatten().tolist()==list(range(5,19))
    assert torch.equal(mask[17],mask[18])
    assert not mask[:19,147:].any() and not mask[19:21,128:147].any()


@pytest.mark.parametrize('shapes,text_lengths', [
    ([(19,1,17),(6,1,31),(2,1,65)], [129,7,131]),
    ([(21,2,8),(4,1,33)], [3,125]),
])
def test_metadata_block_mask_matches_original_dense_builder(shapes, text_lengths):
    from torch.nn.attention.flex_attention import create_block_mask
    a=ChunkCausalAttention(shapes,text_lengths,device='cpu')
    text_pad=((sum(text_lengths)+127)//128)*128
    old=create_block_mask(a.mask_mod(text_pad),B=None,H=None,
        Q_LEN=a.gen_pad,KV_LEN=text_pad+a.gen_pad,device='cpu',BLOCK_SIZE=128,_compile=False)
    new=a.block_mask(text_pad)
    assert torch.equal(new.to_dense(),old.to_dense())
    # Preserve the existing ascending, all-partial traversal, including full tiles.
    dense=old.to_dense()
    assert torch.equal(new.kv_num_blocks,dense.sum(-1).to(torch.int32))
    assert torch.equal(new.kv_indices,torch.argsort(dense.to(torch.int32),dim=-1,
        descending=True,stable=True).to(torch.int32))
    assert new.full_kv_num_blocks is None
    q=torch.arange(a.gen_pad)[:,None]
    k=torch.arange(text_pad+a.gen_pad)[None,:]
    assert torch.equal(new.mask_mod(0,0,q,k),old.mask_mod(0,0,q,k))


def test_large_layout_constructs_only_metadata_pair_matrix(monkeypatch):
    a=ChunkCausalAttention([(301,15,16),(7,15,16)],[511,257],device='cpu')
    original=a.mask_mod
    evaluated_pairs=[]
    def tracked_mask_mod(text_pad):
        predicate=original(text_pad)
        def tracked(b,h,q,k):
            evaluated_pairs.append(q.numel()*k.numel())
            assert evaluated_pairs[-1] < 1_000_000
            return predicate(b,h,q,k)
        return tracked
    monkeypatch.setattr(a,'mask_mod',tracked_mask_mod)
    mask=a.block_mask(768)
    assert a.flat_gen_tokens==73920
    assert mask.shape[-2:]==(a.gen_pad,768+a.gen_pad)
    assert len(evaluated_pairs)==1
    assert a.block_mask(768) is mask  # Build once per layout, reused by all layers.
