"""Pure-video chunk causality, locality, packing isolation and history gradients."""
import torch
import torch.nn.functional as F
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
