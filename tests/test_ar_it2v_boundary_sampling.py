"""CPU contracts for the isolated training-free boundary condition probe."""
from types import SimpleNamespace

import pytest
import torch

from cosmos3_ar_it2v.inference import (
    _make_chunk_cache, cache_chunk_index, chunk_ranges, refresh_latents, rollout_chunks,
)
from visualization.boundary_sampling import (
    BoundaryMemoryState, generate_boundary_latents, rollout_boundary_chunks, sample_known_boundary,
)


def reference(frames=19):
    return torch.arange(frames, dtype=torch.float32).reshape(1, 1, frames, 1, 1)


@pytest.mark.parametrize('mode', ['gt', 'generated'])
def test_boundary_is_prior_prediction_at_same_time_only_and_rng_matches_baseline(mode):
    gt = reference()
    calls, writes = [], []
    def denoise(noise, *, start, boundary):
        if start == 1:
            assert boundary is None
        else:
            assert torch.equal(boundary, torch.full_like(boundary, -(start-4)))
            assert not torch.equal(boundary, gt[:, :, start-1:start])
        expected = torch.randn(noise.shape, generator=torch.Generator().manual_seed(42+start))
        assert torch.equal(noise, expected)
        calls.append((start, noise.clone()))
        return torch.full_like(noise, -start)
    def refresh(value, *, start, sigma):
        writes.append((start, value.clone(), sigma))
    out = rollout_boundary_chunks(gt[:, :, :1], 19, chunk_size=4, seed=42, context_sigma=.02,
        denoise=denoise, refresh=refresh, history_mode=mode, gt_latents=gt if mode == 'gt' else None)
    baseline = rollout_chunks(gt[:, :, :1], 19, chunk_size=4, seed=42, context_sigma=.02,
        denoise=lambda noise, *, start: torch.full_like(noise, -start),
        refresh=lambda *a, **k: None, history_mode=mode, gt_latents=gt if mode == 'gt' else None)
    assert torch.equal(out, baseline)  # No duplicate boundary output or changed partition.
    assert [(s, x.shape[2]) for s, x in calls] == [(1,4),(5,4),(9,4),(13,4),(17,2)]
    assert [s for s, _, _ in writes] == [0,1,5,9,13]
    for start, actual, sigma in writes[1:]:
        source = gt[:, :, start:start+4] if mode == 'gt' else out[:, :, start:start+4]
        assert torch.equal(actual, refresh_latents(source, sigma, seed=100042+start))
    assert torch.equal(gt, reference())


def test_native_memory_excludes_last_history_time_once_without_mutating_cache():
    from cosmos_framework.model.generator.utils.kv_cache import ARMemoryState
    cache = _make_chunk_cache(4, 16)
    for start, end in chunk_ranges(35):
        native = ARMemoryState(dual_kv_cache=[cache], frame_idx=cache_chunk_index(start),
            vision_token_shapes=[(end-start+1,1,2)], write_gen_cache=False,
            transfer_history_sink_tokens=0, transfer_history_max_tokens=24)
        if start > 1:
            memory = BoundaryMemoryState(native, 2)
            before = [x.clone() if x is not None else None for x in cache.gen_cache.k_cache]
            value = memory.read_for_layer(0)
            expected = torch.arange(max(0,start-12), start-1).repeat_interleave(2).float()
            assert torch.equal(value.gen_k_hist.flatten(), expected)
            assert torch.equal(value.gen_v_hist.flatten(), -expected)
            assert expected.numel() + (end-start+1)*2 <= 32
            for a,b in zip(before, cache.gen_cache.k_cache, strict=True):
                assert (a is None and b is None) or torch.equal(a,b)
        x = torch.arange(start,end).repeat_interleave(2).float().reshape(1,-1,1,1)
        cache.gen_cache.store_kv(x,-x,frame_idx=cache_chunk_index(start))


def test_memory_rejects_sampling_writes():
    with pytest.raises(ValueError, match='never write'):
        BoundaryMemoryState(SimpleNamespace(write_gen_cache=True), 1)


def make_pack(value, start, condition):
    from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive
    return pack_input_sequence_autoregressive(
        vision_latent=value, action_latent=None, text_tokens=None, timestep=1000.,
        fps_vision=[30.], fps_action=None,
        special_tokens={'eos_token_id':2,'start_of_generation':3,'end_of_generation':4},
        latent_patch_size=1, condition_frame_indexes_vision=[0] if condition else [],
        frame_idx=start, temporal_compression_factor=4, video_temporal_causal=True,
        cached_text_offset=3, force_action_tokens=False)


def native_velocity_items(pack, scalar):
    """Exercise the real model output reconstruction, including batch singleton.

    Native model predictions are list[[1,C,T,H,W]], not the older denoise
    docstring's list[[C,T,H,W]]. Condition frames are restored as zeros.
    """
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
    network = SimpleNamespace(latent_patch_size=1,latent_channel=1)
    patches = torch.full((len(pack.vision.mse_loss_indexes),1),scalar)
    return Cosmos3VFMNetwork.unpatchify_and_unpack_latents(network,patches,
        pack.vision.token_shapes,pack.vision.noisy_frame_indexes)


@pytest.mark.parametrize('guidance', [1., 2.])
def test_actual_official_unipc_and_cfg_keep_boundary_clean_and_outside_solver(monkeypatch, guidance):
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
    from cosmos_framework.data.generator.sequence_packing import PackedSequence
    # Native packs remain on CPU for this orchestration test.
    monkeypatch.setattr(PackedSequence, 'to_cuda', lambda self: self)
    seen = []
    known = torch.full((1,1,1,1,1),77.)
    noise = torch.arange(4.).reshape(1,1,4,1,1)
    packs = [make_pack(torch.cat((known,noise),2),4,True) for _ in range(1 if guidance==1 else 2)]
    class Model:
        tensor_kwargs = dict(device='cpu',dtype=torch.float32)
        parallel_dims = None
        config = SimpleNamespace(sigma_shift=5.,
            rectified_flow_inference_config=SimpleNamespace(scheduler_type='unipc'))
        sampler = UniPCSampler(tensor_kwargs=tensor_kwargs)
        def denoise(self, *, data_batch_packed, memory):
            pack = data_batch_packed
            value = pack.vision.tokens[0]
            assert torch.equal(value[:, :, :1], known)
            assert len(pack.vision.timesteps) == 4
            assert torch.equal(pack.vision.condition_mask[0].flatten(),torch.tensor([1.,0.,0.,0.,0.]))
            assert (pack.vision.timesteps > 0).all()
            seen.append(value.clone())
            result = native_velocity_items(pack,2. if pack is packs[0] else 0.)
            assert result[0].shape == value.shape
            assert torch.count_nonzero(result[0][:, :, :1]) == 0
            return {'preds_vision':result}
    result = sample_known_boundary(Model(), noise, known, packed_sequences=packs,
        memories=[object() for _ in packs], guidance=guidance, num_steps=7, seed=42, start=5,latent_frames=19)
    # Compare against the exact native shifted schedule, including its nonzero
    # initial sigma floor; don't invent an idealized sigma=1 endpoint.
    expected = Model().sampler(lambda state,t:torch.full_like(state,2*guidance),
        noise.flatten(start_dim=1),num_steps=7,shift=5.,seed=47).reshape(noise.shape)
    assert torch.equal(result, expected)
    assert len(seen) == 7*len(packs)
    assert result.shape == noise.shape and torch.equal(known, torch.full_like(known,77.))


@pytest.mark.parametrize('mode', ['gt','generated'])
def test_real_entry_continuous_gt_encode_once_boundary_time_and_refresh_contract(monkeypatch, mode):
    from cosmos_framework.data.generator.sequence_packing import PackedSequence
    from cosmos_framework.model.generator.utils.kv_cache import ARMemoryState
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
    monkeypatch.setattr(PackedSequence,'to_cuda',lambda self:self)
    gt = reference(11)
    from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive
    full = pack_input_sequence_autoregressive(vision_latent=gt,action_latent=None,
        text_tokens=[7,8],timestep=1000.,fps_vision=[30.],fps_action=None,
        special_tokens={'eos_token_id':2,'start_of_generation':3,'end_of_generation':4},
        latent_patch_size=1,condition_frame_indexes_vision=[0],frame_idx=0,
        temporal_compression_factor=4,video_temporal_causal=True,
        enable_fps_modulation=True,base_fps=24.,
        unified_3d_mrope_temporal_modality_margin=2,force_action_tokens=False)
    full_positions = full.position_ids[:,full.vision.sequence_indexes]
    events = []
    class Model:
        parallel_dims = None
        tensor_kwargs = dict(device='cpu',dtype=torch.float32)
        config = SimpleNamespace(action_gen=False,compile=SimpleNamespace(enabled=False),
            frames_per_chunk=4,local_attention_frames=16,sigma_shift=5.,max_action_dim=32,
            diffusion_expert_config=SimpleNamespace(patch_spatial=1,enable_fps_modulation=True,
                base_fps=24.,unified_3d_mrope_temporal_modality_margin=2),
            rectified_flow_inference_config=SimpleNamespace(scheduler_type='unipc'))
        net = SimpleNamespace(num_hidden_layers=1)
        tokenizer_vision_gen = SimpleNamespace(temporal_compression_factor=4)
        rectified_flow_video = SimpleNamespace(noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)))
        llm_special_tokens = {'eos_token_id':2,'start_of_generation':3,'end_of_generation':4}
        sampler = UniPCSampler(tensor_kwargs=tensor_kwargs)
        def get_data_and_condition(self,batch,*,vision_condition_indexes):
            assert vision_condition_indexes is None
            events.append(('encode',))
            return SimpleNamespace(batch_size=1,x0_tokens_action=None,x0_tokens_vision=[gt],fps_vision=torch.tensor([30.]))
        def _get_inference_text_tokens(self,*a):
            return [[7,8]],[[]]
        def build_memory_state(self,packed,info):
            return ARMemoryState(vision_token_shapes=packed.vision.token_shapes,**{
                key:value for key,value in info.items() if key!='use_ar_rolling'})
        def generate_next_frame(self,**kw):
            assert kw['frame_idx']==1 and kw['cache_frame_idx']==1
            events.append(('original_first',))
            return torch.full_like(kw['curr_vision_latent'],-10.)
        def denoise(self,*,data_batch_packed,memory):
            pack = data_batch_packed
            value = pack.vision.tokens[0]
            native = memory.native if isinstance(memory,BoundaryMemoryState) else memory
            memory.init({'_num_full_tokens':value.shape[2]},torch.device('cpu'))
            if isinstance(memory,BoundaryMemoryState):
                hist = memory.read_for_layer(0).gen_k_hist
                events.append(('boundary',native.frame_idx,value[:, :, :1].item(),hist.shape[1]))
                assert native.write_gen_cache is False
                # Absolute time is 4,8 and next targets remain 5:9,9:11.
                positions = pack.position_ids[:, pack.vision.sequence_indexes]
                start = 4 if native.frame_idx==2 else 8
                assert torch.allclose(positions,full_positions[:,start:start+value.shape[2]],atol=2e-6)
                if native.frame_idx==2:
                    assert value[:, :, :1].item() == -10.
                return {'preds_vision':native_velocity_items(pack,1.)}
            keys = value.flatten().reshape(1,-1,1,1)
            zero = torch.zeros(1,0,1,1)
            memory.write_for_layer(0,(keys,keys,zero,zero))
            events.append(('refresh',native.frame_idx,value.clone()))
            return {'preds_vision':native_velocity_items(pack,0.)}
    out, continuous = generate_boundary_latents(Model(),{},num_steps=3,history_mode=mode,return_reference=True)
    assert torch.equal(continuous,gt) and out.shape==gt.shape
    assert [x[0] for x in events].count('encode')==1
    assert [x[1] for x in events if x[0]=='refresh']==[0,1,2]
    assert torch.equal(out[:, :, 1:5],torch.full_like(out[:, :, 1:5],-10.))
    assert all(x[3]==(4 if x[1]==2 else 8) for x in events if x[0]=='boundary')
    for _,idx,value in (x for x in events if x[0]=='refresh' and x[1]>0):
        start = 1+(idx-1)*4
        source = gt[:, :, start:start+4] if mode=='gt' else out[:, :, start:start+4]
        assert torch.equal(value,refresh_latents(source,.02,seed=100042+start))
