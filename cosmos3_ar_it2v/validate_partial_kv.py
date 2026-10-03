"""One real-Nano, forward-only VAE/partial-chunk/cache validation gate."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--toml', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--subset-root', type=Path, required=True)
    args = p.parse_args()
    args.subset_root = args.subset_root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(hostname=socket.gethostname(), pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   checkpoint=args.checkpoint, cases=[], status='running', started_unix=time.time(),
                   comparison='same fixed xt and sigma=.5, clean first latent, no stochastic refresh',
                   relative_rmse_limit=.03, cosine_minimum=.999)
    def save():
        (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2))
    save()
    start = time.monotonic()
    initialized = False
    try:
        import torch
        from cosmos_framework.utils import distributed
        from cosmos_framework.utils.lazy_config import instantiate
        from cosmos_framework.data.generator.sequence_packing.autoregressive import pack_input_sequence_autoregressive
        from cosmos_framework.data.generator.sequence_packing.modality import compute_text_split_length
        from cosmos_framework.model.generator.utils.kv_cache import DualKVCache
        from cosmos3_ar_it2v.inference import load_model, training_layout_batch, chunk_ranges, cache_chunk_index
        from cosmos3_ar_it2v.model import latent_valid_weights
        distributed.init()
        initialized = True
        model, config = load_model(args.toml, args.checkpoint)
        ds_cfg = next(iter(config.dataloader_train.dataloader.datasets.values())).dataset
        dataset = instantiate(ds_cfg, segments_manifest=str(args.subset_root/'segments.csv'),
            sample_mode='full_segment', max_sequence_length=65536,
            segment_statistics_path=str(args.subset_root/'summary.json'),
            iterable_shuffle=False, random_window=False, cfg_dropout_rate=0.)
        indexes = [next(i for i,r in enumerate(dataset.rows) if int(r['end_idx'])-int(r['start_idx'])==n)
                   for n in (96,187)]
        torch.manual_seed(42)
        for index in indexes:
            case_start = time.monotonic()
            sample = dataset.get_item_at_window(index)
            batch = training_layout_batch(sample)
            true_frames = int(sample['video_true_num_frames'])
            padded_frames = sample['video'].shape[1]
            case = dict(sample_id=sample['sample_id'],true_frames=true_frames,padded_frames=padded_frames,
                temporal_padding=int(sample['video_temporal_padding']),chunks=[],status='running')
            receipt['cases'].append(case)
            save()
            print(json.dumps(dict(event='case_start',true_frames=true_frames)),flush=True)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                clean = model.get_data_and_condition(batch, vision_condition_indexes=[[0]])
                latent = clean.x0_tokens_vision[0].to(**model.tensor_kwargs)
                expected_t = 1+(true_frames-1+3)//4
                assert latent.shape[2]==expected_t
                case['latent_shape']=list(latent.shape)
                decoded = model.decode(latent)
                assert decoded.shape[2]==padded_frames and torch.isfinite(decoded).all()
                assert decoded[:,:,:true_frames].shape[2]==len(sample['source_frame_indices'])
                case['decoded_shape']=list(decoded.shape)
                case['cropped_frames']=true_frames
                case['active_latent_weight']=latent_valid_weights(true_frames,expected_t).sum().item()
                del decoded
                cond, _ = model._get_inference_text_tokens(batch, False)
                text = cond[0]
                offset = compute_text_split_length(len(text),model.llm_special_tokens,has_generation=True)
                expert=model.config.diffusion_expert_config
                patch=expert.patch_spatial
                fps=clean.fps_vision.tolist()
                tcf=model.tokenizer_vision_gen.temporal_compression_factor
                max_t=model.rectified_flow_video.noise_scheduler.config.num_train_timesteps
                generator=torch.Generator(device=latent.device).manual_seed(42)
                eps=torch.randn(latent.shape,device=latent.device,dtype=torch.float32,generator=generator)
                xt=(.5*latent.float()+.5*eps).to(**model.tensor_kwargs)
                xt[:,:,:1]=latent[:,:,:1]
                def pack(value, start, include_text):
                    packed=pack_input_sequence_autoregressive(vision_latent=value,action_latent=None,
                        text_tokens=text if include_text else None,timestep=.5*max_t,
                        fps_vision=fps,fps_action=None,special_tokens=model.llm_special_tokens,
                        latent_patch_size=patch,condition_frame_indexes_vision=[0] if start==0 else [],
                        condition_frame_indexes_action=[],frame_idx=start,temporal_compression_factor=tcf,
                        video_temporal_causal=True,action_dim=model.config.max_action_dim,
                        enable_fps_modulation=expert.enable_fps_modulation,base_fps=expert.base_fps,
                        cached_text_offset=None if include_text else offset,
                        unified_3d_mrope_temporal_modality_margin=expert.unified_3d_mrope_temporal_modality_margin,
                        force_action_tokens=False)
                    packed.to_cuda()
                    return packed
                full=pack(xt,0,True)
                full_positions=full.position_ids[:,full.vision.sequence_indexes].clone()
                memory=model.build_memory_state(full,{})
                expected=model.denoise(data_batch_packed=full,memory=memory)['preds_vision'][0].float()
                case['native_prediction_shape']=list(expected.shape)
                if expected.ndim==5 and expected.shape[0]==1:
                    expected=expected[0]
                assert expected.ndim==4 and expected.shape[1]==expected_t
                del full,memory
                caches=[DualKVCache(gen_cache_size=None,preallocate_ring=False) for _ in range(model.net.num_hidden_layers)]
                patches=math.ceil(latent.shape[3]/patch)*math.ceil(latent.shape[4]/patch)
                history=(model.config.local_attention_frames-model.config.frames_per_chunk)*patches
                all_differences=[]
                all_reference=[]
                for a,b in chunk_ranges(expected_t,model.config.frames_per_chunk):
                    current=pack(xt[:,:,a:b],a,a==0)
                    pos=current.position_ids[:,current.vision.sequence_indexes]
                    pos_error=(pos-full_positions[:,a*patches:b*patches]).abs().max().item()
                    # Native large modality offsets (~15000) have FP32 ULP
                    # around .001; equivalent addition orders need not match
                    # below one ULP. Keep this separate from network tolerance.
                    rope_tolerance=2*torch.finfo(pos.dtype).eps*max(1.,pos.abs().max().item())
                    assert pos_error<=rope_tolerance
                    state=model.build_memory_state(current,dict(dual_kv_cache=caches,
                        frame_idx=cache_chunk_index(a,model.config.frames_per_chunk),write_gen_cache=True,
                        use_ar_rolling=False,transfer_history_sink_tokens=0,transfer_history_max_tokens=history))
                    got=model.denoise(data_batch_packed=current,memory=state)['preds_vision'][0].float()
                    if got.ndim==5 and got.shape[0]==1:
                        got=got[0]
                    ref=expected[:,a:b]
                    assert got.shape==ref.shape and torch.isfinite(got).all(), (got.shape,ref.shape)
                    error=got-ref
                    rel=(error.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-8)).item()
                    cosine=torch.nn.functional.cosine_similarity(got.reshape(1,-1),ref.reshape(1,-1)).item()
                    item=dict(start=a,frames=b-a,rope_max_error=pos_error,rope_tolerance=rope_tolerance,
                        max_abs=error.abs().max().item(),
                        mean_abs=error.abs().mean().item(),relative_rmse=rel,cosine=cosine)
                    case['chunks'].append(item)
                    all_differences.append(error.detach().cpu())
                    all_reference.append(ref.detach().cpu())
                    save()
                    print(json.dumps(dict(event='chunk',true_frames=true_frames,**item)),flush=True)
                    if a>0 and (rel>=receipt['relative_rmse_limit'] or cosine<receipt['cosine_minimum']):
                        raise AssertionError(f'Whole-pass/cache mismatch at chunk {a}: {item}')
                    del current,state,got,ref,error
                error=torch.cat(all_differences,dim=1)
                ref=torch.cat(all_reference,dim=1)
                case.update(status='passed',relative_rmse=(error.square().mean().sqrt()/ref.square().mean().sqrt()).item(),
                    max_abs=error.abs().max().item(),mean_abs=error.abs().mean().item(),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,elapsed_seconds=time.monotonic()-case_start)
                save()
                del latent,xt,eps,expected,caches,batch,sample,clean,error,ref
                torch.cuda.empty_cache()
        receipt['status']='passed'
    except BaseException:
        receipt['status']='failed'
        receipt['error']=traceback.format_exc()
        if receipt['cases'] and receipt['cases'][-1]['status']=='running':
            receipt['cases'][-1]['status']='failed'
        raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-start
        save()
        if initialized:
            distributed.destroy_process_group()


if __name__=='__main__':
    main()
