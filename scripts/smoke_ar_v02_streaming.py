"""Full Nano streaming wiring smoke. Does not claim numerical/latency acceptance."""

import argparse
import json
from pathlib import Path
import time
import torch
from check_ar_v02_nano_cache import load_model, DEFAULT_CKPT, ROOT, fingerprint
from ar_v02_nano_validation import prepare_real_sampler
from cosmos3_joint_video_hand_pose.src.ar_v02_streaming import StreamingJointSampler
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import ACTION
from cosmos3_joint_video_hand_pose.src.ar_chunk_state import decode_chunk_camera_state


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--toml", type=Path, default=ROOT / "cosmos3_joint_video_hand_pose/configs/ar_v0_2.toml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="running", scope="streaming_wiring_smoke_only", full_acceptance=False, source=fingerprint(), chunks=[]
    )
    try:
        torch.set_num_threads(1)
        model, ds_cfg = load_model(args)
        for c in (1, 2, 3, 4):
            base, fixture = prepare_real_sampler(model, ds_cfg, c, ROOT)
            v = base.gt_video
            stream = StreamingJointSampler(
                model,
                text_ids=base.text[0],
                latent_shape=(v.shape[1], v.shape[3], v.shape[4]),
                chunk_size=c,
                state_normalizer=base.state_normalizer,
                future_normalizer=base.future_normalizer,
                source_fps=base.source_fps,
                history="gt",
                initial_temporal_offset=base.memory_info["initial_temporal_offset"],
            )
            for b, frames in zip(base.layout.boundaries, (c, 1)):
                vi = base.layout.video_indexes(b.chunk_id, False)[:frames].to(v.device)
                ui = base.layout.video_indexes(b.chunk_id, True).to(v.device)
                rows = torch.where((base.roles == ACTION) & (base.chunks == b.chunk_id))[0][: frames * 8]
                state = decode_chunk_camera_state(
                    base.gt_states[b.latent_start - 1], base.state_normalizer, source_index=b.source_start
                )
                result = stream.step(
                    v[:, :, ui], state, gt_video=v[:, :, vi], gt_action=base.gt_action[rows], frames=frames
                )
                assert result.action.shape == (frames * 8, 64)
                assert result.report["forward_calls"] == 32
                assert torch.isfinite(result.action).all() and torch.isfinite(result.video).all()
                assert torch.isfinite(result.decoded.rigid_chunk).all()
                record = dict(C=c, frames=frames, fixture=fixture, **result.report)
                report["chunks"].append(record)
                print(json.dumps(record), flush=True)
                del result
            del stream, base
            torch.cuda.empty_cache()
        report["status"] = "success"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["finished"] = time.time()
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
