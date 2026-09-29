import pytest
import torch

from cosmos3_joint_video_hand_pose.src.audit_dataloader import audit_batch


@pytest.mark.parametrize("wrapped", [False, True])
def test_audit_accepts_native_media_batch_wrapping(wrapped):
    frames = 5
    video = torch.zeros((3, frames, 368, 640), dtype=torch.uint8)
    action = torch.zeros((frames, 64))
    action_raw = torch.zeros((frames, 57))
    visibility = torch.ones((frames, 2), dtype=torch.bool)
    if wrapped:
        video = video.unsqueeze(0)
        action = action.unsqueeze(0)
        action_raw = action_raw.unsqueeze(0)
        visibility = visibility.unsqueeze(0)
    batch = {
        "text_token_ids": [torch.ones((1, 8), dtype=torch.long)],
        "video": [video],
        "action": [action],
        "action_raw": [action_raw],
        "hand_visibility": [visibility],
        "sequence_plan": [object()],
        "raw_action_dim": [torch.tensor(57)],
        "sample_id": ["episode:span:0:5"],
    }

    result = audit_batch(batch, cap=90_000)

    assert result["num_samples"] == 1
    assert result["frame_counts"] == [frames]
    assert result["sample_ids"] == ["episode:span:0:5"]


@pytest.mark.parametrize("frames", [5, 17])
def test_v02_audit_requires_explicit_k_and_keeps_state_out_of_future(frames):
    groups = (frames - 1) // 4
    rows = groups * 8
    batch = {
        "text_token_ids": [torch.ones((1, 8), dtype=torch.long)],
        "video": [torch.zeros((3, frames, 368, 640), dtype=torch.uint8)],
        "action": [torch.zeros((rows, 64))],
        "action_raw": [torch.zeros((rows, 57))],
        "hand_visibility": [torch.ones((rows, 2), dtype=torch.bool)],
        "sequence_plan": [object()],
        "raw_action_dim": [torch.tensor(57)],
        "ar_layout_version": ["joint_chunk_cond_v1"],
        "ar_boundary_states": [torch.zeros((groups, 64))],
    }
    with pytest.raises(ValueError, match="explicit v0.2 layout version or K"):
        audit_batch(batch, cap=60000)
    result = audit_batch(batch, cap=60000, action_tokens_per_latent=8)
    assert result["frame_counts"] == [frames]
    batch["action"][0] = torch.zeros((rows + 1, 64))
    with pytest.raises(ValueError, match="action must be"):
        audit_batch(batch, cap=60000, action_tokens_per_latent=8)
