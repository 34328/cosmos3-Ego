import torch

from cosmos3_joint_video_hand_pose.src.inference import _default_to_regular_weights, build_egoverse_wam_batch


def test_wam_inference_batch_conditions_on_first_video_and_action_only():
    batch = build_egoverse_wam_batch(
        video=torch.zeros((3, 1, 360, 640), dtype=torch.uint8),
        initial_action=torch.arange(57, dtype=torch.float32),
        prompt="move both hands",
        frames=49,
        fps=30.0,
        device="cpu",
    )

    video = batch["video"][0][0]
    action = batch["action"][0][0]
    plan = batch["sequence_plan"][0]
    assert video.shape == (3, 49, 368, 640)
    assert action.shape == (49, 64)
    torch.testing.assert_close(action[0, :57], torch.arange(57, dtype=torch.float32))
    assert torch.count_nonzero(action[1:]).item() == 0
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == [0]
    assert plan.action_start_frame_offset == 0
    assert int(batch["raw_action_dim"][0]) == 57
    assert int(batch["domain_id"][0]) == 3
    torch.testing.assert_close(
        batch["image_size"].cpu(),
        torch.tensor([[368, 640, 368, 640]], dtype=torch.float32),
    )


def test_project_inference_defaults_to_regular_weights():
    argv = ["inference", "--checkpoint-path", "checkpoint"]
    _default_to_regular_weights(argv)
    assert argv[-1] == "--no-use-ema-weights"

    explicit = ["inference", "--use-ema-weights"]
    _default_to_regular_weights(explicit)
    assert explicit == ["inference", "--use-ema-weights"]
