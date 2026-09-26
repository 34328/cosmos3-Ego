import torch

from cosmos3_joint_video_hand_pose.src.loss import visibility_weighted_action_flow_loss


def test_visibility_loss_renormalizes_groups_and_excludes_padding():
    pred = torch.zeros(3, 64, requires_grad=True)
    with torch.no_grad():
        pred[:, 0:9] = 1.0
        pred[:, 9:33] = 2.0
        pred[:, 33:57] = 3.0
        pred[:, 57:64] = 100.0
    target = torch.zeros_like(pred)
    condition = torch.tensor([[1.0], [0.0], [0.0]])
    visibility = torch.tensor([[False, False], [True, True], [False, True]])
    loss, metrics = visibility_weighted_action_flow_loss(
        pred=[pred],
        target=[target],
        condition_mask=[condition],
        visibility=[visibility],
        lambda_out_of_fov=0.0,
        subblock_equal_weight=True,
    )
    expected = (2 * 1.0 + 3 * 4.0 + 3 * 9.0) / 8
    torch.testing.assert_close(loss, torch.tensor(expected))
    assert metrics["right_wrist_translation_weight"].item() == 1
    assert metrics["right_wrist_rotation_weight"].item() == 1
    assert metrics["right_hand_latent_weight"].item() == 1
    assert metrics["left_wrist_translation_weight"].item() == 2
    assert metrics["left_wrist_rotation_weight"].item() == 2
    assert metrics["left_hand_latent_weight"].item() == 2
    loss.backward()
    assert torch.count_nonzero(pred.grad[0]).item() == 0
    assert torch.count_nonzero(pred.grad[2, 9:33]).item() == 0
    assert torch.count_nonzero(pred.grad[:, 57:64]).item() == 0


def test_fully_invisible_hand_group_is_omitted_not_zero_averaged():
    pred = torch.zeros(2, 64)
    pred[:, 0:9] = 1
    pred[:, 9:33] = 100
    pred[:, 33:57] = 3
    loss, metrics = visibility_weighted_action_flow_loss(
        pred=[pred],
        target=[torch.zeros_like(pred)],
        condition_mask=[torch.tensor([[1.0], [0.0]])],
        visibility=[torch.tensor([[False, False], [False, True]])],
        lambda_out_of_fov=0.0,
        subblock_equal_weight=True,
    )
    torch.testing.assert_close(loss, torch.tensor((2 * 1.0 + 3 * 9.0) / 5))
    assert metrics["active_action_blocks"].item() == 5


def test_lambda_changes_supervision_not_hand_loss_scale_for_constant_error():
    pred = torch.zeros(3, 64)
    pred[:, 9:33] = 2
    kwargs = dict(
        pred=[pred],
        target=[torch.zeros_like(pred)],
        condition_mask=[torch.tensor([[1.0], [0.0], [0.0]])],
        visibility=[torch.tensor([[False, False], [True, False], [False, False]])],
    )
    _, zero = visibility_weighted_action_flow_loss(
        **kwargs, lambda_out_of_fov=0.0, subblock_equal_weight=True
    )
    _, soft = visibility_weighted_action_flow_loss(
        **kwargs, lambda_out_of_fov=0.2, subblock_equal_weight=True
    )
    for block in ("right_wrist_translation", "right_wrist_rotation", "right_hand_latent"):
        torch.testing.assert_close(zero[f"{block}_loss"], torch.tensor(4.0))
        torch.testing.assert_close(soft[f"{block}_loss"], torch.tensor(4.0))


def test_time_weight_accepts_per_sample_scalar():
    pred = torch.ones(3, 64)
    target = torch.zeros_like(pred)
    kwargs = dict(
        pred=[pred],
        target=[target],
        condition_mask=[torch.zeros(3, 1)],
        visibility=[torch.ones(3, 2, dtype=torch.bool)],
        time_weight=lambda _sample, _frames, _ref: torch.tensor(2.0),
    )
    loss, _ = visibility_weighted_action_flow_loss(**kwargs)
    # Constant temporal weight scales the complete weighted action objective.
    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_samples_are_equally_averaged_instead_of_frame_weighted():
    short = torch.zeros(2, 64)
    short[1, :57] = 1.0
    long = torch.zeros(5, 64)
    long[1:, :57] = 3.0
    loss, metrics = visibility_weighted_action_flow_loss(
        pred=[short, long],
        target=[torch.zeros_like(short), torch.zeros_like(long)],
        condition_mask=[torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]])],
        visibility=[torch.ones(2, 2, dtype=torch.bool), torch.ones(5, 2, dtype=torch.bool)],
    )
    torch.testing.assert_close(metrics["per_sample_losses"], torch.tensor([1.0, 9.0]))
    torch.testing.assert_close(loss, torch.tensor(5.0))


def test_invisible_group_is_skipped_per_sample_only():
    first = torch.zeros(2, 64)
    first[1, 0:9] = 1.0
    first[1, 9:33] = 100.0
    first[1, 33:57] = 3.0
    second = torch.zeros(2, 64)
    second[1, 0:9] = 2.0
    second[1, 9:33] = 4.0
    second[1, 33:57] = 5.0
    loss, metrics = visibility_weighted_action_flow_loss(
        pred=[first, second],
        target=[torch.zeros_like(first), torch.zeros_like(second)],
        condition_mask=[torch.tensor([[1.0], [0.0]]), torch.tensor([[1.0], [0.0]])],
        visibility=[torch.tensor([[False, False], [False, True]]), torch.ones(2, 2, dtype=torch.bool)],
        lambda_out_of_fov=0.0,
        subblock_equal_weight=True,
    )
    first_expected = (2 * 1.0 + 3 * 9.0) / 5
    second_expected = (2 * 4.0 + 3 * 16.0 + 3 * 25.0) / 8
    torch.testing.assert_close(
        metrics["per_sample_losses"], torch.tensor([first_expected, second_expected])
    )
    torch.testing.assert_close(loss, torch.tensor((first_expected + second_expected) / 2))


def test_config_enables_cross_rank_sample_level_averaging():
    from cosmos3_joint_video_hand_pose.src.config import _model_config

    flow_config = _model_config()["rectified_flow_training_config"]
    assert flow_config["sample_level_loss_averaging"] is True
    assert flow_config["independent_action_schedule"] is False


def test_overfit_v0_0_config_is_the_single_joint_baseline():
    from cosmos3_joint_video_hand_pose.src.config import egoverse_joint_video_hand_pose_overfit_v0_0

    config = egoverse_joint_video_hand_pose_overfit_v0_0
    assert config["job"]["name"] == "overfit_v0.0"
    assert config["job"]["wandb_mode"] == "online"
    flow = config["model"]["config"]["rectified_flow_training_config"]
    assert flow["loss_scale"] == 10.0
    assert flow["action_loss_weight"] == 7.0
    assert config["model"]["subblock_equal_weight"] is True
    assert config["trainer"]["logging_iter"] == 1
    assert config["trainer"]["max_iter"] == 2000
    assert config["checkpoint"]["save_iter"] == 300
    assert config["model"]["config"]["parallelism"]["context_parallel_shard_degree"] == 2
    assert config["model"]["config"]["parallelism"]["data_parallel_shard_degree"] == 4
    callback_names = set(config["trainer"]["callbacks"])
    assert callback_names.isdisjoint({"training_stats", "param_count", "dataloader_speed"})


def test_wandb_loss_metric_sources_are_complete():
    from cosmos3_joint_video_hand_pose.src.wandb_metrics import extract_loss_metrics, filter_wandb_metrics

    output = {
        "egoverse_loss_video_raw": torch.tensor(1.0),
        "egoverse_loss_action_raw": torch.tensor(2.0),
        "egoverse_loss_video_weighted": torch.tensor(3.0),
        "egoverse_loss_action_weighted": torch.tensor(4.0),
        "egoverse_loss_total": torch.tensor(5.0),
    }
    metrics = extract_loss_metrics(output)
    assert set(metrics) == {
        "loss/video_raw",
        "loss/action_raw",
        "loss/total",
    }
    assert metrics["loss/total"].item() == 5.0
    lr_metrics = {
        "optim/lr_base": 1e-4,
        "optim/lr_action": 5e-4,
    }
    assert filter_wandb_metrics(metrics | lr_metrics | {"optim/lr": 1e-4}) == metrics | lr_metrics
