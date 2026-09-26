from cosmos3_joint_video_hand_pose.src.wandb_compat import ensure_wandb_generate_id


def test_wandb_generate_id_compatibility_helper_is_callable():
    import wandb

    ensure_wandb_generate_id()
    assert callable(wandb.util.generate_id)
    assert len(wandb.util.generate_id()) == 8
