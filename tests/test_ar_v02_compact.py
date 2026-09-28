import pytest
import torch
from test_ar_v02_packing import make_pack
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout, VIDEO, ACTION
from cosmos3_joint_video_hand_pose.src.ar_v02_compact import compact_joint_targets, restore_joint_predictions


@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_compact_preserves_all_future_rows_rope_losses_and_scatter_gradient(c):
    layouts = [JointChunkLayout(n, 6, c) for n in (9, 17, 33)]
    full = make_pack(layouts, texts=[[3], [4, 5, 6], [7, 8]])
    for name in ("vision", "action"):
        for token in getattr(full, name).tokens:
            token.copy_(torch.arange(token.numel()).reshape_as(token))
    compact, maps = compact_joint_targets(full)
    assert compact.text_ids.numel() == 0
    assert not compact.action_state_mask.any() and not compact.vision_condition_type_mask.any()
    assert compact.sequence_length == sum((l.num_frames - 1) * 14 for l in layouts)
    for name in ("vision", "action"):
        src, dst = getattr(full, name), getattr(compact, name)
        torch.testing.assert_close(
            full.position_ids[:, src.mse_loss_indexes], compact.position_ids[:, dst.mse_loss_indexes], atol=0, rtol=0
        )
        torch.testing.assert_close(src.timesteps, dst.timesteps, atol=0, rtol=0)
        for i, l in enumerate(layouts):
            rows = (
                torch.where(l.video_metadata()[0] == VIDEO)[0]
                if name == "vision"
                else torch.where(l.action_metadata()[0] == ACTION)[0]
            )
            assert len(rows) == (l.num_frames - 1) * (1 if name == "vision" else 8)
            torch.testing.assert_close(dst.tokens[i], src.tokens[i].index_select(2 if name == "vision" else 0, rows))
    output = {
        "preds_" + name: [t.clone().requires_grad_() for t in getattr(compact, name).tokens]
        for name in ("vision", "action")
    }
    restored = restore_joint_predictions(output, full, maps)
    for name in ("vision", "action"):
        dim = 2 if name == "vision" else 0
        for p, r, idx in zip(output["preds_" + name], restored["preds_" + name], maps[name]):
            torch.testing.assert_close(r.index_select(dim, idx), p)
            r.sum().backward()
            torch.testing.assert_close(p.grad, torch.ones_like(p))
