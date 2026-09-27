"""CPU tests for the lingbot-va teacher-forcing visibility rules."""

import pytest
import torch

from cosmos3_joint_video_hand_pose.src.ar_attention import (
    ARChunkLayout,
    frame_chunk_ids,
    make_mask_mod,
    token_block_ids,
)


def test_chunk_partition_keeps_frame_zero_alone_and_allows_short_tail():
    assert frame_chunk_ids(9, 3).tolist() == [0, 1, 1, 1, 2, 2, 2, 3, 3]
    assert frame_chunk_ids(5, 1).tolist() == [0, 1, 2, 3, 4]
    assert ARChunkLayout(9, 2, 3, 3).num_chunks == 4


def test_block_ids_follow_supertoken_order():
    layout = ARChunkLayout(num_frames=3, action_tokens=2, vision_tokens=3, chunk_size=1)
    assert token_block_ids(layout).tolist() == [1, 1, 0, 0, 0, 3, 3, 2, 2, 2, 5, 5, 4, 4, 4]


def _roles(layout):
    chunks = frame_chunk_ids(layout.num_frames, layout.chunk_size)
    roles = []
    for frame in range(layout.num_frames):
        roles += [(int(chunks[frame]), "a")] * layout.action_tokens + [(int(chunks[frame]), "v")] * layout.vision_tokens
    return roles


def _grid(layout, *, noisy, window, text_pad=4, text_len=3, gen_pad=None):
    gen_pad = gen_pad or layout.num_tokens
    ids = torch.full((gen_pad,), -1, dtype=torch.int64)
    ids[: layout.num_tokens] = token_block_ids(layout)
    mask_mod = make_mask_mod(ids, gen_pad_len=gen_pad, text_pad_len=text_pad, text_len=text_len, noisy=noisy, window=window)
    kv_len = text_pad + gen_pad * (2 if noisy else 1)
    q = torch.arange(gen_pad)[:, None]
    kv = torch.arange(kv_len)[None, :]
    return mask_mod(0, 0, q, kv)


def _block_id(chunk, role):
    return 2 * chunk + (1 if role == "a" else 0)


def _expected_clean(qc, qr, kc, kr, window):
    qi, ki = _block_id(qc, qr), _block_id(kc, kr)
    return ki <= qi and (window is None or qi - ki <= window)


def _expected_noisy(qc, qr, kc, kr, key_clean, window):
    qi, ki = _block_id(qc, qr), _block_id(kc, kr)
    relation = ki < qi if key_clean else ki == qi
    return relation and (window is None or abs(qi - ki) <= window)


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
@pytest.mark.parametrize("window", [None, 2, 4])
def test_noisy_pass_matches_lingbot_rules(chunk_size, window):
    layout = ARChunkLayout(num_frames=6, action_tokens=2, vision_tokens=3, chunk_size=chunk_size, window=window)
    grid = _grid(layout, noisy=True, window=window)
    roles = _roles(layout)
    n = layout.num_tokens
    for q, (qc, qr) in enumerate(roles):
        assert grid[q, :3].all() and not grid[q, 3].item()  # real text only
        for k, (kc, kr) in enumerate(roles):
            assert grid[q, 4 + k].item() == _expected_noisy(qc, qr, kc, kr, False, window), (q, k)
            assert grid[q, 4 + n + k].item() == _expected_noisy(qc, qr, kc, kr, True, window), (q, k)


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
@pytest.mark.parametrize("window", [None, 3])
def test_clean_pass_matches_lingbot_rules(chunk_size, window):
    layout = ARChunkLayout(num_frames=6, action_tokens=2, vision_tokens=3, chunk_size=chunk_size, window=window)
    grid = _grid(layout, noisy=False, window=window)
    roles = _roles(layout)
    for q, (qc, qr) in enumerate(roles):
        for k, (kc, kr) in enumerate(roles):
            assert grid[q, 4 + k].item() == _expected_clean(qc, qr, kc, kr, window), (q, k)


def test_design_table_semantics():
    """Doc section 4: noisy V_k never sees A_k; noisy A_k sees clean V_k; nothing sees the future."""
    layout = ARChunkLayout(num_frames=5, action_tokens=2, vision_tokens=3, chunk_size=2)
    grid = _grid(layout, noisy=True, window=None)
    roles = _roles(layout)
    n = layout.num_tokens
    for q, (qc, qr) in enumerate(roles):
        for k, (kc, kr) in enumerate(roles):
            noisy_ok, clean_ok = grid[q, 4 + k].item(), grid[q, 4 + n + k].item()
            if kc > qc:
                assert not noisy_ok and not clean_ok
            if qr == "v" and kc == qc and kr == "a":
                assert not noisy_ok and not clean_ok
            if qr == "a" and kc == qc and kr == "v":
                assert clean_ok and not noisy_ok
            if kc < qc:
                assert clean_ok and not noisy_ok


def test_padding_rows_are_isolated():
    layout = ARChunkLayout(num_frames=2, action_tokens=2, vision_tokens=3, chunk_size=1)
    n, pad = layout.num_tokens, 16
    for noisy in (False, True):
        grid = _grid(layout, noisy=noisy, window=None, gen_pad=pad)
        # Real queries never read padded GEN keys.
        assert not grid[:n, 4 + n : 4 + pad].any()
        # Padded queries read exactly the padded keys of their own stream.
        assert not grid[n:, :4].any() and not grid[n:, 4 : 4 + n].any()
        assert grid[n:, 4 + n : 4 + pad].all()
        if noisy:
            assert not grid[n:, 4 + pad :].any()


def test_layout_validation():
    with pytest.raises(ValueError):
        ARChunkLayout(0, 8, 240, 4)
    with pytest.raises(ValueError):
        ARChunkLayout(3, 8, 240, 4, window=-1)
