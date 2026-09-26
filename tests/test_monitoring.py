import numpy as np

from cosmos3_joint_video_hand_pose.src.monitoring import _font, _wrap_prompt, project_f0_to_pixels, select_monitor_rows


def test_select_monitor_rows_uses_complete_aligned_segments_and_varied_lengths():
    rows = [
        {"split": "train", "manifest_task": "a", "episode_hash": "2", "span_index": "0", "start_idx": "0", "end_idx": "93"},
        {"split": "train", "manifest_task": "b", "episode_hash": "3", "span_index": "0", "start_idx": "0", "end_idx": "101"},
        {"split": "train", "manifest_task": "c", "episode_hash": "4", "span_index": "0", "start_idx": "0", "end_idx": "102"},
        {"split": "train", "manifest_task": "d", "episode_hash": "5", "span_index": "0", "start_idx": "0", "end_idx": "129"},
        {"split": "test", "manifest_task": "e", "episode_hash": "6", "span_index": "0", "start_idx": "0", "end_idx": "81"},
    ]
    selected = select_monitor_rows(rows, split="train", min_frames=81, max_frames=121, count=2)
    assert [row["manifest_task"] for row in selected] == ["a", "b"]
    assert [int(row["end_idx"]) - int(row["start_idx"]) for row in selected] == [93, 101]


def test_seeded_monitor_selection_is_reproducible_and_prefers_distinct_episodes():
    rows = [
        {
            "split": "train",
            "manifest_task": "brushing_shoes",
            "episode_hash": str(episode),
            "span_index": str(span),
            "start_idx": "0",
            "end_idx": str(frames),
        }
        for episode, span, frames in ((1, 0, 201), (1, 1, 205), (2, 0, 209), (3, 0, 213), (4, 0, 217))
    ]
    first = select_monitor_rows(rows, split="train", min_frames=201, max_frames=217, count=4, seed=42)
    second = select_monitor_rows(rows, split="train", min_frames=201, max_frames=217, count=4, seed=42)
    assert first == second
    assert len({row["episode_hash"] for row in first}) == 4


def test_project_f0_to_pixels_uses_predicted_headcam_extrinsics():
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    headcam = np.eye(4)
    headcam[0, 3] = 0.5
    intrinsics = np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 180.0], [0.0, 0.0, 1.0]])
    pixels, valid = project_f0_to_pixels(points, headcam, intrinsics)
    np.testing.assert_allclose(pixels[:2], [[270.0, 180.0], [370.0, 180.0]])
    np.testing.assert_array_equal(valid, [True, True, False])


def test_wrap_prompt_respects_pixel_width_without_spaces():
    font = _font(16)
    prompt = "把红色积木拿起来然后放到蓝色积木旁边" * 8
    lines = _wrap_prompt(prompt, font, max_width=180)
    assert len(lines) > 1
    assert "".join(lines) == prompt
    assert all(font.getlength(line) <= 180 for line in lines)
