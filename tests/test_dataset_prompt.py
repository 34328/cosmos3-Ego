import json

import pytest

from cosmos3_joint_video_hand_pose.src.dataset import CosmosActionPromptFormatter, build_prompt_text


def test_official_action_json_prompt_contract():
    prompt = CosmosActionPromptFormatter()("pick up the cup", frames=21, fps=30.0)
    structured = json.loads(prompt)

    assert structured["cinematography"]["framing"].startswith(
        "This video is captured from a first-person perspective"
    )
    assert structured["actions"] == [
        {"time": "0.00-0.70s", "description": "pick up the cup."}
    ]
    assert structured["duration"] == "0.70s"
    assert structured["fps"] == 30.0
    assert structured["resolution"] == {"H": 368, "W": 640}
    assert structured["aspect_ratio"] == "40,23"
    assert "ego" not in structured


def test_prompt_combines_episode_context_and_current_segment_explicitly():
    prompt = build_prompt_text(
        "scrub shoe with brush",
        "Cleaning a black shoe with a brush and cloth.",
        mode="episode_context_and_segment",
    )
    assert prompt == (
        "Overall task: Cleaning a black shoe with a brush and cloth. "
        "Current segment: scrub shoe with brush"
    )


def test_prompt_mode_rejects_empty_or_unknown_input():
    with pytest.raises(ValueError, match="both segment text"):
        build_prompt_text("", "")
    with pytest.raises(ValueError, match="unsupported prompt mode"):
        build_prompt_text("scrub shoe", "clean a shoe", mode="unknown")
