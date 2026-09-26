import numpy as np

from cosmos3_joint_video_hand_pose.src.materialize_visibility import project_palm_visibility


def test_project_palm_visibility_uses_head_pose_and_original_image_bounds():
    head_pose = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    palm_world = np.array(
        [
            [0, 0, 1],
            [4, 0, 1],
            [1, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.float64,
    )
    projection = np.array(
        [[100, 0, 320, 0], [0, 100, 180, 0], [0, 0, 1, 0]],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        project_palm_visibility(head_pose, palm_world, projection),
        np.array([1, 0, 1, 0], dtype=np.uint8),
    )
