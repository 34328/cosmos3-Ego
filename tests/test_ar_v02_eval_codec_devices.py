"""A second CPU window must remain encodable after inference moves its codecs."""
from types import SimpleNamespace

import torch

from test_fixed_camera_codec import artifact
from cosmos3_joint_video_hand_pose.src.codec_fixed_camera import FrozenFixedCameraHandAE15
from cosmos3_joint_video_hand_pose.src.ar_v02_eval import _inference_hand_codecs


def test_inference_device_move_does_not_poison_next_cpu_window(artifact):
    path, points = artifact
    codecs = tuple(FrozenFixedCameraHandAE15(path, allow_unvalidated=True) for _ in range(2))
    raw = SimpleNamespace(fixed_hand_codecs=codecs)
    expected = tuple(c.encode(points) for c in codecs)
    inference = _inference_hand_codecs(raw)
    for source, decoder, encoded in zip(codecs, inference, expected):
        assert source is not decoder
        assert source.checkpoint_sha256 == decoder.checkpoint_sha256
        assert source.metadata == decoder.metadata
        torch.testing.assert_close(decoder.decode(encoded), source.decode(encoded), atol=0, rtol=0)
        for a, b in zip(source.buffers(), decoder.buffers()):
            assert a.data_ptr() != b.data_ptr()
        # Meta exercises a real nn.Module device migration without requiring a GPU.
        decoder.to('meta')
    for _ in range(2):
        for source, encoded in zip(codecs, expected):
            assert all(b.device.type == 'cpu' for b in source.buffers())
            torch.testing.assert_close(source.encode(points), encoded, atol=0, rtol=0)


def test_legacy_dataset_without_fixed_codecs_is_unchanged():
    assert _inference_hand_codecs(SimpleNamespace()) is None
