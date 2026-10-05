"""CPU checks of prefix ownership and absolute causal-VAE output cropping."""
from types import SimpleNamespace

import pytest
import torch

from cosmos3_ar_it2v.visualization.decoder_diagnostic import (
    LATENT_FORMAT, decode_comparison, load_latent_archive, rgb_frame_range, save_latent_archive,
)


class RecordingDecoder:
    tensor_kwargs = {"device": "cpu"}

    def __init__(self):
        self.calls = []
        self.tokenizer_vision_gen = SimpleNamespace(
            is_causal=True, temporal_compression_factor=4, _keep_decoder_cache=False,
        )

    def decode(self, latent):
        self.calls.append(latent.clone())
        # A deliberately history-sensitive causal stub, not a replacement VAE.
        accumulated = latent[:, :1].cumsum(dim=2)
        frames = torch.cat((accumulated[:, :, :1], accumulated[:, :, 1:].repeat_interleave(4, dim=2)), dim=2)
        return frames.repeat(1, 3, 1, 1, 1)


def latent_pair(length=7, dtype=torch.float32):
    gt = torch.arange(1, length + 1, dtype=dtype).reshape(1, 1, length, 1, 1)
    predicted = gt.clone()
    predicted[:, :, 1:] += 100
    return predicted, gt


def test_absolute_frame_ranges_singleton_first_and_partial_block():
    assert rgb_frame_range(0, 1) == (0, 1)
    assert rgb_frame_range(1, 5) == (1, 17)
    assert rgb_frame_range(5, 7) == (17, 25)
    with pytest.raises(ValueError):
        rgb_frame_range(5, 5)


def test_same_predictions_complete_gt_prefix_no_target_or_future_and_true_crop():
    predicted, gt = latent_pair()
    original_predicted, original_gt = predicted.clone(), gt.clone()
    model = RecordingDecoder()
    result = decode_comparison(model, predicted, gt, true_frames=23)
    assert all(v.shape == (1, 3, 23, 1, 1) for v in result.values())
    assert len(model.calls) == 5  # A; B's singleton, full block, partial block; C.
    assert torch.equal(model.calls[0], predicted)
    for actual, (start, end) in zip(model.calls[1:-1], [(0, 1), (1, 5), (5, 7)]):
        assert actual.shape[2] == end
        assert torch.equal(actual[:, :, :start], gt[:, :, :start])
        assert torch.equal(actual[:, :, start:end], predicted[:, :, start:end])
    assert torch.equal(model.calls[-1], gt)
    assert torch.equal(predicted, original_predicted) and torch.equal(gt, original_gt)
    # The first future block has the same first-latent prefix in A and B.
    assert torch.equal(result["predicted_prefix"][:, :, :17], result["gt_prefix_montage"][:, :, :17])
    # Next block uses GT history, then the SAME predictions; the partial tail keeps six true frames.
    assert (result["gt_prefix_montage"][:, :, 17:21] == gt[:, :, :5].sum() + predicted[:, :, 5].item()).all()
    assert (result["gt_prefix_montage"][:, :, 21:] == gt[:, :, :5].sum() + predicted[:, :, 5:].sum()).all()
    assert not torch.equal(result["predicted_prefix"][:, :, 17:], result["gt_prefix_montage"][:, :, 17:])


def test_gt_target_change_cannot_change_earlier_block_diagnostic():
    predicted, gt = latent_pair(length=11)
    baseline = decode_comparison(RecordingDecoder(), predicted, gt, true_frames=41)
    altered_gt = gt.clone()
    altered_gt[:, :, 5:] += 10000
    altered = decode_comparison(RecordingDecoder(), predicted, altered_gt, true_frames=41)
    assert torch.equal(baseline["gt_prefix_montage"][:, :, :33], altered["gt_prefix_montage"][:, :, :33])
    assert torch.equal(baseline["predicted_prefix"], altered["predicted_prefix"])
    assert not torch.equal(baseline["gt_prefix_montage"][:, :, 33:], altered["gt_prefix_montage"][:, :, 33:])


def test_when_predictions_are_gt_all_three_decodes_match():
    _, gt = latent_pair()
    result = decode_comparison(RecordingDecoder(), gt, gt, true_frames=23)
    assert torch.equal(result["predicted_prefix"], result["gt_prefix_montage"])
    assert torch.equal(result["predicted_prefix"], result["gt_reconstruction"])


def test_decoder_cache_scope_is_rejected():
    predicted, gt = latent_pair()
    model = RecordingDecoder()
    model.tokenizer_vision_gen._keep_decoder_cache = True
    with pytest.raises(ValueError, match="cached decoder scope"):
        decode_comparison(model, predicted, gt, true_frames=23)
    assert not model.calls


@pytest.mark.parametrize("true_frames", [0, 21, 26])
def test_inconsistent_length_is_rejected(true_frames):
    predicted, gt = latent_pair()
    with pytest.raises(ValueError, match="true"):
        decode_comparison(RecordingDecoder(), predicted, gt, true_frames=true_frames)


def test_latent_archive_preserves_normalized_values_dtype_and_provenance(tmp_path):
    predicted, gt = latent_pair(dtype=torch.bfloat16)
    path = tmp_path / "latents.pt"
    metadata = save_latent_archive(path, predicted, gt, true_frames=23,
        metadata={"seed": 42, "history_mode": "gt", "checkpoint": "actual-checkpoint"})
    archive = load_latent_archive(path)
    assert archive["format"] == LATENT_FORMAT
    assert torch.equal(archive["predicted_latents"], predicted)
    assert torch.equal(archive["gt_latents"], gt)
    assert archive["predicted_latents"].dtype == torch.bfloat16
    assert archive["metadata"] == metadata
    assert metadata["video_temporal_padding"] == 2
    assert metadata["latent_chunk_ranges"] == [[0, 1], [1, 5], [5, 7]]
    assert metadata["rgb_chunk_ranges"] == [[0, 1], [1, 17], [17, 23]]
    assert metadata["seed"] == 42 and metadata["history_mode"] == "gt"
    assert metadata["gt_prefix_montage_is_continuous_rollout"] is False
    with pytest.raises(FileExistsError):
        save_latent_archive(path, predicted, gt, true_frames=23)


def test_mixed_dtype_roundtrip_and_decode_preserve_sampler_output(tmp_path):
    predicted, gt = latent_pair()
    predicted[:, :, 1:] += 0.125  # These sampler values would be lost in BF16.
    gt = gt.to(torch.bfloat16)
    path = tmp_path / "mixed.pt"
    metadata = save_latent_archive(path, predicted, gt, true_frames=23)
    archive = load_latent_archive(path)
    assert archive["predicted_latents"].dtype == torch.float32
    assert archive["gt_latents"].dtype == torch.bfloat16
    assert torch.equal(archive["predicted_latents"], predicted)
    assert torch.equal(archive["gt_latents"], gt)
    assert metadata["latent_dtype"] == "torch.float32"
    assert metadata["gt_latent_dtype"] == "torch.bfloat16"
    model = RecordingDecoder()
    result = decode_comparison(model, archive["predicted_latents"], archive["gt_latents"], true_frames=23)
    assert model.calls[0].dtype == torch.float32
    assert torch.equal(model.calls[0], predicted)
    assert model.calls[-1].dtype == torch.bfloat16
    assert all(value.shape[2] == 23 for value in result.values())
    archive["metadata"]["gt_latent_dtype"] = "torch.float32"
    invalid = tmp_path / "invalid_dtype.pt"
    torch.save(archive, invalid)
    with pytest.raises(ValueError, match="disagrees"):
        load_latent_archive(invalid)


def test_old_or_inconsistent_archive_is_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"generated": torch.zeros(3, 17, 1, 1)}, path)
    with pytest.raises(ValueError, match="explicit continuous"):
        load_latent_archive(path)
    predicted, gt = latent_pair()
    current = tmp_path / "current.pt"
    save_latent_archive(current, predicted, gt, true_frames=23)
    payload = torch.load(current, weights_only=True)
    payload["metadata"]["aligned_frames"] = 26
    torch.save(payload, path)
    with pytest.raises(ValueError, match="disagrees"):
        load_latent_archive(path)
