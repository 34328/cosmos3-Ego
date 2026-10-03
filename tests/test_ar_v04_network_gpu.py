"""V0.4 real packed network/noising/loss/history-gradient contract.

Reuse the reviewed native network fixture and its real loss assertions; this
does not substitute a small network for Nano-scale VAE/lifecycle validation.
Continuous encode/gather is separately covered in test_ar_v04_model.py.
"""
import pytest
import torch

from test_ar_v03_network_gpu import (
    test_noisy_chunk_receives_self_and_later_loss_gradients_through_full_network as _gradient_contract,
)
from cosmos3_joint_video_hand_pose.src.ar_v04_model import EgoVerseARV04Model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("prefix,mask", [
    pytest.param(False, False, id="original-sigma-no-mask"),
    pytest.param(True, False, id="prefix-sigma-no-mask"),
    pytest.param(True, True, id="prefix-numerator-only"),
])
def test_v04_real_loss_history_gradients(prefix, mask):
    _gradient_contract(EgoVerseARV04Model, prefix, mask)
