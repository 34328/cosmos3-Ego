import copy
import pytest
import torch
from cosmos_framework.model.generator.teacher_forcing import make_teacher_forcing_clean_pack
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos3_joint_video_hand_pose.src.ar_v02_layout import JointChunkLayout, VIDEO
from cosmos3_joint_video_hand_pose.src.ar_v02_packing import pack_joint_sequence
from cosmos3_joint_video_hand_pose.src.ar_v02_history_noise import history_noise_pack, sample_history_sigmas


def fixture(c, device="cpu"):
    layouts = [JointChunkLayout(2*c+2, 2, c), JointChunkLayout(c+1, 2, c)]
    data = GenerationDataClean(batch_size=2, is_image_batch=False,
        x0_tokens_vision=[torch.randn(1, 4, l.num_video_frames, 2, 4, device=device) for l in layouts],
        x0_tokens_action=[torch.randn(l.num_action_rows, 64, device=device) for l in layouts],
        fps_vision=torch.tensor([7.5, 7.5]), fps_action=torch.tensor([15., 15.]))
    pack = pack_joint_sequence(layout=layouts, gen_data_clean=data, text_ids=[[3,4], [5,6]],
        special_tokens={"eos_token_id":1, "start_of_generation":2},
        timesteps=[500,600], latent_patch_size=2, condition_frames=[(), ()])
    return layouts, pack


@pytest.mark.parametrize("c", [1,2,3,4])
def test_zero_identity_and_positive_only_video_no_gt_mutation(c):
    layouts, targets = fixture(c)
    saved = copy.deepcopy(targets)
    clean = make_teacher_forcing_clean_pack(targets)
    rng_before = torch.get_rng_state().clone()
    zeros, generator = sample_history_sigmas(layouts, probability=1, sigma_max=0, seed=42, device="cpu")
    assert history_noise_pack(clean, zeros, generator=generator, max_timestep=1000) is clean
    sigma, generator = sample_history_sigmas(layouts, probability=1, sigma_max=.2, seed=42, device="cpu")
    modified = history_noise_pack(clean, sigma, generator=generator, max_timestep=1000)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert modified.action is clean.action
    assert modified.text_ids is clean.text_ids
    cursor = 0
    for i, layout in enumerate(layouts):
        roles, chunks, _ = layout.video_metadata()
        u, v = roles != VIDEO, roles == VIDEO
        assert torch.equal(modified.vision.tokens[i][:,:,u], clean.vision.tokens[i][:,:,u])
        assert not torch.equal(modified.vision.tokens[i][:,:,v], clean.vision.tokens[i][:,:,v])
        assert torch.equal(targets.vision.tokens[i], saved.vision.tokens[i])
        assert torch.equal(targets.action.tokens[i], saved.action.tokens[i])
        for k in chunks[v].unique():
            assert sigma[i][chunks == k][roles[chunks == k] == VIDEO].unique().numel() == 1
        n = int(v.sum()) * layout.vision_tokens
        assert torch.equal(modified.vision.timesteps[cursor:cursor+n], (sigma[i][v]*1000).repeat_interleave(layout.vision_tokens))
        cursor += n
    assert torch.equal(targets.vision.mse_loss_indexes, saved.vision.mse_loss_indexes)
    assert torch.equal(targets.vision.timesteps, saved.vision.timesteps)


def test_replay_sample_gate_and_rng_isolation():
    layouts, _ = fixture(4)
    a, _ = sample_history_sigmas(layouts*1000, probability=.5, sigma_max=.2, seed=42, device="cpu")
    b, _ = sample_history_sigmas(layouts*1000, probability=.5, sigma_max=.2, seed=42, device="cpu")
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert .45 < sum(bool(x.any()) for x in a)/len(a) < .55


def test_fixed_window_whitelist_preserves_audit():
    from cosmos3_joint_video_hand_pose.src.ar_dataset import select_fixed_windows
    rows=[dict(episode_hash="e",span_index="0",start_idx="0",end_idx="600",_clip_frames=273,_valid_starts=[0,8])]
    items=[dict(sample_id="e:0:0:600", start=8, frames=273)]
    assert select_fixed_windows(rows,items)[0]["_valid_starts"] == [8]
    assert rows[0]["_valid_starts"] == [0,8]
    with pytest.raises(ValueError):
        select_fixed_windows(rows,[dict(items[0], start=7)])


def test_noise_recipe_only_changes_explicit_augmentation_and_job_identity():
    from omegaconf import OmegaConf
    from cosmos3_joint_video_hand_pose.src.config import _ar_v02_fixed_camera_experiment, _history_noise_experiment
    base, noise = _ar_v02_fixed_camera_experiment(), _history_noise_experiment()
    assert noise.model.history_video_noise_prob == .5
    assert noise.model.history_video_noise_sigma_max == .2
    noise.model.history_video_noise_prob = base.model.history_video_noise_prob
    noise.model.history_video_noise_sigma_max = base.model.history_video_noise_sigma_max
    noise.job = base.job
    assert OmegaConf.to_yaml(noise) == OmegaConf.to_yaml(base)


def test_condition_hook_does_not_touch_target_pass(monkeypatch):
    from types import SimpleNamespace
    from cosmos3_joint_video_hand_pose.src.ar_v02_model import EgoVerseARV02Model
    from cosmos3_joint_video_hand_pose.src.ar_model import EgoVerseARModel
    layouts, target = fixture(4)
    model = object.__new__(EgoVerseARV02Model)
    torch.nn.Module.__init__(model)
    model._joint_layouts, model._joint_layout = layouts, None
    model.history_video_noise_prob, model.history_video_noise_sigma_max = 1., .2
    model.ar_seed, model._history_noise_iteration = 42, 7
    model.rectified_flow_video = SimpleNamespace(noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)))
    model.config = SimpleNamespace(compact_noisy_training=False)
    monkeypatch.setattr(EgoVerseARModel, "denoise", lambda self, net, pack, memory, causal: pack)
    clean = make_teacher_forcing_clean_pack(target)
    assert model.denoise(data_batch_packed=clean, memory=SimpleNamespace(pass_number=1)) is not clean
    assert model.denoise(data_batch_packed=target, memory=SimpleNamespace(pass_number=2)) is target
    model.eval()
    assert model.denoise(data_batch_packed=clean, memory=SimpleNamespace(pass_number=1)) is clean


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@torch.no_grad()
def test_zero_sigma_bitwise_official_network_output(monkeypatch):
    from test_ar_v02_network_gpu import make_network, adapter_memory
    layouts, target = fixture(4, "cuda")
    # Exercise real official timestep embedding, projection and attention modules.
    net = make_network().eval()
    clean = make_teacher_forcing_clean_pack(target)
    clean.to_cuda()
    zeros, gen = sample_history_sigmas(layouts, probability=1, sigma_max=0, seed=42, device="cuda")
    changed = history_noise_pack(clean, zeros, generator=gen, max_timestep=1000)
    assert changed is clean
    target.to_cuda()
    def run(condition):
        memory = adapter_memory(monkeypatch, target)
        net(condition, memory=memory)
        memory.pass_number = 2
        return net(target, memory=memory)
    original, zero = run(clean), run(changed)
    for key in ("preds_vision", "preds_action"):
        assert all(torch.equal(a,b) for a,b in zip(original[key],zero[key]))
    sigmas, gen = sample_history_sigmas(layouts, probability=1, sigma_max=.2, seed=42, device="cuda")
    augmented = history_noise_pack(clean, sigmas, generator=gen, max_timestep=1000)
    positive = run(augmented)
    assert all(torch.isfinite(x).all() for k in ("preds_vision", "preds_action") for x in positive[k])
    assert not torch.equal(original["preds_action"][0], positive["preds_action"][0])
    # Current clean target is not visible to its own noisy chunk.
    first = layouts[0].boundaries[0]
    torch.testing.assert_close(original["preds_action"][0][:first.action_count+1],
        positive["preds_action"][0][:first.action_count+1], atol=0, rtol=0)
