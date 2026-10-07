"""CPU checks for paired complete-segment history modes and media isolation."""
import asyncio
import json
import pytest
from aiohttp.test_utils import TestClient, TestServer

from visualization.build_page import build_page, load_gallery
from visualization.serve import create_app


def make_gallery(root, *, comparison=True):
    selection = dict(version='ar_it2v_v0.2', checkpoint_step=1500, seed=42,
                     denoise_steps=35, guidance=1.0, context_sigma=.02, fps=30,
                     preview_mode='full_segment', windows=[])
    if comparison:
        selection.update(history_comparison=True, default_history_mode='gt')
    for mode in (('generated', 'gt') if comparison else ('generated',)):
        directory = root / ('full_segments' if mode == 'generated' else 'gt_history') / 'train_01'
        directory.mkdir(parents=True)
        manifest = dict(frames=33, fps=30, history=mode, sample_id='internal-sample-id',
                        segment_start_frame=100, segment_end_frame_exclusive=133,
                        source_frame_indices=list(range(100, 133)))
        if mode == 'gt':
            manifest['history_mode'] = mode
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        for filename in ('preview.mp4', 'generated.mp4', 'gt.mp4'):
            (directory / filename).write_bytes(b'0123456789')
        window = dict(id='train_01', split='train', title='拿起并放下杯子', caption='pick up cup',
                      length_group='short', output_dir=str(directory.relative_to(root)))
        if comparison:
            window['history_mode'] = mode
        selection['windows'].append(window)
    path = root / 'selection.json'
    path.write_text(json.dumps(selection))
    return path


def test_paired_modes_default_gt_and_do_not_publish_internal_identity(tmp_path):
    path = make_gallery(tmp_path)
    gallery = load_gallery(path)
    assert gallery['default_history_mode'] == 'gt'
    assert [s['history_mode'] for s in gallery['samples']] == ['generated', 'gt']
    assert [s['decoder_mode'] for s in gallery['samples']] == ['predicted_prefix', 'predicted_prefix']
    assert gallery['samples'][0]['media']['preview.mp4'].startswith('full_segments/train_01/')
    assert gallery['samples'][1]['media']['preview.mp4'].startswith('gt_history/train_01/')
    html = build_page(path).read_text()
    assert '真实视频历史' in html and '模型生成历史' in html
    assert '当前目标块不会获得其真实图像' in html
    assert 'internal-sample-id' not in html
    assert 'segment_start_frame' not in html and 'source_frame_indices' not in html


@pytest.mark.parametrize('mutation', ['missing_pair', 'same_mode', 'wrong_caption',
                                     'wrong_source', 'wrong_history', 'conflicting_history'])
def test_refuse_mislabeled_or_unpaired_results(tmp_path, mutation):
    path = make_gallery(tmp_path)
    selection = json.loads(path.read_text())
    if mutation == 'missing_pair':
        selection['windows'].pop()
    elif mutation == 'same_mode':
        selection['windows'][1]['history_mode'] = 'generated'
    elif mutation == 'wrong_caption':
        selection['windows'][1]['caption'] = 'different action'
    else:
        manifest_path = tmp_path / 'gt_history/train_01/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if mutation == 'wrong_source':
            manifest['sample_id'] = 'different-sample'
        elif mutation == 'wrong_history':
            manifest.pop('history_mode')
            manifest['history'] = 'generated'
        else:
            manifest['history'] = 'generated'
        manifest_path.write_text(json.dumps(manifest))
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        load_gallery(path)


def test_original_generated_only_gallery_is_compatible(tmp_path):
    path = make_gallery(tmp_path, comparison=False)
    gallery = load_gallery(path)
    assert not gallery['history_comparison']
    assert gallery['default_history_mode'] == 'generated'
    assert len(gallery['samples']) == 1
    assert build_page(path).is_file()


def test_new_matched_gt_and_existing_generated_results_have_real_decoder_labels(tmp_path):
    path = make_gallery(tmp_path)
    manifest_path = tmp_path / 'gt_history/train_01/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['decoder_mode'] = 'gt_prefix_montage'
    manifest_path.write_text(json.dumps(manifest))
    selection = json.loads(path.read_text())
    selection['windows'][1]['decoder_mode'] = 'gt_prefix_montage'
    path.write_text(json.dumps(selection))
    gallery = load_gallery(path)
    assert [s['decoder_mode'] for s in gallery['samples']] == ['predicted_prefix', 'gt_prefix_montage']
    assert 'gt_prefix_montage' in gallery['samples'][1]['media']['preview.mp4']
    html = build_page(path).read_text()
    assert '匹配真实前缀解码' in html and '单块诊断拼图' in html
    assert '原预测前缀解码' in html
    assert json.loads(manifest_path.read_text()) == manifest  # Build never rewrites evidence.


@pytest.mark.parametrize('mutation', ['unknown', 'null', 'generated_gt_prefix',
                                     'window_mismatch', 'selection_mismatch'])
def test_refuse_decoder_mislabeling_including_legacy_upgrade(tmp_path, mutation):
    path = make_gallery(tmp_path)
    selection = json.loads(path.read_text())
    if mutation in ('unknown', 'null', 'generated_gt_prefix'):
        mode = 'full_segments' if mutation == 'generated_gt_prefix' else 'gt_history'
        manifest_path = tmp_path / mode / 'train_01/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['decoder_mode'] = {'unknown': 'independent_blocks', 'null': None,
                                   'generated_gt_prefix': 'gt_prefix_montage'}[mutation]
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == 'window_mismatch':
        selection['windows'][1]['decoder_mode'] = 'gt_prefix_montage'
    else:
        selection['decoder_mode'] = 'gt_prefix_montage'
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        load_gallery(path)


def test_gt_only_page_infers_real_history_and_rejects_explicit_default_mismatch(tmp_path):
    path = make_gallery(tmp_path)
    selection = json.loads(path.read_text())
    selection.pop('history_comparison')
    selection.pop('default_history_mode')
    selection['windows'] = selection['windows'][1:]
    path.write_text(json.dumps(selection))
    gallery = load_gallery(path)
    assert gallery['default_history_mode'] == 'gt'
    assert gallery['samples'][0]['decoder_mode'] == 'predicted_prefix'
    selection['default_history_mode'] = 'generated'
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match='default_history_mode'):
        load_gallery(path)


def test_generated_preview_never_encodes_or_uses_gt():
    from types import SimpleNamespace
    import torch
    from visualization.preview import decode_preview
    predicted = torch.ones(1, 1, 7, 1, 1)
    calls = []
    def decode(value):
        calls.append(value.clone())
        return torch.ones(1, 3, 25, 1, 1)
    def forbidden_gt(*args, **kwargs):
        raise AssertionError('generated history must never request GT')
    model = SimpleNamespace(tensor_kwargs={'device': 'cpu'}, decode=decode,
                            get_data_and_condition=forbidden_gt)
    result, mode = decode_preview(model, {'video': 'unavailable'}, predicted,
                                 history_mode='generated', true_frames=23)
    assert mode == 'predicted_prefix' and result.shape == (1, 3, 23, 1, 1)
    assert len(calls) == 1 and torch.equal(calls[0], predicted)


def test_gt_preview_uses_official_complete_encode_and_only_matched_decoder(monkeypatch):
    from types import SimpleNamespace
    import torch
    from visualization import decoder_diagnostic
    from visualization.preview import decode_preview
    predicted = torch.ones(1, 1, 7, 1, 1)
    gt = predicted.clone()
    gt[:, :, 1:] = 8
    calls = []
    batch = {'video': 'complete original segment'}
    def encode(value, *, vision_condition_indexes):
        assert value is batch and vision_condition_indexes is None
        calls.append('complete_encode')
        return SimpleNamespace(batch_size=1, x0_tokens_action=None, x0_tokens_vision=[gt])
    def matched(model, actual_predicted, actual_gt, **kwargs):
        assert actual_predicted is predicted and torch.equal(actual_gt, gt)
        assert kwargs == {'true_frames': 23, 'frames_per_chunk': 4}
        calls.append('matched_decode')
        return torch.ones(1, 3, 23, 1, 1)
    def forbidden_comparison(*args, **kwargs):
        raise AssertionError('preview must not decode unused A/C variants')
    model = SimpleNamespace(tensor_kwargs={'device': 'cpu'}, get_data_and_condition=encode,
                            decode=forbidden_comparison)
    monkeypatch.setattr(decoder_diagnostic, 'decode_gt_prefix', matched)
    result, mode = decode_preview(model, batch, predicted, history_mode='gt', true_frames=23)
    assert calls == ['complete_encode', 'matched_decode']
    assert mode == 'gt_prefix_montage' and result.shape[2] == 23


@pytest.mark.parametrize('escape', ['relative', 'symlink'])
def test_media_cannot_escape_parent_gallery(tmp_path, escape):
    path = make_gallery(tmp_path)
    selection = json.loads(path.read_text())
    if escape == 'relative':
        selection['windows'][0]['output_dir'] = '../outside'
    else:
        (tmp_path / 'escape').symlink_to(tmp_path.parent, target_is_directory=True)
        selection['windows'][0]['output_dir'] = 'escape/outside'
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        load_gallery(path)


def test_paired_server_only_serves_selected_media_with_ranges(tmp_path):
    path = make_gallery(tmp_path)
    build_page(path)
    (tmp_path / 'private.log').write_text('private')

    async def check():
        async with TestClient(TestServer(create_app(tmp_path))) as client:
            assert (await client.get('/')).status == 200
            for mode in ('full_segments', 'gt_history'):
                response = await client.get(f'/{mode}/train_01/preview.mp4', headers={'Range': 'bytes=2-4'})
                assert response.status == 206
                assert await response.read() == b'234'
                assert (await client.head(f'/{mode}/train_01/generated.mp4')).status == 200
                assert (await client.get(f'/{mode}/train_01/manifest.json')).status == 404
            for name in ('selection.json', 'private.log', 'full_segments/', 'gt_history/'):
                assert (await client.get('/' + name)).status == 404

    asyncio.run(check())


def test_extra_gallery_preserves_old_page_and_private_files(tmp_path):
    build_page(make_gallery(tmp_path))
    nested = tmp_path / 'boundary_diagnostic'
    nested.mkdir()
    build_page(make_gallery(nested, comparison=False))
    (nested / 'latents.pt').write_bytes(b'private')

    async def check():
        async with TestClient(TestServer(create_app(tmp_path, extra_pages=['boundary_diagnostic']))) as client:
            assert (await client.get('/')).status == 200
            assert (await client.get('/boundary_diagnostic/index.html')).status == 200
            response = await client.get('/boundary_diagnostic/full_segments/train_01/preview.mp4',
                                        headers={'Range': 'bytes=2-4'})
            assert response.status == 206 and await response.read() == b'234'
            for name in ('selection.json', 'latents.pt', 'full_segments/train_01/manifest.json'):
                assert (await client.get('/boundary_diagnostic/' + name)).status == 404

    asyncio.run(check())
    with pytest.raises(ValueError):
        create_app(tmp_path, extra_pages=['../outside'])
