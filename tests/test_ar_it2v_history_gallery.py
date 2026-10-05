"""CPU checks for paired complete-segment history modes and media isolation."""
import asyncio
import json
import pytest
from aiohttp.test_utils import TestClient, TestServer

from cosmos3_ar_it2v.visualization.build_page import build_page, load_gallery
from cosmos3_ar_it2v.visualization.serve import create_app


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
