"""Build the completed IT2V preview gallery; does not infer, render, or serve."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from urllib.parse import quote

PARAMETERS = ('checkpoint_step', 'seed', 'denoise_steps', 'guidance', 'context_sigma', 'fps')
MEDIA = ('preview.mp4', 'short_preview.mp4', 'generated.mp4', 'gt.mp4')
DECODER_MODES = ('predicted_prefix', 'gt_prefix_montage')


def media_names(selection):
    return tuple(name for name in MEDIA if name != 'short_preview.mp4') if selection.get('preview_mode') == 'full_segment' else MEDIA


def inline_json(value):
    return json.dumps(value, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')


def relative_path(root, value):
    if not isinstance(value, str) or not value or ':' in value or '\\' in value:
        raise ValueError('Expected a nonempty relative media directory')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Media must stay inside the preview directory')
    full = root / path
    full.resolve().relative_to(root.resolve())
    return full


def finite_number(value, key, *, minimum=0, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{key} must be a finite number')
    if value < minimum or (integer and int(value) != value):
        raise ValueError(f'Invalid {key}: {value}')
    return value


def load_gallery(selection_path):
    root = selection_path.parent
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    full_segment = selection.get('preview_mode') == 'full_segment'
    history_comparison = selection.get('history_comparison', False)
    if history_comparison and not full_segment:
        raise ValueError('History comparison requires complete segments')
    default_history = selection.get('default_history_mode', 'gt')
    if history_comparison and default_history not in ('gt', 'generated'):
        raise ValueError('default_history_mode must be gt or generated')
    for key in PARAMETERS:
        if key not in selection:
            raise ValueError(f'Missing selection.{key}')
        finite_number(selection[key], key, minimum=1 if key in ('fps', 'denoise_steps') else 0,
                      integer=key in ('checkpoint_step', 'seed', 'denoise_steps'))
    if selection['context_sigma'] > 1:
        raise ValueError('context_sigma must be in [0,1]')
    if not isinstance(selection.get('version'), str) or not selection['version']:
        raise ValueError('Missing selection.version')
    windows = selection.get('windows')
    if not isinstance(windows, list) or not windows:
        raise ValueError('selection.windows must be a nonempty list')
    samples, ids, directories, pairs = [], set(), set(), {}
    for index, window in enumerate(windows):
        if window.get('split') not in ('train', 'test'):
            raise ValueError('A window split must be train or test')
        sample_id = window.get('id')
        history_mode = window.get('history_mode', selection.get('history_mode', 'generated'))
        if history_mode not in ('gt', 'generated'):
            raise ValueError('history_mode must be gt or generated')
        if history_comparison and 'history_mode' not in window:
            raise ValueError('History comparison windows require history_mode')
        identity = (sample_id, history_mode) if history_comparison else sample_id
        if not isinstance(sample_id, str) or not sample_id or identity in ids:
            raise ValueError('Window IDs must be distinct nonempty strings')
        ids.add(identity)
        directory = relative_path(root, window['output_dir'])
        if directory.resolve() in directories:
            raise ValueError('Each sample must have its own output directory')
        directories.add(directory.resolve())
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        manifest_history = manifest.get('history_mode', manifest.get('history', 'generated'))
        if ('history_mode' in manifest and 'history' in manifest
                and manifest['history_mode'] != manifest['history']):
            raise ValueError(f'{sample_id}: manifest history fields disagree')
        if manifest_history != history_mode:
            raise ValueError(f'{sample_id}: history_mode differs from the displayed experiment')
        # Missing fields identify the original predicted-prefix decode. Never
        # silently relabel completed legacy RGB as the new GT-prefix montage.
        decoder_mode = manifest.get('decoder_mode', 'predicted_prefix')
        if decoder_mode not in DECODER_MODES:
            raise ValueError(f'{sample_id}: unknown decoder_mode')
        if decoder_mode == 'gt_prefix_montage' and history_mode != 'gt':
            raise ValueError(f'{sample_id}: GT-prefix decoding requires GT history')
        displayed_decoder = window.get('decoder_mode', selection.get('decoder_mode', decoder_mode))
        if displayed_decoder != decoder_mode:
            raise ValueError(f'{sample_id}: decoder_mode differs from the displayed experiment')
        # Missing manifests/media are failures, never rendered as a completed sample.
        for key in PARAMETERS:
            if key in manifest and manifest[key] != selection[key]:
                raise ValueError(f'{sample_id}: {key} differs from the displayed experiment')
        frames = finite_number(manifest['frames'], 'frames', minimum=1, integer=True)
        fps = finite_number(manifest['fps'], 'fps', minimum=1)
        duration = frames / fps
        start, short = None, None
        if full_segment:
            if window.get('length_group') not in ('short', 'long'):
                raise ValueError(f'{sample_id}: full segments require short/long length_group')
        else:
            start = finite_number(manifest['short_start_seconds'], 'short_start_seconds')
            short = finite_number(manifest['short_duration_seconds'], 'short_duration_seconds', minimum=1/fps)
            if start + short > duration + 1/fps:
                raise ValueError(f'{sample_id}: short clip extends outside the full rollout')
        media = {}
        for filename in media_names(selection):
            path = directory / filename
            path.resolve().relative_to(root.resolve())
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f'Incomplete preview media: {path}')
            revision = quote(f"{selection['version']}-{selection['checkpoint_step']}-{selection.get('preview_mode', 'window')}-{history_mode}-{decoder_mode}", safe='')
            media[filename] = quote(path.relative_to(root).as_posix(), safe='/') + '?v=' + revision
        caption = window.get('caption', manifest.get('caption', ''))
        if not isinstance(caption, str):
            raise ValueError('Caption must be text')
        if history_comparison:
            signature = (window['split'], window.get('title'), caption, frames, fps,
                         window.get('length_group'),
                         *(json.dumps(manifest.get(key), sort_keys=True) for key in
                           ('sample_id', 'segment_start_frame', 'segment_end_frame_exclusive',
                            'source_frame_indices', 'conditioning_caption')))
            pair = pairs.setdefault(sample_id, {})
            pair[history_mode] = signature
        samples.append(dict(split=window['split'], title=window.get('title') or f'片段 {index+1:02d}',
                            caption=caption, frames=frames, fps=fps, duration=duration,
                            short_start=start, short_duration=short,
                            length_group=window.get('length_group'), history_mode=history_mode,
                            decoder_mode=decoder_mode, media=media))
    if history_comparison:
        for sample_id, pair in pairs.items():
            if set(pair) != {'gt', 'generated'} or pair['gt'] != pair['generated']:
                raise ValueError(f'{sample_id}: history modes must pair the same complete segment')
    else:
        actual_modes = {sample['history_mode'] for sample in samples}
        default_history = next(iter(actual_modes)) if len(actual_modes) == 1 else 'generated'
        if ('default_history_mode' in selection
                and selection['default_history_mode'] != default_history):
            raise ValueError('default_history_mode differs from the displayed results')
    # Internal IDs, source hashes, checkpoint paths and raw manifest fields are not published.
    return dict(version=selection['version'], preview_mode='full_segment' if full_segment else 'window',
                history_comparison=history_comparison,
                default_history_mode=default_history,
                **{k: selection[k] for k in PARAMETERS}, samples=samples)


def build_page(selection_path):
    selection_path = Path(selection_path).resolve()
    gallery = load_gallery(selection_path)
    template = Path(__file__).with_name('index_template.html').read_text(encoding='utf-8')
    destination = selection_path.parent / 'index.html'
    content = template.replace('__GALLERY_JSON__', inline_json(gallery))
    temporary = destination.with_suffix('.html.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    print(build_page(args.selection))


if __name__ == '__main__':
    main()
