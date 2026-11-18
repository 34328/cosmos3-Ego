"""Build the completed IT2V preview gallery; does not infer, render, or serve."""
from __future__ import annotations

import argparse
import hashlib
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


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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
    samples, ids, directories, pairs, pair_indices = [], set(), set(), {}, {}
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
        samples.append(dict(pair_index=pair_indices.setdefault(sample_id, len(pair_indices)),
                            split=window['split'], title=window.get('title') or f'片段 {index+1:02d}',
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


def public_metrics(report_path, selection_path):
    """Publish display values only; input hashes and pooled sums stay private."""
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['selection_sha256'] != sha256_file(selection_path):
        raise ValueError('Metrics belong to a different selection')
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    windows = {(row['id'], row['history_mode']): row for row in selection['windows']}
    identities = [(row['id'], row['history_mode']) for row in report['samples']]
    if len(set(identities)) != len(identities) or set(identities) != windows.keys():
        raise ValueError('Metrics must cover every displayed history result exactly once')
    for row in report['samples']:
        window = windows[(row['id'], row['history_mode'])]
        directory = relative_path(selection_path.parent, window['output_dir'])
        for name, filename in (('manifest', 'manifest.json'), ('gt', 'gt.mp4'),
                               ('prediction', 'generated.mp4')):
            if sha256_file(directory / filename) != row['source'][name]['sha256']:
                raise ValueError(f"{row['id']}: {filename} changed after metrics were computed")
    def stats(value):
        return {part: dict(psnr_db=value[part]['psnr_db'],
                           **{scope: {key: value[part][scope][key]
                                      for key in ('direction_cosine', 'motion_ratio')}
                              for scope in ('total', 'within_chunk')})
                for part in ('full', 'early', 'late')}
    return dict(samples=[dict(id=row['id'], history_mode=row['history_mode'], stats=stats(row['stats']))
                 for row in report['samples']],
        groups=[dict(split=row['split'], history_mode=row['history_mode'],
                     samples_count=row['samples_count'], stats=stats(row['stats']))
                for row in report['groups']])


def load_catalog(catalog_path):
    catalog_path = Path(catalog_path).resolve()
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    entries, identities = [], set()
    for item in catalog['galleries']:
        root = relative_path(catalog_path.parent, item['path'])
        selection_path = root / 'selection.json'
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        identity = (selection['version'], selection['checkpoint_step'])
        if identity in identities:
            raise ValueError('Each version/checkpoint must have one published gallery')
        identities.add(identity)
        entries.append(dict(path=item['path'], selection_path=selection_path,
                            metrics_path=relative_path(root, item['metrics']) if item.get('metrics') else None,
                            version=selection['version'], checkpoint_step=selection['checkpoint_step'],
                            base_url='/' + quote(root.relative_to(catalog_path.parent).as_posix(), safe='/')))
    if catalog['default'] not in {entry['path'] for entry in entries}:
        raise ValueError('Catalog default must identify a published gallery')
    return entries, catalog['default']


def run_selector(gallery, navigation):
    data = dict(version=gallery['version'], checkpoint_step=gallery['checkpoint_step'],
                entries=navigation)
    return '''<div class="run-selectors" aria-label="选择模型与训练步数">
<label>模型版本<select id="model-version" aria-label="模型版本"></select></label>
<label>训练步数<select id="training-step" aria-label="训练步数"></select></label>
</div>
<script id="run-catalog" type="application/json">''' + inline_json(data) + '''</script>
<script>
{
  const catalog=JSON.parse(document.querySelector('#run-catalog').textContent);
  const version=document.querySelector('#model-version'),step=document.querySelector('#training-step');
  const page=location.pathname.endsWith('results.html')?'results.html':'index.html';
  function option(select,value,label){const item=document.createElement('option');item.value=value;item.textContent=label;select.append(item);}
  for(const value of new Set(catalog.entries.map(entry=>entry.version)))option(version,value,value.replace(/^ar_it2v_/,''));
  version.value=catalog.version;
  function fillSteps(selected){step.replaceChildren();for(const entry of catalog.entries.filter(entry=>entry.version===version.value).sort((a,b)=>a.checkpoint_step-b.checkpoint_step))option(step,entry.checkpoint_step,entry.checkpoint_step+' 步');if([...step.options].some(item=>item.value===String(selected)))step.value=selected;}
  function navigate(){const entry=catalog.entries.find(entry=>entry.version===version.value&&entry.checkpoint_step===Number(step.value));location.href=entry.base_url+'/'+page;}
  fillSteps(catalog.checkpoint_step);
  version.addEventListener('change',()=>{fillSteps(step.value);navigate();});
  step.addEventListener('change',navigate);
}
</script>'''


def build_page(selection_path, metrics_path=None, *, navigation=None):
    selection_path = Path(selection_path).resolve()
    gallery = load_gallery(selection_path)
    if metrics_path is not None:
        metrics = public_metrics(Path(metrics_path), selection_path)
        keyed = {(row['id'], row['history_mode']): row['stats'] for row in metrics['samples']}
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        for window, sample in zip(selection['windows'], gallery['samples']):
            sample['metrics'] = keyed[(window['id'], sample['history_mode'])]
        gallery['metrics'] = dict(groups=metrics['groups'])
    navigation = navigation or [dict(version=gallery['version'], checkpoint_step=gallery['checkpoint_step'], base_url='.')]
    selector = run_selector(gallery, navigation)
    data = inline_json(gallery)
    for filename, template_name in (('results.html', 'index_template.html'), ('index.html', 'home_template.html')):
        template = Path(__file__).with_name(template_name).read_text(encoding='utf-8')
        destination = selection_path.parent / filename
        temporary = destination.with_suffix('.html.tmp')
        temporary.write_text(template.replace('__GALLERY_JSON__', data).replace('__RUN_SELECTOR__', selector), encoding='utf-8')
        temporary.replace(destination)
    return destination


def build_catalog(catalog_path):
    entries, _ = load_catalog(catalog_path)
    navigation = [{key: entry[key] for key in ('version', 'checkpoint_step', 'base_url')} for entry in entries]
    return [build_page(entry['selection_path'], entry['metrics_path'], navigation=navigation) for entry in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--selection', type=Path)
    source.add_argument('--catalog', type=Path, help='Explicit published version/checkpoint galleries')
    parser.add_argument('--metrics', type=Path, help='Completed metrics report for this exact selection')
    args = parser.parse_args()
    if args.catalog:
        if args.metrics:
            parser.error('--metrics belongs to --selection; catalog entries specify their own metrics')
        for destination in build_catalog(args.catalog):
            print(destination)
    else:
        print(build_page(args.selection, args.metrics))


if __name__ == '__main__':
    main()
