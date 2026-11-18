"""Loopback-only preview server with an explicit media allowlist and HTTP Range."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiohttp import web

from .build_page import load_catalog, media_names, load_gallery, relative_path


def create_app(catalog_path):
    catalog_path = Path(catalog_path).resolve()
    root = catalog_path.parent
    entries, default = load_catalog(catalog_path)
    allowed = {}
    for entry in entries:
        page_root = entry['selection_path'].parent
        selection_path = page_root / 'selection.json'
        load_gallery(selection_path)  # Fail on incomplete or inconsistent media.
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        for filename in ('index.html', 'results.html'):
            page = page_root / filename
            if not page.is_file() or not page.stat().st_size:
                raise FileNotFoundError('Build both pages before starting the preview server')
            allowed[page.relative_to(root).as_posix()] = page
        for sample in selection['windows']:
            directory = relative_path(page_root, sample['output_dir'])
            for filename in media_names(selection):
                path = directory / filename
                allowed[path.relative_to(root).as_posix()] = path

    async def serve(request):
        name = request.match_info['path']
        if name in ('', 'index.html', 'results.html'):
            page = name or 'index.html'
            base_url = next(entry['base_url'] for entry in entries if entry['path'] == default)
            raise web.HTTPFound(base_url + '/' + page)
        path = allowed.get(name)
        if path is None:
            raise web.HTTPNotFound()
        # FileResponse implements byte ranges and HEAD; never serve a broad directory.
        headers = {'Cache-Control': 'no-cache',
                   'X-Content-Type-Options': 'nosniff'}
        return web.FileResponse(path, headers=headers)

    app = web.Application()
    app.router.add_get('/{path:.*}', serve)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18768)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in [1,65535]')
    web.run_app(create_app(args.catalog), host='127.0.0.1', port=args.port, access_log=None)


if __name__ == '__main__':
    main()
