"""Loopback-only preview server with an explicit media allowlist and HTTP Range."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiohttp import web

from .build_page import media_names, load_gallery, relative_path


def create_app(root, *, extra_pages=()):
    root = Path(root).resolve()
    allowed = {}
    for page_root in [root, *(relative_path(root, name) for name in extra_pages)]:
        selection_path = page_root / 'selection.json'
        load_gallery(selection_path)  # Fail on incomplete or inconsistent media.
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        index = page_root / 'index.html'
        if not index.is_file() or not index.stat().st_size:
            raise FileNotFoundError('Build index.html before starting the preview server')
        allowed[index.relative_to(root).as_posix()] = index
        for sample in selection['windows']:
            directory = relative_path(page_root, sample['output_dir'])
            for filename in media_names(selection):
                path = directory / filename
                allowed[path.relative_to(root).as_posix()] = path

    async def serve(request):
        name = request.match_info['path'] or 'index.html'
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
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18768)
    parser.add_argument('--extra-page', action='append', default=[],
                        help='Allow one completed gallery below root; never serve its directory')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in [1,65535]')
    web.run_app(create_app(args.root, extra_pages=args.extra_page), host='127.0.0.1', port=args.port, access_log=None)


if __name__ == '__main__':
    main()
