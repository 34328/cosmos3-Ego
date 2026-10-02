"""Serve a rendered gallery and its media over a private, forwardable HTTP port."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiohttp import web


def create_app(directory: Path) -> web.Application:
    root = directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    allowed = {"index.html"}
    for sample in manifest["samples"]:
        allowed.update(sample[key] for key in ("full_video", "short_video", "poster"))
        if sample.get("reference_video"):
            allowed.add(sample["reference_video"])

    async def serve(request: web.Request) -> web.FileResponse:
        relative = request.match_info["path"] or "index.html"
        path = (root / relative).resolve()
        if relative not in allowed or not path.is_relative_to(root) or not path.is_file():
            raise web.HTTPNotFound()
        # Native FileResponse supports byte ranges, so video seeking need not fetch a full file.
        return web.FileResponse(path, headers={"Cache-Control": "no-cache"})

    app = web.Application()
    app.router.add_get("/{path:.*}", serve)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18766)
    args = parser.parse_args()
    web.run_app(create_app(args.directory), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
