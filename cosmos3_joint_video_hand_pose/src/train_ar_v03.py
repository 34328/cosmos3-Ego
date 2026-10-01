"""Register AR V0.3, then delegate to the official Cosmos training CLI."""
from __future__ import annotations

import os
import runpy

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from . import ar_v03_config as _config  # noqa: E402,F401


def main() -> None:
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
