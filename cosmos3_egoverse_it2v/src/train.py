from __future__ import annotations

import os
import runpy

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from . import config as _config  # noqa: E402,F401

if __name__ == "__main__":
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
