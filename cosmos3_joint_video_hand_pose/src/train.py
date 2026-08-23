"""Register the EgoVerse experiment, then delegate to Cosmos' training CLI."""

from __future__ import annotations

import os
import runpy

# Dynamic packed shapes otherwise leave several GiB reserved but unusable near
# the H800 memory boundary. This allocator is also used by Cosmos cookbooks.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from . import config as _config  # noqa: F401


def main() -> None:
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
