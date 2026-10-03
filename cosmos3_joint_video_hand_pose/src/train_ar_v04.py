"""Register V0.4 and delegate to the official Cosmos training CLI."""
import os
import runpy

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
from . import ar_v04_config as _config  # noqa: E402,F401


def main():
    runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
