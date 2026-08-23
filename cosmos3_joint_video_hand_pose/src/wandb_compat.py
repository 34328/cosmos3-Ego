"""Small compatibility shim for the W&B version installed with Cosmos."""

from __future__ import annotations


def ensure_wandb_generate_id() -> None:
    """Restore the public helper expected by Cosmos on newer W&B releases."""
    import wandb

    if hasattr(wandb.util, "generate_id"):
        return
    from wandb.sdk.lib.runid import generate_id

    wandb.util.generate_id = generate_id

