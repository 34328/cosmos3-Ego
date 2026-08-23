from __future__ import annotations


def ensure_wandb_generate_id() -> None:
    import wandb

    if hasattr(wandb.util, "generate_id"):
        return
    from wandb.sdk.lib.runid import generate_id

    wandb.util.generate_id = generate_id
